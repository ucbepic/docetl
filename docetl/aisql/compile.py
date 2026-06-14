"""Compile an AI-SQL query into an ordered list of execution stages.

v1 handles::

    SELECT <items, some ai_*(...) AS alias>
    FROM <source>
    [WHERE <relational> AND <ai_filter(...)> AND <ai_score(...) > k>]

lowering to a sequence of DuckDB (relational) and DocETL (semantic)
stages threaded over Arrow. The WHERE split is the design's selection
pushdown across the boundary: relational conjuncts run first in DuckDB
(file pushdown), AI filters and score-comparisons shrink the set before
the SELECT-list maps, so the LLM runs on as few rows as the query allows.

WHERE conjuncts are routed by kind:

  - relational (no AI)            → the DuckDB scan's WHERE
  - ``ai_filter(col, 'q')``       → a DocETL ``filter``
  - ``ai_fn(col, 'q') <cmp> v``   → a DocETL ``map`` (hidden value column)
                                    + a DuckDB residual filter over it

Deferred (clear errors): ``OR``/``NOT`` around an AI predicate, an AI
function on both sides of a comparison, and AI functions in
GROUP BY / JOIN / ORDER (later milestones).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from docetl.aisql.functions import (
    ALL_FUNCTIONS,
    AIFunctionCall,
    build_compare_map_op,
    build_filter_op,
    build_map_op,
)

PREV = "_prev"  # name under which a stage's output is registered for the next
_COMPARISONS = (exp.GT, exp.LT, exp.GTE, exp.LTE, exp.EQ, exp.NEQ)


@dataclass
class ScoreSpec:
    """An AI function used inside a WHERE comparison: a map op computing
    a hidden column, plus the residual relational predicate over it."""

    op: dict
    residual_sql: str
    hidden: str


@dataclass
class RelationalStage:
    """A SQL string for DuckDB. ``reads_prev`` means it queries the prior
    stage's output (registered as ``_prev``) rather than source files."""

    sql: str
    reads_prev: bool = False


@dataclass
class SemanticStage:
    """DocETL operations applied to the prior stage's rows."""

    operations: list[dict] = field(default_factory=list)


@dataclass
class CompiledQuery:
    stages: list  # RelationalStage | SemanticStage, in execution order


def _ai_calls_in(node: exp.Expression) -> list[exp.Anonymous]:
    """AI function nodes anywhere under *node* (custom names parse as
    Anonymous functions in sqlglot)."""
    return [a for a in node.find_all(exp.Anonymous) if a.name.lower() in ALL_FUNCTIONS]


def _parse_call(fn: exp.Anonymous, alias: str) -> AIFunctionCall:
    args = list(fn.expressions)
    if not args or not isinstance(args[0], exp.Column):
        raise NotImplementedError(
            f"{fn.name}(...) must take a column as its first argument in v1"
        )
    column = args[0].name
    literals = tuple(a.this for a in args[1:] if isinstance(a, exp.Literal))
    return AIFunctionCall(
        name=fn.name.lower(), column=column, literals=literals, alias=alias
    )


def compile_sql(query: str) -> CompiledQuery:
    parsed = sqlglot.parse_one(query, read="duckdb")
    if not isinstance(parsed, exp.Select):
        raise NotImplementedError("only SELECT queries are supported")

    for label, cls in (
        ("GROUP BY", exp.Group),
        ("HAVING", exp.Having),
        ("JOIN", exp.Join),
        ("ORDER BY", exp.Order),
    ):
        node = parsed.find(cls)
        if node is not None and _ai_calls_in(node):
            raise NotImplementedError(f"AI functions in {label} are not yet supported")

    # Split WHERE into relational conjuncts (DuckDB), ai_filter conjuncts
    # (DocETL filters), and comparisons over an AI function (a map that
    # computes a column + a residual relational predicate over it).
    where = parsed.args.get("where")
    relational_where, filter_calls, score_specs = _split_where(where)

    # Pair each SELECT item with the AI call it contains (if any).
    select_ai: list[tuple[exp.Expression, AIFunctionCall | None]] = []
    for item in parsed.expressions:
        calls = _ai_calls_in(item)
        if len(calls) > 1:
            raise NotImplementedError("one AI function per SELECT item in v1")
        if calls:
            alias = item.alias_or_name
            if not alias or isinstance(item, exp.Anonymous):
                raise NotImplementedError("AI function in SELECT needs an AS alias")
            select_ai.append((item, _parse_call(calls[0], alias)))
        else:
            select_ai.append((item, None))

    has_select_ai = any(call for _, call in select_ai)
    if not has_select_ai and not filter_calls and not score_specs:
        # Pure relational — hand the whole query to DuckDB unchanged.
        return CompiledQuery(stages=[RelationalStage(sql=parsed.sql(dialect="duckdb"))])

    # Stage 1: scan + the relational half of WHERE, project everything.
    scan = exp.select("*").from_(parsed.find(exp.From).this)
    if relational_where is not None:
        scan = scan.where(relational_where)
    stages: list = [RelationalStage(sql=scan.sql(dialect="duckdb"))]

    filter_ops = [
        build_filter_op(call, name=f"aifilter_{i}")
        for i, call in enumerate(filter_calls)
    ]
    select_ops = [
        build_map_op(call, name=f"ai_{call.alias}")
        for _, call in select_ai
        if call is not None
    ]

    if score_specs:
        # WHERE ai_filters + the score maps; then a DuckDB stage applies
        # the residual comparisons (and drops the hidden score columns),
        # so the SELECT-list maps run only on the survivors.
        stages.append(
            SemanticStage(operations=filter_ops + [s.op for s in score_specs])
        )
        residual = " AND ".join(s.residual_sql for s in score_specs)
        hidden = ", ".join(s.hidden for s in score_specs)
        stages.append(
            RelationalStage(
                sql=f"SELECT * EXCLUDE ({hidden}) FROM {PREV} WHERE {residual}",
                reads_prev=True,
            )
        )
        if select_ops:
            stages.append(SemanticStage(operations=select_ops))
    else:
        # ai_filters first (they shrink), then the SELECT-list maps.
        stages.append(SemanticStage(operations=filter_ops + select_ops))

    # Final stage: the original SELECT list with AI calls replaced by
    # their alias column, over the semantic output.
    final = exp.select(
        *[_final_select_item(item, call) for item, call in select_ai]
    ).from_(PREV)
    stages.append(RelationalStage(sql=final.sql(dialect="duckdb"), reads_prev=True))

    return CompiledQuery(stages=stages)


def _final_select_item(item: exp.Expression, call: AIFunctionCall | None):
    if call is None:
        return item
    return exp.column(call.alias)


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    """Flatten a top-level AND chain into its conjuncts."""
    if isinstance(node, exp.Paren):
        return _conjuncts(node.this)
    if isinstance(node, exp.And):
        return _conjuncts(node.left) + _conjuncts(node.right)
    return [node]


def _split_where(
    where: exp.Where | None,
) -> tuple[exp.Expression | None, list[AIFunctionCall], list[ScoreSpec]]:
    """Partition a WHERE into (relational predicate, ai_filter calls,
    AI-comparison specs).

    Supports a top-level ``AND`` whose conjuncts are each: relational (no
    AI), a bare ``ai_filter(col, 'q')``, or a comparison with an AI
    function on one side (``ai_score(col, 'q') > 0.8``). An AI function
    inside an ``OR``/``NOT``, on both sides of a comparison, or otherwise
    nested raises — those remain unsupported.
    """
    if where is None:
        return None, [], []

    relational: list[exp.Expression] = []
    filters: list[AIFunctionCall] = []
    scores: list[ScoreSpec] = []
    for conj in _conjuncts(where.this):
        if not _ai_calls_in(conj):
            relational.append(conj)
        elif isinstance(conj, exp.Anonymous) and conj.name.lower() == "ai_filter":
            filters.append(_parse_call(conj, alias="keep"))
        elif isinstance(conj, _COMPARISONS):
            scores.append(_parse_comparison(conj, idx=len(scores)))
        else:
            raise NotImplementedError(
                "in WHERE, AI functions are supported as a top-level "
                "ai_filter(col, 'q') conjunct or a comparison like "
                "ai_score(col, 'q') > 0.8; OR/NOT and other nestings are not"
            )

    relational_pred = None
    if relational:
        relational_pred = relational[0]
        for extra in relational[1:]:
            relational_pred = exp.and_(relational_pred, extra)
    return relational_pred, filters, scores


def _parse_comparison(conj: exp.Expression, idx: int) -> ScoreSpec:
    left, right = conj.left, conj.right
    left_ai, right_ai = _ai_calls_in(left), _ai_calls_in(right)
    if bool(left_ai) == bool(right_ai):
        raise NotImplementedError(
            "a comparison must have an AI function on exactly one side"
        )
    ai_side, other = (left, right) if left_ai else (right, left)
    if not (isinstance(ai_side, exp.Anonymous) and len(_ai_calls_in(ai_side)) == 1):
        raise NotImplementedError(
            "the AI side of a comparison must be a single AI function call"
        )

    hidden = f"__aicmp_{idx}"
    call = _parse_call(ai_side, alias=hidden)
    output_type = (
        "string" if isinstance(other, exp.Literal) and other.is_string else "number"
    )
    op = build_compare_map_op(call, name=hidden, output_type=output_type)
    # Rewrite the comparison to reference the hidden column for the
    # residual relational filter, then render it.
    ai_side.replace(exp.column(hidden))
    return ScoreSpec(op=op, residual_sql=conj.sql(dialect="duckdb"), hidden=hidden)
