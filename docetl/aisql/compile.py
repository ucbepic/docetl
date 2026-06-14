"""Compile an AI-SQL query into an ordered list of execution stages.

v1 handles::

    SELECT <items, some ai_*(...) AS alias>
    FROM <source>
    [WHERE <relational conjuncts> AND <ai_filter(...) conjuncts>]

which lowers to up to three stages (any absent when not needed):

  1. relational  — ``SELECT * FROM <source> WHERE <relational conjuncts>``
  2. semantic    — DocETL ``filter`` per WHERE ``ai_filter`` (run first, to
     shrink), then a ``map`` per SELECT-list AI function
  3. relational  — the original SELECT list (AI calls replaced by their
     alias) over the semantic output

The WHERE split is the design's selection pushdown across the boundary:
relational conjuncts run first in DuckDB (file pushdown), the LLM filters
run only on survivors. Deferred (clear errors): ``OR``/``NOT`` around an
AI predicate, and AI functions inside a comparison (``ai_score(...) > k``);
plus AI functions in GROUP BY / JOIN / ORDER (later milestones).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from docetl.aisql.functions import (
    ALL_FUNCTIONS,
    AIFunctionCall,
    build_filter_op,
    build_map_op,
)

PREV = "_prev"  # name under which a stage's output is registered for the next


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

    # Split WHERE into relational conjuncts (DuckDB) and ai_filter
    # conjuncts (DocETL filters).
    where = parsed.args.get("where")
    relational_where, filter_calls = _split_where(where)

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
    if not has_select_ai and not filter_calls:
        # Pure relational — hand the whole query to DuckDB unchanged.
        return CompiledQuery(stages=[RelationalStage(sql=parsed.sql(dialect="duckdb"))])

    # Stage 1: scan + the relational half of WHERE, project everything.
    scan = exp.select("*").from_(parsed.find(exp.From).this)
    if relational_where is not None:
        scan = scan.where(relational_where)
    stages: list = [RelationalStage(sql=scan.sql(dialect="duckdb"))]

    # Stage 2: WHERE ai_filters first (they shrink the set), then a map
    # per SELECT-list AI function.
    ops = [
        build_filter_op(call, name=f"aifilter_{i}")
        for i, call in enumerate(filter_calls)
    ]
    ops += [
        build_map_op(call, name=f"ai_{call.alias}")
        for _, call in select_ai
        if call is not None
    ]
    stages.append(SemanticStage(operations=ops))

    # Stage 3: the original SELECT list with AI calls replaced by their
    # alias column, read from the semantic output.
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
) -> tuple[exp.Expression | None, list[AIFunctionCall]]:
    """Partition a WHERE into (relational predicate, ai_filter calls).

    Only top-level ``AND`` of [relational | ``ai_filter(col, 'q')``] is
    supported. An AI function anywhere else in the predicate (inside an
    ``OR``/``NOT``, or as an argument to a comparison) raises — those are
    the next splitter sub-milestones.
    """
    if where is None:
        return None, []

    relational: list[exp.Expression] = []
    filters: list[AIFunctionCall] = []
    for conj in _conjuncts(where.this):
        calls = _ai_calls_in(conj)
        if not calls:
            relational.append(conj)
            continue
        if isinstance(conj, exp.Anonymous) and conj.name.lower() == "ai_filter":
            filters.append(_parse_call(conj, alias="keep"))
        else:
            raise NotImplementedError(
                "in WHERE, AI functions are only supported as a top-level "
                "ai_filter(col, 'question') conjunct; OR/NOT and comparisons "
                "like ai_score(col, 'q') > 0.8 are not yet supported"
            )

    relational_pred = None
    if relational:
        relational_pred = relational[0]
        for extra in relational[1:]:
            relational_pred = exp.and_(relational_pred, extra)
    return relational_pred, filters
