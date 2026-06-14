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
    build_equijoin_op,
    build_filter_op,
    build_map_op,
    build_reduce_op,
    build_resolve_op,
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
class JoinStage:
    """Two relational inputs joined by a DocETL ``equijoin``. The output
    is registered as ``_prev`` for the following stage."""

    left_sql: str
    right_sql: str
    operation: dict


@dataclass
class CompiledQuery:
    stages: list  # RelationalStage | SemanticStage, in execution order


def _ai_func_name(node: exp.Expression) -> str | None:
    """The AI function name if *node* is one of ours, else None. Handles
    both ``Anonymous`` (e.g. ai_filter) and sqlglot's specialized
    function nodes (e.g. AI_AGG → exp.AIAgg, AI_CLASSIFY → exp.AIClassify)."""
    if not isinstance(node, exp.Func):
        return None
    name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
    return name if name in ALL_FUNCTIONS else None


def _func_args(fn: exp.Func) -> list[exp.Expression]:
    """A function's arguments in order, whether sqlglot put them in
    ``expressions`` (Anonymous) or in named slots like ``this`` /
    ``expression`` / ``categories`` (specialized AI nodes)."""
    if fn.expressions:
        return list(fn.expressions)
    args: list[exp.Expression] = []
    this = fn.args.get("this")
    if isinstance(this, exp.Expression):
        args.append(this)
    for key, value in fn.args.items():
        if key != "this" and isinstance(value, exp.Expression):
            args.append(value)
    return args


def _ai_calls_in(node: exp.Expression) -> list[exp.Func]:
    """AI function nodes anywhere under *node*."""
    return [f for f in node.find_all(exp.Func) if _ai_func_name(f) is not None]


def _parse_call(fn: exp.Func, alias: str) -> AIFunctionCall:
    args = _func_args(fn)
    if not args or not isinstance(args[0], exp.Column):
        raise NotImplementedError(
            f"{_ai_func_name(fn)}(...) must take a column as its first argument in v1"
        )
    literals = tuple(a.this for a in args[1:] if isinstance(a, exp.Literal))
    return AIFunctionCall(
        name=_ai_func_name(fn), column=args[0].name, literals=literals, alias=alias
    )


def compile_sql(query: str) -> CompiledQuery:
    parsed = sqlglot.parse_one(query, read="duckdb")
    if not isinstance(parsed, exp.Select):
        raise NotImplementedError("only SELECT queries are supported")

    # Dispatch by query shape. Each AI surface compiles independently;
    # the SELECT/WHERE path handles everything else.
    resolve_fn = _resolve_source(parsed)
    if resolve_fn is not None:
        return _compile_resolve(parsed, resolve_fn)
    join = parsed.find(exp.Join)
    if join is not None and _ai_calls_in(join):
        return _compile_join(parsed, join)
    group = parsed.find(exp.Group)
    if group is not None and _select_ai_agg(parsed):
        return _compile_grouped(parsed, group)
    return _compile_select_where(parsed)


def _compile_select_where(parsed: exp.Select) -> CompiledQuery:
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
            if not alias or _ai_func_name(item) is not None:
                raise NotImplementedError("AI function in SELECT needs an AS alias")
            select_ai.append((item, _parse_call(calls[0], alias)))
        else:
            select_ai.append((item, None))

    has_select_ai = any(call for _, call in select_ai)
    if not has_select_ai and not filter_calls and not score_specs:
        # Pure relational — hand the whole query to DuckDB unchanged.
        return CompiledQuery(stages=[RelationalStage(sql=parsed.sql(dialect="duckdb"))])

    # Stage 1: scan + the relational half of WHERE, project everything.
    stages: list = [_scan_stage(parsed, relational_where)]

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
    stages.append(
        _final_projection([_final_select_item(item, call) for item, call in select_ai])
    )
    return CompiledQuery(stages=stages)


def _final_select_item(item: exp.Expression, call: AIFunctionCall | None):
    if call is None:
        return item
    return exp.column(call.alias)


def _scan_stage(
    parsed: exp.Select, where_pred: exp.Expression | None
) -> RelationalStage:
    """A DuckDB scan of the query's FROM source, projecting everything,
    with an optional relational predicate."""
    scan = exp.select("*").from_(parsed.find(exp.From).this)
    if where_pred is not None:
        scan = scan.where(where_pred)
    return RelationalStage(sql=scan.sql(dialect="duckdb"))


def _final_projection(items: list[exp.Expression]) -> RelationalStage:
    """The query's SELECT list (AI calls already replaced by their alias
    columns) read from the prior stage's output."""
    return RelationalStage(
        sql=exp.select(*items).from_(PREV).sql(dialect="duckdb"), reads_prev=True
    )


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
        elif _ai_func_name(conj) == "ai_filter":
            filters.append(_parse_call(conj, alias="keep"))
        elif isinstance(conj, _COMPARISONS):
            scores.append(_parse_comparison(conj, idx=len(scores)))
        else:
            raise NotImplementedError(
                "in WHERE, AI functions are supported as a top-level "
                "ai_filter(col, 'q') conjunct or a comparison like "
                "ai_score(col, 'q') > 0.8; OR/NOT and other nestings are not"
            )

    relational_pred = exp.and_(*relational) if relational else None
    return relational_pred, filters, scores


def _select_ai_agg(parsed: exp.Select) -> bool:
    return any(
        _ai_func_name(a) == "ai_agg"
        for item in parsed.expressions
        for a in _ai_calls_in(item)
    )


def _compile_grouped(parsed: exp.Select, group: exp.Group) -> CompiledQuery:
    """``SELECT k, ai_agg(col, 'q') AS x FROM src [WHERE relational] GROUP BY k``
    → DuckDB scan, DocETL reduce, DuckDB projection."""
    if parsed.args.get("having") is not None:
        raise NotImplementedError("HAVING with ai_agg is not yet supported")
    where = parsed.args.get("where")
    if where is not None and _ai_calls_in(where):
        raise NotImplementedError(
            "AI functions in WHERE with GROUP BY are not yet supported"
        )

    reduce_key = [e.name for e in group.expressions]
    if not all(isinstance(e, exp.Column) for e in group.expressions):
        raise NotImplementedError("GROUP BY must be over plain columns in v1")

    agg_calls = [
        _parse_call(_ai_calls_in(item)[0], item.alias_or_name)
        for item in parsed.expressions
        if any(_ai_func_name(c) == "ai_agg" for c in _ai_calls_in(item))
    ]
    if len(agg_calls) != 1:
        raise NotImplementedError("exactly one ai_agg per GROUP BY query in v1")

    reduce_op = build_reduce_op(agg_calls[0], name="ai_agg", reduce_key=reduce_key)
    return CompiledQuery(
        stages=[
            _scan_stage(parsed, where.this if where is not None else None),
            SemanticStage(operations=[reduce_op]),
            _final_projection(
                [_final_select_item(i, _select_call(i)) for i in parsed.expressions]
            ),
        ]
    )


def _compile_join(parsed: exp.Select, join: exp.Join) -> CompiledQuery:
    """``SELECT ... FROM l JOIN r ON ai_match(l.a, r.b, 'q')`` → two DuckDB
    scans, a DocETL equijoin, a DuckDB projection."""
    if parsed.args.get("where") is not None:
        raise NotImplementedError("WHERE with an AI join is not yet supported")
    if _ai_calls_in(exp.select(*parsed.expressions)):
        raise NotImplementedError(
            "AI functions in SELECT with an AI join are not yet supported"
        )
    on = join.args.get("on")
    on_args = _func_args(on) if _ai_func_name(on) == "ai_match" else []
    if not (
        len(on_args) == 3
        and isinstance(on_args[0], exp.Column)
        and isinstance(on_args[1], exp.Column)
        and isinstance(on_args[2], exp.Literal)
    ):
        raise NotImplementedError(
            "join ON must be ai_match(left_col, right_col, 'question')"
        )
    left_col, right_col, prompt = on_args
    op = build_equijoin_op("aijoin", left_col.name, right_col.name, prompt.this)
    left_src = parsed.find(exp.From).this
    return CompiledQuery(
        stages=[
            JoinStage(
                left_sql=exp.select("*").from_(left_src).sql(dialect="duckdb"),
                right_sql=exp.select("*").from_(join.this).sql(dialect="duckdb"),
                operation=op,
            ),
            _final_projection(list(parsed.expressions)),
        ]
    )


def _resolve_source(parsed: exp.Select) -> exp.Func | None:
    """The ``ai_resolve(...)`` table function in FROM, if present."""
    from_node = parsed.find(exp.From)
    if from_node is None:
        return None
    for f in from_node.find_all(exp.Func):
        if _ai_func_name(f) == "ai_resolve":
            return f
    return None


def _compile_resolve(parsed: exp.Select, fn: exp.Func) -> CompiledQuery:
    """``SELECT ... FROM ai_resolve(table, on := col, prompt := 'q')`` →
    DuckDB scan, DocETL resolve, DuckDB projection."""
    args = _func_args(fn)
    if not args:
        raise NotImplementedError("ai_resolve(table, ...) needs a table as first arg")
    src = args[0]
    if isinstance(src, exp.Column):  # bare table name
        source = src.name
    elif isinstance(src, exp.Literal) and src.is_string:  # quoted file path
        source = f"'{src.this}'"
    else:
        raise NotImplementedError(
            "ai_resolve's first arg must be a table name or a quoted file path"
        )
    named: dict[str, exp.Expression] = {}
    positional: list[exp.Expression] = []
    for a in args[1:]:
        if isinstance(a, exp.PropertyEQ):
            named[a.this.name.lower()] = a.expression
        else:
            positional.append(a)
    on_node = named.get("on") or (positional[0] if positional else None)
    prompt_node = named.get("prompt") or (
        positional[1] if len(positional) > 1 else None
    )
    if on_node is None or prompt_node is None:
        raise NotImplementedError("ai_resolve needs on:=column and prompt:='question'")
    on_col = on_node.name if isinstance(on_node, exp.Column) else on_node.this
    prompt = (
        prompt_node.this if isinstance(prompt_node, exp.Literal) else prompt_node.name
    )
    op = build_resolve_op("airesolve", column=on_col, prompt=prompt, output_key=on_col)
    return CompiledQuery(
        stages=[
            RelationalStage(sql=f"SELECT * FROM {source}"),
            SemanticStage(operations=[op]),
            _final_projection(list(parsed.expressions)),
        ]
    )


def _select_call(item: exp.Expression) -> AIFunctionCall | None:
    calls = _ai_calls_in(item)
    return _parse_call(calls[0], item.alias_or_name) if calls else None


def _parse_comparison(conj: exp.Expression, idx: int) -> ScoreSpec:
    left, right = conj.left, conj.right
    left_ai, right_ai = _ai_calls_in(left), _ai_calls_in(right)
    if bool(left_ai) == bool(right_ai):
        raise NotImplementedError(
            "a comparison must have an AI function on exactly one side"
        )
    ai_side, other = (left, right) if left_ai else (right, left)
    if not (_ai_func_name(ai_side) is not None and len(_ai_calls_in(ai_side)) == 1):
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
