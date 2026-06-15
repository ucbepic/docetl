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

Relational ``ORDER BY`` / ``LIMIT`` are carried into the final
projection. ``NOT ai_filter`` negates the predicate. AI functions on
both sides of a comparison are supported. ``OR`` between an AI
predicate and another predicate raises (can't push the relational side
ahead). AI functions inside GROUP BY are not yet supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from docetl.aisql.functions import (
    ALL_FUNCTIONS,
    EXTRACT_FUNCTIONS,
    AIFunctionCall,
    _extract_output_col,
    _parse_schema_spec,
    build_compare_map_op,
    build_equijoin_op,
    build_extract_op,
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
    skip_checks: bool = False


_NOCHECK_RE = re.compile(r"/\*\s*nocheck\s*\*/", re.IGNORECASE)


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
    raw_literals = [a.this for a in args[1:] if isinstance(a, exp.Literal)]
    if raw_literals and "||" in raw_literals[0]:
        prompt, schema = raw_literals[0].split("||", 1)
        literals = (prompt, schema)
    else:
        literals = tuple(raw_literals)
    return AIFunctionCall(
        name=_ai_func_name(fn), column=args[0].name, literals=literals, alias=alias
    )


_SCHEMA_FNS = "|".join(
    sorted(
        {"ai", "ai_map", "ai_agg", "ai_summarize", "ai_classify"},
        key=len,
        reverse=True,
    )
)
_SCHEMA_ARG_RE = re.compile(
    rf"((?:{_SCHEMA_FNS})\s*\(\s*\w+\s*,\s*')([^']*)'(\s*,\s*')([^']*)'\s*\)",
    re.IGNORECASE,
)


def _merge_schema_args(sql: str) -> str:
    """Merge 3-arg AI calls ``ai_fn(col, 'prompt', 'schema')`` into
    2-arg ``ai_fn(col, 'prompt||schema')`` so sqlglot can parse them
    (some AI function names like ai_agg are recognized by sqlglot with
    a 2-arg max)."""
    return _SCHEMA_ARG_RE.sub(r"\1\2||\4')", sql)


def compile_sql(query: str) -> CompiledQuery:
    skip_checks = bool(_NOCHECK_RE.search(query))
    if skip_checks:
        query = _NOCHECK_RE.sub("", query).strip()

    query = _merge_schema_args(query)
    parsed = sqlglot.parse_one(query, read="duckdb")
    if not isinstance(parsed, exp.Select):
        raise NotImplementedError("only SELECT queries are supported")

    # Dispatch by query shape. Each AI surface compiles independently;
    # the SELECT/WHERE path handles everything else.
    resolve_fn = _resolve_source(parsed)
    if resolve_fn is not None:
        result = _compile_resolve(parsed, resolve_fn)
    elif (join := parsed.find(exp.Join)) is not None and _ai_calls_in(join):
        result = _compile_join(parsed, join)
    elif (group := parsed.find(exp.Group)) is not None and _select_ai_agg(parsed):
        result = _compile_grouped(parsed, group)
    else:
        result = _compile_select_where(parsed)

    result.skip_checks = skip_checks
    return result


def _compile_select_where(parsed: exp.Select) -> CompiledQuery:
    for label, cls in (
        ("GROUP BY", exp.Group),
        ("HAVING", exp.Having),
        ("JOIN", exp.Join),
    ):
        node = parsed.find(cls)
        if node is not None and _ai_calls_in(node):
            raise NotImplementedError(f"AI functions in {label} are not yet supported")

    where = parsed.args.get("where")
    relational_where, filter_calls, score_specs = _split_where(where)
    select_ai = _collect_select_ai(parsed)

    has_select_ai = any(call for _, call in select_ai)
    order = parsed.args.get("order")
    has_order_ai = order is not None and bool(_ai_calls_in(order))
    if not has_select_ai and not filter_calls and not score_specs and not has_order_ai:
        return CompiledQuery(stages=[RelationalStage(sql=parsed.sql(dialect="duckdb"))])

    stages: list = [_scan_stage(parsed, relational_where)]
    select_ops = _build_select_ops(select_ai)

    if score_specs:
        _emit_where_stages(stages, filter_calls, score_specs)
        if select_ops:
            stages.append(SemanticStage(operations=select_ops))
    else:
        _emit_where_stages(stages, filter_calls, score_specs, extra_ops=select_ops)

    final_items = []
    for item, call in select_ai:
        final_items.extend(_final_select_items(item, call))
    stages.extend(_final_projection(final_items, parsed))
    return CompiledQuery(stages=stages)


def _final_select_items(item: exp.Expression, call: AIFunctionCall | None):
    """Return a list of projection columns for one SELECT item."""
    if call is None:
        return [item]
    if call.name in EXTRACT_FUNCTIONS:
        return [exp.column(_extract_output_col(call)).as_(call.alias)]
    if len(call.literals) >= 2:
        schema = _parse_schema_spec(call.literals[1])
        return [exp.column(k) for k in schema]
    return [exp.column(call.alias)]


def _scan_stage(
    parsed: exp.Select, where_pred: exp.Expression | None
) -> RelationalStage:
    """A DuckDB scan of the query's FROM source, projecting everything,
    with an optional relational predicate."""
    scan = exp.select("*").from_(parsed.find(exp.From).this)
    if where_pred is not None:
        scan = scan.where(where_pred)
    return RelationalStage(sql=scan.sql(dialect="duckdb"))


def _collect_select_ai(
    parsed: exp.Select,
) -> list[tuple[exp.Expression, AIFunctionCall | None]]:
    """Pair each SELECT item with its AI call (if any), validating aliases."""
    result: list[tuple[exp.Expression, AIFunctionCall | None]] = []
    for item in parsed.expressions:
        calls = _ai_calls_in(item)
        if len(calls) > 1:
            raise NotImplementedError("one AI function per SELECT item in v1")
        if calls:
            alias = item.alias_or_name
            if not alias or _ai_func_name(item) is not None:
                raise NotImplementedError("AI function in SELECT needs an AS alias")
            result.append((item, _parse_call(calls[0], alias)))
        else:
            result.append((item, None))
    return result


def _build_select_ops(
    select_ai: list[tuple[exp.Expression, AIFunctionCall | None]],
) -> list[dict]:
    """Build DocETL ops for AI functions in the SELECT list."""
    return [
        (
            build_extract_op(call, name=f"ai_{call.alias}")
            if call.name in EXTRACT_FUNCTIONS
            else build_map_op(call, name=f"ai_{call.alias}")
        )
        for _, call in select_ai
        if call is not None
    ]


def _emit_where_stages(
    stages: list,
    filter_calls: list[AIFunctionCall],
    score_specs: list[ScoreSpec],
    extra_ops: list[dict] | None = None,
) -> None:
    """Append semantic + residual relational stages for WHERE AI predicates.

    Handles ai_filter ops, score-comparison ops (including both-sides-AI
    where some ScoreSpec.op may be None), and optional extra ops to merge
    into the same semantic stage."""
    filter_ops = [
        build_filter_op(call, name=f"aifilter_{i}")
        for i, call in enumerate(filter_calls)
    ]
    all_ops = filter_ops + (extra_ops or [])

    if score_specs:
        score_ops = [s.op for s in score_specs if s.op is not None]
        stages.append(SemanticStage(operations=all_ops + score_ops))
        residual_parts = [s.residual_sql for s in score_specs if s.residual_sql]
        hidden_parts = [s.hidden for s in score_specs if s.hidden]
        residual = " AND ".join(residual_parts)
        if hidden_parts:
            hidden = ", ".join(hidden_parts)
            stages.append(
                RelationalStage(
                    sql=f"SELECT * EXCLUDE ({hidden}) FROM {PREV} WHERE {residual}",
                    reads_prev=True,
                )
            )
        elif residual:
            stages.append(
                RelationalStage(
                    sql=f"SELECT * FROM {PREV} WHERE {residual}",
                    reads_prev=True,
                )
            )
    elif all_ops:
        stages.append(SemanticStage(operations=all_ops))


def _final_projection(
    items: list[exp.Expression],
    parsed: exp.Select,
) -> list[RelationalStage | SemanticStage]:
    """The query's SELECT list (AI calls already replaced by their alias
    columns) read from the prior stage's output, carrying the query's
    ORDER BY / LIMIT.

    Always returns a list of stages (length 1 or 3 when ORDER BY has
    AI functions)."""
    order = parsed.args.get("order")
    order_ai_ops: list[dict] = []
    order_hidden: list[str] = []

    if order is not None and _ai_calls_in(order):
        for i, ordered in enumerate(order.expressions):
            ai_calls = _ai_calls_in(ordered)
            if ai_calls:
                hidden = f"__aiord_{i}"
                call = _parse_call(ai_calls[0], alias=hidden)
                op = build_compare_map_op(call, name=hidden, output_type="number")
                order_ai_ops.append(op)
                order_hidden.append(hidden)
                ai_calls[0].replace(exp.column(hidden))

    extra_cols = [exp.column(h) for h in order_hidden]
    final = exp.select(*items, *extra_cols).from_(PREV)
    if order is not None:
        final = final.order_by(*[o.copy() for o in order.expressions])
    limit = parsed.args.get("limit")
    if limit is not None:
        final = final.limit(limit.expression.copy())

    stages: list = []
    if order_ai_ops:
        stages.append(SemanticStage(operations=order_ai_ops))
    stages.append(RelationalStage(sql=final.sql(dialect="duckdb"), reads_prev=True))
    if order_hidden:
        exclude = ", ".join(order_hidden)
        stages.append(
            RelationalStage(
                sql=f"SELECT * EXCLUDE ({exclude}) FROM {PREV}",
                reads_prev=True,
            )
        )
    return stages


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
    AI), a bare ``ai_filter(col, 'q')``, ``NOT ai_filter(col, 'q')``,
    or a comparison with AI function(s) on one or both sides.

    ``OR`` between an AI predicate and anything else is not supported
    (it would prevent pushing the relational side ahead).
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
        elif isinstance(conj, exp.Not) and _ai_func_name(conj.this) == "ai_filter":
            call = _parse_call(conj.this, alias="keep")
            call.negated = True
            filters.append(call)
        elif isinstance(conj, _COMPARISONS):
            scores.extend(_parse_comparison(conj, idx=len(scores)))
        elif isinstance(conj, exp.Or):
            _rewrite_or_ai(conj, scores)
        else:
            raise NotImplementedError(
                f"unsupported AI predicate form in WHERE: {conj.sql()}"
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
    """``SELECT k, ai_agg(col, 'q') AS x FROM src [WHERE ...] GROUP BY k``
    → DuckDB scan, [AI filters], DocETL reduce(s), [HAVING filter],
    DuckDB projection."""
    reduce_key = [e.name for e in group.expressions]
    if not all(isinstance(e, exp.Column) for e in group.expressions):
        raise NotImplementedError("GROUP BY must be over plain columns")

    # WHERE: split into relational (DuckDB) and AI (DocETL) parts
    where = parsed.args.get("where")
    relational_where, filter_calls, score_specs = _split_where(where)

    stages: list = [_scan_stage(parsed, relational_where)]

    if filter_calls or score_specs:
        _emit_where_stages(stages, filter_calls, score_specs)

    # Multiple ai_agg calls: each becomes its own reduce op chained in
    # sequence (each reads the prior's output). Schemas are merged so
    # later reduces pass through earlier columns.
    agg_calls = []
    for item in parsed.expressions:
        calls = _ai_calls_in(item)
        if any(_ai_func_name(c) == "ai_agg" for c in calls):
            agg_calls.append(_parse_call(calls[0], item.alias_or_name))
    if not agg_calls:
        raise NotImplementedError("GROUP BY query has no ai_agg calls")

    for i, call in enumerate(agg_calls):
        reduce_op = build_reduce_op(
            call,
            name=f"ai_agg_{i}" if len(agg_calls) > 1 else "ai_agg",
            reduce_key=reduce_key,
        )
        stages.append(SemanticStage(operations=[reduce_op]))

    # HAVING: runs as a DuckDB filter on the reduce output
    having = parsed.args.get("having")
    if having is not None:
        if _ai_calls_in(having):
            raise NotImplementedError("AI functions in HAVING are not yet supported")
        stages.append(
            RelationalStage(
                sql=f"SELECT * FROM {PREV} WHERE {having.this.sql(dialect='duckdb')}",
                reads_prev=True,
            )
        )

    final_items = []
    for i in parsed.expressions:
        final_items.extend(_final_select_items(i, _select_call(i)))
    stages.extend(_final_projection(final_items, parsed))
    return CompiledQuery(stages=stages)


def _compile_join(parsed: exp.Select, join: exp.Join) -> CompiledQuery:
    """``SELECT ... FROM l JOIN r ON ai_match(l.a, r.b, 'q') [WHERE ...]``
    → two DuckDB scans, a DocETL equijoin, [WHERE filter], [SELECT AI ops],
    DuckDB projection."""
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

    stages: list = [
        JoinStage(
            left_sql=exp.select("*").from_(left_src).sql(dialect="duckdb"),
            right_sql=exp.select("*").from_(join.this).sql(dialect="duckdb"),
            operation=op,
        ),
    ]

    where = parsed.args.get("where")
    if where is not None:
        relational_where, filter_calls, score_specs = _split_where(where)
        if relational_where is not None:
            stages.append(
                RelationalStage(
                    sql=f"SELECT * FROM {PREV} WHERE {relational_where.sql(dialect='duckdb')}",
                    reads_prev=True,
                )
            )
        if filter_calls or score_specs:
            _emit_where_stages(stages, filter_calls, score_specs)

    select_ai = _collect_select_ai(parsed)
    select_ops = _build_select_ops(select_ai)
    if select_ops:
        stages.append(SemanticStage(operations=select_ops))

    final_items = []
    for item, call in select_ai:
        final_items.extend(_final_select_items(item, call))
    stages.extend(_final_projection(final_items, parsed))
    return CompiledQuery(stages=stages)


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
    stages = [
        RelationalStage(sql=f"SELECT * FROM {source}"),
        SemanticStage(operations=[op]),
    ]
    stages.extend(_final_projection(list(parsed.expressions), parsed))
    return CompiledQuery(stages=stages)


def _select_call(item: exp.Expression) -> AIFunctionCall | None:
    calls = _ai_calls_in(item)
    return _parse_call(calls[0], item.alias_or_name) if calls else None


def _rewrite_or_ai(node: exp.Expression, scores: list[ScoreSpec]) -> None:
    """Rewrite AI calls inside an OR expression into hidden boolean columns.

    Each ai_filter becomes a map op producing a boolean column. The AI
    call node is replaced in-place with a column reference, so the
    whole OR expression becomes a purely relational residual predicate.
    NOT is preserved in the SQL (``NOT __aior_0``) — no prompt negation."""
    for fn in list(node.find_all(exp.Func)):
        name = _ai_func_name(fn)
        if name is None:
            continue
        if name != "ai_filter":
            raise NotImplementedError(
                f"only ai_filter is supported inside OR (found {name})"
            )
        idx = len(scores)
        hidden = f"__aior_{idx}"
        call = _parse_call(fn, alias=hidden)
        op = build_compare_map_op(call, name=hidden, output_type="boolean")
        fn.replace(exp.column(hidden))
        scores.append(ScoreSpec(op=op, residual_sql="", hidden=hidden))
    residual = node.sql(dialect="duckdb")
    scores.append(ScoreSpec(op=None, residual_sql=residual, hidden=""))


def _parse_comparison(conj: exp.Expression, idx: int) -> list[ScoreSpec]:
    left, right = conj.left, conj.right
    left_ai, right_ai = _ai_calls_in(left), _ai_calls_in(right)

    if left_ai and right_ai:
        hidden_l = f"__aicmp_{idx}_l"
        hidden_r = f"__aicmp_{idx}_r"
        call_l = _parse_call(left_ai[0], alias=hidden_l)
        call_r = _parse_call(right_ai[0], alias=hidden_r)
        op_l = build_compare_map_op(call_l, name=hidden_l, output_type="number")
        op_r = build_compare_map_op(call_r, name=hidden_r, output_type="number")
        left.replace(exp.column(hidden_l))
        right.replace(exp.column(hidden_r))
        return [
            ScoreSpec(op=op_l, residual_sql="", hidden=hidden_l),
            ScoreSpec(op=op_r, residual_sql="", hidden=hidden_r),
            ScoreSpec(op=None, residual_sql=conj.sql(dialect="duckdb"), hidden=""),
        ]

    if not left_ai and not right_ai:
        raise NotImplementedError(
            "a comparison in the AI WHERE split must have an AI function"
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
    ai_side.replace(exp.column(hidden))
    return [ScoreSpec(op=op, residual_sql=conj.sql(dialect="duckdb"), hidden=hidden)]
