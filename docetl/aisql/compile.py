"""Compile an AI-SQL query into an ordered list of execution stages.

v1 handles the straight-line shape::

    SELECT <items, some ai_*(...) AS alias>  FROM <source>  [WHERE <relational>]

which lowers to three stages (the middle one absent when there are no AI
functions):

  1. relational  — ``SELECT * FROM <source> WHERE <relational>`` in DuckDB
  2. semantic    — one DocETL ``map`` per AI function
  3. relational  — the original SELECT list (AI calls replaced by their
     alias) over the semantic output

AI functions in WHERE/GROUP BY/JOIN are detected and rejected with a clear
error — those are the splitter / aggregate / join milestones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from docetl.aisql.functions import ALL_FUNCTIONS, AIFunctionCall, build_map_op

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

    where = parsed.args.get("where")
    if where is not None and _ai_calls_in(where):
        raise NotImplementedError(
            "AI functions in WHERE need the predicate splitter (milestone 4)"
        )
    for label, cls in (
        ("GROUP BY", exp.Group),
        ("HAVING", exp.Having),
        ("JOIN", exp.Join),
        ("ORDER BY", exp.Order),
    ):
        node = parsed.find(cls)
        if node is not None and _ai_calls_in(node):
            raise NotImplementedError(f"AI functions in {label} are not yet supported")

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

    if not any(call for _, call in select_ai):
        # Pure relational — hand the whole query to DuckDB unchanged.
        return CompiledQuery(stages=[RelationalStage(sql=parsed.sql(dialect="duckdb"))])

    # Stage 1: scan + relational filter, project everything.
    scan = exp.select("*").from_(parsed.find(exp.From).this)
    if where is not None:
        scan = scan.where(where.this)
    stages: list = [RelationalStage(sql=scan.sql(dialect="duckdb"))]

    # Stage 2: a map per AI function.
    ops = [
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
