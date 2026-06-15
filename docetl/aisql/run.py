"""Execute a CompiledQuery by threading Arrow tables between DuckDB
(relational stages) and DocETL (semantic stages).

When ``agent_mode=True``, pre-execution checks validate the query
before any LLM calls run.  Without agent mode, long text is handled
by DocETL's normal truncation (with a warning pointing to split/gather).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docetl.aisql.compile import (
    PREV,
    CompiledQuery,
    JoinStage,
    RelationalStage,
    SemanticStage,
    compile_sql,
)
from docetl.aisql.engine import DuckDBEngine
from docetl.checks import (
    _DEFAULT_MAX_AI_ROWS,
    _DEFAULT_MAX_CHUNK_TOKENS,
    check_cardinality,
    check_chunk_overflow,
    check_empty,
    check_missing_cols,
    check_row_count,
)

if TYPE_CHECKING:
    import pyarrow as pa


# ── helpers ──────────────────────────────────────────────────────────────


def _run_stage_checks(
    table: "pa.Table",
    ops: list[dict],
    *,
    max_chunk_tokens: int,
    max_ai_rows: int,
    last_rel_sql: str | None,
) -> None:
    """Run all agent-mode checks for a semantic stage's Arrow table."""
    check_empty(table, sql_hint=last_rel_sql)
    check_row_count(table, max_ai_rows=max_ai_rows)

    for op in ops:
        check_missing_cols(table, op)
        if op.get("type") in ("reduce", "resolve"):
            check_cardinality(table, op)
        check_chunk_overflow(table, op, max_chunk_tokens=max_chunk_tokens)


# ── execution ─────────────────────────────────────────────────────────────


def run_compiled(
    compiled: CompiledQuery,
    max_threads: int | None = None,
    max_chunk_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS,
    agent_mode: bool = False,
    max_ai_rows: int = _DEFAULT_MAX_AI_ROWS,
) -> tuple["pa.Table", float]:
    """Execute a compiled query, returning ``(arrow_table, total_llm_cost)``.

    *agent_mode* enables pre-execution validation that raises
    informative errors instead of proceeding with suboptimal queries.
    The checks are defined in :mod:`docetl.checks` and run in order;
    a ``/* nocheck */`` comment in the SQL (detected at compile time
    via :attr:`CompiledQuery.skip_checks`) disables them.
    """
    import docetl

    run_checks = agent_mode and not compiled.skip_checks
    table: "pa.Table | None" = None
    total_cost = 0.0
    last_rel_sql: str | None = None
    with DuckDBEngine() as engine:
        for stage in compiled.stages:
            if isinstance(stage, RelationalStage):
                if stage.reads_prev:
                    assert table is not None, "relational stage has no input to read"
                    engine.register(PREV, table)
                table = engine.sql(stage.sql)
                last_rel_sql = stage.sql
            elif isinstance(stage, SemanticStage):
                assert table is not None, "semantic stage has no input rows"
                in_schema = table.schema

                if run_checks:
                    _run_stage_checks(
                        table,
                        stage.operations,
                        max_chunk_tokens=max_chunk_tokens,
                        max_ai_rows=max_ai_rows,
                        last_rel_sql=last_rel_sql,
                    )

                frame = docetl.from_arrow(table)
                for op in stage.operations:
                    frame = frame._append_op(op["type"], op["name"], _op_kwargs(op))
                table = frame.to_arrow(max_threads=max_threads)
                total_cost += frame.total_cost

                if table.num_columns == 0:
                    table = _empty_like(in_schema, stage.operations)
            elif isinstance(stage, JoinStage):
                left = docetl.from_arrow(engine.sql(stage.left_sql), name="left")
                right = docetl.from_arrow(engine.sql(stage.right_sql), name="right")
                joined = left.equijoin(right, **_op_kwargs(stage.operation))
                table = joined.to_arrow(max_threads=max_threads)
                total_cost += joined.total_cost
            else:  # pragma: no cover - exhaustive
                raise TypeError(f"unknown stage type: {type(stage).__name__}")
    assert table is not None, "query compiled to no stages"
    return table, total_cost


def run_sql(
    query: str,
    max_threads: int | None = None,
    max_chunk_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS,
    agent_mode: bool = False,
    max_ai_rows: int = _DEFAULT_MAX_AI_ROWS,
) -> tuple["pa.Table", float]:
    """Compile and execute an AI-SQL query.

    Returns ``(arrow_table, total_llm_cost)`` where cost is the sum of all
    LLM calls made by semantic stages (extractions, filters, etc.).

    When *agent_mode* is True, raises structured errors instead of
    proceeding with suboptimal queries.
    A ``/* nocheck */`` comment anywhere in the SQL bypasses these checks
    (detected at compile time via :attr:`CompiledQuery.skip_checks`).
    """
    return run_compiled(
        compile_sql(query),
        max_threads=max_threads,
        max_chunk_tokens=max_chunk_tokens,
        agent_mode=agent_mode,
        max_ai_rows=max_ai_rows,
    )


def _op_kwargs(op: dict) -> dict:
    return {k: v for k, v in op.items() if k not in ("name", "type")}


def _empty_like(in_schema, operations):
    """A zero-row Arrow table whose columns cover what a downstream stage
    might reference: the input columns (keeping their types) plus each
    op's output-schema keys (filters add none — their decision key is
    consumed)."""
    import pyarrow as pa

    fields = list(in_schema)
    have = set(in_schema.names)
    for op in operations:
        if op.get("type") == "filter":
            continue
        for col in (op.get("output") or {}).get("schema") or {}:
            if col not in have:
                fields.append(pa.field(col, pa.string()))
                have.add(col)
    return pa.schema(fields).empty_table()
