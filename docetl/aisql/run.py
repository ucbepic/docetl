"""Execute a CompiledQuery by threading Arrow tables between DuckDB
(relational stages) and DocETL (semantic stages)."""

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

if TYPE_CHECKING:
    import pyarrow as pa


def run_compiled(compiled: CompiledQuery, max_threads: int | None = None) -> "pa.Table":
    import docetl

    table: "pa.Table | None" = None
    with DuckDBEngine() as engine:
        for stage in compiled.stages:
            if isinstance(stage, RelationalStage):
                if stage.reads_prev:
                    assert table is not None, "relational stage has no input to read"
                    engine.register(PREV, table)
                table = engine.sql(stage.sql)
            elif isinstance(stage, SemanticStage):
                assert table is not None, "semantic stage has no input rows"
                in_schema = table.schema
                frame = docetl.from_arrow(table)
                for op in stage.operations:
                    frame = frame._append_op(op["type"], op["name"], _op_kwargs(op))
                table = frame.to_arrow(max_threads=max_threads)
                if table.num_columns == 0:
                    # All rows dropped (e.g. a filter): from_pylist([]) loses
                    # the schema, which breaks the next relational stage.
                    # Rebuild an empty table carrying the columns the next
                    # stage may reference (input columns + op outputs).
                    table = _empty_like(in_schema, stage.operations)
            elif isinstance(stage, JoinStage):
                left = docetl.from_arrow(engine.sql(stage.left_sql), name="left")
                right = docetl.from_arrow(engine.sql(stage.right_sql), name="right")
                joined = left.equijoin(right, **_op_kwargs(stage.operation))
                table = joined.to_arrow(max_threads=max_threads)
            else:  # pragma: no cover - exhaustive
                raise TypeError(f"unknown stage type: {type(stage).__name__}")
    assert table is not None, "query compiled to no stages"
    return table


def run_sql(query: str, max_threads: int | None = None) -> "pa.Table":
    """Compile and execute an AI-SQL query, returning Arrow."""
    return run_compiled(compile_sql(query), max_threads=max_threads)


def _op_kwargs(op: dict) -> dict:
    return {k: v for k, v in op.items() if k not in ("name", "type")}


def _empty_like(in_schema, operations):
    """A zero-row Arrow table whose columns cover what a downstream stage
    might reference: the input columns (keeping their types) plus each
    op's output-schema keys (filters add none — their decision key is
    consumed)."""
    import pyarrow as pa

    fields = list(in_schema)  # input columns keep their original types
    have = set(in_schema.names)
    for op in operations:
        if op.get("type") == "filter":
            continue
        for col in (op.get("output") or {}).get("schema") or {}:
            if col not in have:
                fields.append(pa.field(col, pa.string()))
                have.add(col)
    return pa.schema(fields).empty_table()
