"""AI SQL: a SQL frontend that compiles to DocETL operators with plain
relational work delegated to DuckDB. See ``docs/design/ai-sql.md``.

Optional feature — install with ``pip install docetl[aisql]`` (duckdb,
sqlglot, pyarrow).
"""

from docetl.aisql.compile import (
    CompiledQuery,
    RelationalStage,
    SemanticStage,
    compile_sql,
)
from docetl.aisql.engine import DuckDBEngine
from docetl.aisql.run import run_compiled, run_sql

__all__ = [
    "CompiledQuery",
    "DuckDBEngine",
    "RelationalStage",
    "SemanticStage",
    "compile_sql",
    "run_compiled",
    "run_sql",
]
