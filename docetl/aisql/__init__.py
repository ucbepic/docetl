"""AI SQL: a SQL frontend that compiles to DocETL operators with plain
relational work delegated to DuckDB. See ``docs/design/ai-sql.md``.

Optional feature — install with ``pip install docetl[aisql]`` (duckdb,
sqlglot, pyarrow).
"""

from docetl.aisql.engine import DuckDBEngine

__all__ = ["DuckDBEngine"]
