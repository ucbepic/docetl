"""DuckDB delegate: run the relational fragments of an AI-SQL query.

DocETL conducts; this is the subroutine it hands relational work to. The
engine reads source files directly (with DuckDB's pushdown), executes a
relational SQL string, and returns Arrow; DocETL operator output is
registered back as a named table so relational steps can run on top of
it. The handoff is zero-copy Arrow in both directions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa


def _require_duckdb():
    try:
        import duckdb
    except ModuleNotFoundError as e:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "The AI SQL layer needs DuckDB. Install with: pip install docetl[aisql]"
        ) from e
    return duckdb


class DuckDBEngine:
    """A thin wrapper over a DuckDB connection for the AI-SQL boundary.

    One engine per query run: register intermediate Arrow tables under
    names, then run relational SQL that references them (or reads source
    files directly). Use as a context manager so the connection closes.
    """

    def __init__(self) -> None:
        duckdb = _require_duckdb()
        self._con = duckdb.connect()
        # Keep registered Arrow tables alive for the connection's life;
        # DuckDB holds a view over them, not a copy.
        self._registered: dict[str, "pa.Table"] = {}

    def register(self, name: str, table: "pa.Table") -> None:
        """Expose an Arrow table to SQL under *name* (zero-copy)."""
        self._registered[name] = table
        self._con.register(name, table)

    def sql(self, query: str) -> "pa.Table":
        """Run a relational SQL string and return the result as Arrow."""
        return self._con.execute(query).fetch_arrow_table()

    def close(self) -> None:
        self._con.close()
        self._registered.clear()

    def __enter__(self) -> "DuckDBEngine":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
