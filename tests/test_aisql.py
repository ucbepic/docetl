"""AI SQL layer: Arrow adapters, DuckDB delegate, SQL compilation.

LLM-free — the AI operators are exercised structurally (we assert the
compiled plan), and execution tests use relational/code paths only.
"""

import pyarrow as pa

import docetl


class TestArrowAdapters:
    ROWS = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, {"id": 3, "text": "c"}]

    # The runner needs at least one op, so every case carries a code_map
    # (the realistic shape: load Arrow, then transform).
    NOOP = "def transform(doc):\n    return {}"

    def test_from_arrow_feeds_pipeline(self):
        table = pa.Table.from_pylist(self.ROWS)
        out = docetl.from_arrow(table).code_map(
            code="def transform(doc):\n    return {'n': doc['id'] * 2}"
        ).collect()
        assert [r["n"] for r in out] == [2, 4, 6]
        assert out[0] == {"id": 1, "text": "a", "n": 2}

    def test_to_arrow_returns_table(self):
        out = docetl.from_list(self.ROWS).code_map(code=self.NOOP).to_arrow()
        assert isinstance(out, pa.Table)
        assert out.to_pylist() == self.ROWS

    def test_round_trip(self):
        table = pa.Table.from_pylist(self.ROWS)
        out = docetl.from_arrow(table).code_map(code=self.NOOP).to_arrow()
        assert out.to_pylist() == self.ROWS


class TestDuckDBEngine:
    def test_register_and_filter(self):
        from docetl.aisql import DuckDBEngine

        rows = [{"id": i, "price": i * 5} for i in range(1, 6)]
        with DuckDBEngine() as eng:
            eng.register("items", pa.Table.from_pylist(rows))
            out = eng.sql("SELECT id FROM items WHERE price > 10 ORDER BY id")
        assert out.to_pylist() == [{"id": 3}, {"id": 4}, {"id": 5}]

    def test_reads_parquet_file(self, tmp_path):
        import pyarrow.parquet as pq

        from docetl.aisql import DuckDBEngine

        path = tmp_path / "calls.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [{"id": 1, "dur": 100}, {"id": 2, "dur": 400}, {"id": 3, "dur": 500}]
            ),
            path,
        )
        with DuckDBEngine() as eng:
            out = eng.sql(f"SELECT id FROM '{path}' WHERE dur > 300 ORDER BY id")
        assert out.to_pylist() == [{"id": 2}, {"id": 3}]


class TestCompile:
    def test_leading_example(self):
        from docetl.aisql import compile_sql, RelationalStage, SemanticStage

        stages = compile_sql(
            "SELECT ai_summarize(transcript) AS summary "
            "FROM calls WHERE duration > 300"
        ).stages
        assert len(stages) == 3
        assert isinstance(stages[0], RelationalStage)
        assert stages[0].sql == "SELECT * FROM calls WHERE duration > 300"
        assert not stages[0].reads_prev
        assert isinstance(stages[1], SemanticStage)
        (op,) = stages[1].operations
        assert op["type"] == "map"
        assert op["output"]["schema"] == {"summary": "string"}
        assert "{{ input.transcript }}" in op["prompt"]
        assert isinstance(stages[2], RelationalStage)
        assert stages[2].sql == "SELECT summary FROM _prev"
        assert stages[2].reads_prev

    def test_pure_relational_single_stage(self):
        from docetl.aisql import compile_sql, RelationalStage

        stages = compile_sql("SELECT id FROM t WHERE x > 1").stages
        assert len(stages) == 1
        assert isinstance(stages[0], RelationalStage)
        assert stages[0].sql == "SELECT id FROM t WHERE x > 1"

    def test_passthrough_column_kept(self):
        from docetl.aisql import compile_sql

        stages = compile_sql("SELECT id, ai_summarize(body) AS s FROM docs").stages
        assert stages[0].sql == "SELECT * FROM docs"
        assert stages[2].sql == "SELECT id, s FROM _prev"

    def test_ai_in_where_rejected(self):
        import pytest

        from docetl.aisql import compile_sql

        with pytest.raises(NotImplementedError, match="WHERE"):
            compile_sql("SELECT id FROM t WHERE ai_filter(text, 'q')")

    def test_ai_in_select_needs_alias(self):
        import pytest

        from docetl.aisql import compile_sql

        with pytest.raises(NotImplementedError, match="alias"):
            compile_sql("SELECT ai_summarize(body) FROM docs")


class TestRun:
    def _parquet(self, tmp_path):
        import pyarrow.parquet as pq

        path = tmp_path / "products.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [{"id": i, "price": i * 5} for i in range(1, 6)]
            ),
            path,
        )
        return path

    def test_run_pure_relational(self, tmp_path):
        from docetl.aisql import run_sql

        path = self._parquet(tmp_path)
        out = run_sql(f"SELECT id FROM '{path}' WHERE price > 10 ORDER BY id")
        assert out.to_pylist() == [{"id": 3}, {"id": 4}, {"id": 5}]

    def test_hybrid_threads_duckdb_and_docetl(self, tmp_path):
        # Manually-built plan with a code_map semantic stage: exercises the
        # real DuckDB -> DocETL -> DuckDB handoff (register/_prev, Arrow
        # threading) end to end, no LLM.
        from docetl.aisql import (
            CompiledQuery,
            RelationalStage,
            SemanticStage,
            run_compiled,
        )

        path = self._parquet(tmp_path)
        compiled = CompiledQuery(
            stages=[
                RelationalStage(sql=f"SELECT * FROM '{path}' WHERE price > 10"),
                SemanticStage(
                    operations=[
                        {
                            "name": "tax",
                            "type": "code_map",
                            "code": "def transform(doc):\n    return {'tax': doc['price'] * 0.1}",
                        }
                    ]
                ),
                RelationalStage(sql="SELECT id, tax FROM _prev ORDER BY id", reads_prev=True),
            ]
        )
        out = run_compiled(compiled).to_pylist()
        assert out == [
            {"id": 3, "tax": 1.5},
            {"id": 4, "tax": 2.0},
            {"id": 5, "tax": 2.5},
        ]
