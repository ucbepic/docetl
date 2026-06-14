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
