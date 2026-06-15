"""End-to-end AI-SQL tests against a real model (gpt-4o-mini).

These exercise the actual AI operators — the LLM-free tests in
test_aisql.py prove the plumbing; these prove a query really runs through
DuckDB + DocETL + an LLM and returns sensible output. Data is tiny and
chosen so the model's answer is unambiguous; judgment-based cases get
rerun on the occasional LLM wobble.
"""

import json

import pytest
from dotenv import load_dotenv

from docetl.aisql import run_sql as _run_sql


def run_sql_table(query, **kw):
    table, _cost = _run_sql(query, **kw)
    return table

load_dotenv()


@pytest.fixture(autouse=True)
def _aisql_llm_env(monkeypatch):
    # Per-test (not module-global, so it doesn't leak) and cache-bypassed:
    # these assert real model behavior, so a stale/poisoned cache entry must
    # not stand in for a live call.
    monkeypatch.setattr("docetl._config.default_model", "gpt-4o-mini")
    monkeypatch.setattr("docetl._config.bypass_cache", True)


def _file(tmp_path, rows, name="data.json"):
    path = tmp_path / name
    path.write_text(json.dumps(rows))
    return str(path)


class TestMapAndFilter:
    def test_ai_summarize_preserves_rows(self, tmp_path):
        path = _file(
            tmp_path,
            [
                {"id": 1, "text": "The cat sat on the mat all afternoon."},
                {"id": 2, "text": "Quarterly revenue rose 12% year over year."},
            ],
        )
        out = run_sql_table(f"SELECT ai_summarize(text) AS summary FROM '{path}'").to_pylist()
        assert len(out) == 2
        assert all(isinstance(r["summary"], str) and r["summary"] for r in out)

    @pytest.mark.flaky(reruns=2)
    def test_ai_filter_keeps_relevant_rows(self, tmp_path):
        path = _file(
            tmp_path,
            [
                {"id": 1, "text": "The lion is a large cat that lives in Africa."},
                {"id": 2, "text": "The quarterly earnings report beat expectations."},
                {"id": 3, "text": "Dolphins are intelligent marine mammals."},
                {"id": 4, "text": "The new tax policy affects small businesses."},
            ],
        )
        out = run_sql_table(
            f"SELECT id FROM '{path}' "
            "WHERE ai_filter(text, 'Is this text about animals?') ORDER BY id"
        ).to_pylist()
        assert [r["id"] for r in out] == [1, 3]

    @pytest.mark.flaky(reruns=3)
    def test_relational_and_ai_filter(self, tmp_path):
        path = _file(
            tmp_path,
            [
                {"id": 1, "price": 5, "desc": "laptop computer"},
                {"id": 2, "price": 50, "desc": "laptop computer"},
                {"id": 3, "price": 100, "desc": "leather wallet"},
                {"id": 4, "price": 20, "desc": "wireless bluetooth speaker"},
            ],
        )
        # price > 10 -> {2,3,4}; electronics -> {2,4}
        out = run_sql_table(
            f"SELECT id FROM '{path}' WHERE price > 10 "
            "AND ai_filter(desc, 'Is this an electronic device?') ORDER BY id"
        ).to_pylist()
        assert [r["id"] for r in out] == [2, 4]

    @pytest.mark.flaky(reruns=2)
    def test_ai_score_comparison(self, tmp_path):
        path = _file(
            tmp_path,
            [
                {"id": 1, "text": "I absolutely love this — best product ever!"},
                {"id": 2, "text": "Terrible experience, would not recommend."},
                {"id": 3, "text": "This made my day, fantastic and delightful!"},
            ],
        )
        out = run_sql_table(
            f"SELECT id FROM '{path}' "
            "WHERE ai_score(text, 'Rate the positivity from 0 to 1') > 0.5 ORDER BY id"
        ).to_pylist()
        assert [r["id"] for r in out] == [1, 3]


class TestReduce:
    @pytest.mark.flaky(reruns=2)
    def test_group_by_ai_agg(self, tmp_path):
        path = _file(
            tmp_path,
            [
                {"cat": "fruit", "item": "apple"},
                {"cat": "fruit", "item": "banana"},
                {"cat": "vegetable", "item": "carrot"},
            ],
        )
        out = run_sql_table(
            f"SELECT cat, ai_agg(item, 'List the items in one sentence') AS items "
            f"FROM '{path}' GROUP BY cat"
        ).to_pylist()
        # one row per group; order is not guaranteed (reduce doesn't preserve
        # it, and ORDER BY isn't yet carried into the final projection)
        assert {r["cat"] for r in out} == {"fruit", "vegetable"}
        assert all(isinstance(r["items"], str) and r["items"] for r in out)


class TestJoin:
    @pytest.mark.flaky(reruns=2)
    def test_ai_match_join(self, tmp_path):
        products = _file(
            tmp_path,
            [{"pid": 1, "name": "Apple iPhone 15"}, {"pid": 2, "name": "Samsung Galaxy S24"}],
            "products.json",
        )
        listings = _file(
            tmp_path,
            [{"lid": 1, "title": "iPhone 15 by Apple"}, {"lid": 2, "title": "Sony WH-1000XM5 headphones"}],
            "listings.json",
        )
        out = run_sql_table(
            f"SELECT name, title FROM '{products}' p "
            f"JOIN '{listings}' l ON ai_match(p.name, l.title, 'Are these the same product?')"
        ).to_pylist()
        # Only the iPhone pair matches.
        assert len(out) == 1
        assert "iPhone" in out[0]["name"] and "iPhone" in out[0]["title"]


class TestResolve:
    @pytest.mark.flaky(reruns=2)
    def test_ai_resolve_canonicalizes(self, tmp_path):
        path = _file(
            tmp_path,
            [{"name": "John Smith"}, {"name": "J. Smith"}, {"name": "Jane Doe"}],
            "customers.json",
        )
        out = run_sql_table(
            f"SELECT name FROM ai_resolve('{path}', on := name, "
            "prompt := 'Do these refer to the same person?')"
        ).to_pylist()
        assert len(out) == 3  # resolve canonicalizes values, keeps rows
        # the two John variants collapse to one canonical name -> 2 distinct
        assert len({r["name"] for r in out}) == 2


class TestScale:
    @pytest.mark.flaky(reruns=3)
    def test_relational_filter_caps_ai_filter_input(self, tmp_path, monkeypatch):
        # A million-row Parquet file where a relational predicate narrows to
        # exactly 100 rows; the AI filter must then run on those 100 only,
        # never the full million. Proves the DuckDB/DocETL split: without it
        # the LLM would attempt 1M calls.
        import pyarrow as pa
        import pyarrow.parquet as pq

        import docetl

        n, rare = 1_000_000, 100
        animal = [
            "Lions roam the African savanna.",
            "Dolphins are intelligent marine mammals.",
            "The eagle soared over the canyon.",
            "Honeybees pollinate the wildflowers.",
        ]
        finance = [
            "Q3 revenue rose 8% year over year.",
            "The central bank held rates steady.",
            "Shares dipped on weak guidance.",
        ]
        region = ["common"] * (n - rare) + ["rare"] * rare
        text = (
            [finance[i % len(finance)] for i in range(n - rare)]
            + [animal[i % len(animal)] for i in range(60)]
            + [finance[i % len(finance)] for i in range(40)]
        )
        path = tmp_path / "big.parquet"
        pq.write_table(pa.table({"id": list(range(n)), "region": region, "text": text}), path)

        # Spy on the rows handed to each semantic (AI) stage.
        seen = []
        original = docetl.from_arrow
        monkeypatch.setattr(
            docetl,
            "from_arrow",
            lambda t, name="data": (seen.append(t.num_rows), original(t, name=name))[1],
        )

        out = run_sql_table(
            f"SELECT id FROM '{path}' "
            "WHERE region = 'rare' AND ai_filter(text, 'Is this text about animals?')"
        ).to_pylist()

        # The LLM saw exactly the 100 relationally-filtered rows, not 1M.
        assert seen == [rare]
        # Everything kept is within the rare 100, and it found some animals.
        assert out and all(r["id"] >= n - rare for r in out)
        assert len(out) <= rare
