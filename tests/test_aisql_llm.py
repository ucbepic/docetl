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

from docetl.aisql import run_sql

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
        out = run_sql(f"SELECT ai_summarize(text) AS summary FROM '{path}'").to_pylist()
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
        out = run_sql(
            f"SELECT id FROM '{path}' "
            "WHERE ai_filter(text, 'Is this text about animals?') ORDER BY id"
        ).to_pylist()
        assert [r["id"] for r in out] == [1, 3]

    @pytest.mark.flaky(reruns=2)
    def test_relational_and_ai_filter(self, tmp_path):
        path = _file(
            tmp_path,
            [
                {"id": 1, "price": 5, "desc": "USB cable"},
                {"id": 2, "price": 50, "desc": "wireless headphones"},
                {"id": 3, "price": 100, "desc": "leather wallet"},
                {"id": 4, "price": 20, "desc": "bluetooth speaker"},
            ],
        )
        # price > 10 -> {2,3,4}; electronics -> {2,4}
        out = run_sql(
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
        out = run_sql(
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
        out = run_sql(
            f"SELECT cat, ai_agg(item, 'List the items in one sentence') AS items "
            f"FROM '{path}' GROUP BY cat ORDER BY cat"
        ).to_pylist()
        assert [r["cat"] for r in out] == ["fruit", "vegetable"]
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
        out = run_sql(
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
        out = run_sql(
            f"SELECT name FROM ai_resolve('{path}', on := name, "
            "prompt := 'Do these refer to the same person?')"
        ).to_pylist()
        assert len(out) == 3  # resolve canonicalizes values, keeps rows
        # the two John variants collapse to one canonical name -> 2 distinct
        assert len({r["name"] for r in out}) == 2
