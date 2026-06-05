"""Tests for the single-enum map model-cascade path.

No real LLM calls: a fake ``runner.api`` plays proxy and oracle, reading the
ground-truth enum value smuggled through the rendered prompt. A perfectly
calibrated proxy under the accuracy guarantee should label every row, and all
rows are preserved (map keeps its rows; only the enum field is filled).
"""

import re
from types import SimpleNamespace

import pytest

from docetl.operations.map import MapOperation


def _cat_from_messages(messages):
    return re.search(r"cat=(\w+)", messages[0]["content"]).group(1)


class FakeAPI:
    def __init__(self):
        self.proxy_calls = 0
        self.oracle_calls = 0
        self.proxy_cost = 0.0001
        self.oracle_cost = 0.01

    def _classify_with_logprob_with_cost(self, model, messages, labels):
        self.proxy_calls += 1
        return _cat_from_messages(messages), 0.95, self.proxy_cost

    def call_llm(self, model, op_type, messages, schema, **kwargs):
        self.oracle_calls += 1
        key = next(k for k in schema if k != "_short_explanation")
        return SimpleNamespace(
            response={key: _cat_from_messages(messages)}, total_cost=self.oracle_cost
        )


CATS = ["bug", "feature", "question"]


def make_op(output_schema, cascade=None, model="gpt-4o"):
    api = FakeAPI()
    runner = SimpleNamespace(config={}, api=api, is_cancelled=False)
    config = {
        "name": "categorize",
        "type": "map",
        "prompt": "Issue: {{ input.text }} cat={{ input.cat }}",
        "output": {"schema": output_schema},
        "model": model,
    }
    if cascade is not None:
        config["cascade"] = cascade
    return MapOperation(runner, config, model, max_threads=4), api


def make_data(n=60):
    return [{"id": i, "text": f"t{i}", "cat": CATS[i % len(CATS)]} for i in range(n)]


def test_map_cascade_labels_all_rows_and_preserves_them():
    data = make_data(60)
    op, api = make_op(
        {"category": "enum[bug, feature, question]"},
        cascade={"proxy_model": "gpt-4o-mini", "target": 0.9, "delta": 0.1},
    )
    out, cost = op.execute(data)

    # Every row is kept (map semantics) and gets the enum field filled.
    assert len(out) == len(data)
    for item, o in zip(data, out):
        assert o["id"] == item["id"]
        assert o["category"] == item["cat"]  # perfect proxy/oracle agreement

    assert api.proxy_calls == len(data)  # proxy runs on all
    assert api.oracle_calls >= 1  # calibration consults the oracle
    expected = api.proxy_calls * api.proxy_cost + api.oracle_calls * api.oracle_cost
    assert cost == pytest.approx(expected, rel=1e-9)


def test_map_cascade_empty_input():
    op, api = make_op(
        {"category": "enum[a, b]"},
        cascade={"proxy_model": "gpt-4o-mini", "target": 0.9},
    )
    out, cost = op.execute([])
    assert out == [] and cost == 0.0
    assert api.proxy_calls == 0 and api.oracle_calls == 0


def test_map_cascade_rejects_non_enum_output():
    op, _ = make_op(
        {"summary": "string"},
        cascade={"proxy_model": "gpt-4o-mini", "target": 0.9},
    )
    with pytest.raises(ValueError, match="enum"):
        op.execute(make_data(4))


def test_map_cascade_rejects_multi_key_output():
    op, _ = make_op(
        {"category": "enum[a, b]", "other": "string"},
        cascade={"proxy_model": "gpt-4o-mini", "target": 0.9},
    )
    with pytest.raises(ValueError, match="exactly one output key"):
        op.execute(make_data(4))
