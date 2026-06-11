"""Tests for the embedding-model cascade proxy.

When ``cascade.proxy_model`` names an embedding model, the cascade fits a
logistic head on an oracle-labeled training slice instead of running a cheap
LLM per item. No real API calls: a fake ``runner.api`` serves embeddings that
are linearly separable by the ground-truth label smuggled into the rendered
prompt, and plays the oracle.
"""

import json
import re
from types import SimpleNamespace

import pytest

from docetl.operations.filter import FilterOperation
from docetl.operations.utils.cascade_runner import _is_embedding_model


def _label_from_text(text: str) -> bool:
    return re.search(r"label=(True|False)", text).group(1) == "True"


class FakeEmbeddingAPI:
    """Embeddings separable by ground truth; oracle answers ground truth."""

    def __init__(self):
        self.embedding_calls = 0
        self.logprob_calls = 0
        self.oracle_calls = 0
        self.oracle_cost = 0.01

    def gen_embedding(self, model, input):
        self.embedding_calls += 1
        texts = json.loads(input)
        data = []
        for i, text in enumerate(texts):
            gt = _label_from_text(text)
            jitter = 0.01 * (i % 7)
            vec = [1.0 + jitter, jitter] if gt else [jitter, 1.0 + jitter]
            data.append({"embedding": vec})
        return {"data": data}

    def _classify_with_logprob_with_cost(self, model, messages, labels):
        self.logprob_calls += 1
        raise AssertionError("logprob proxy must not run for embedding models")

    def call_llm(self, model, op_type, messages, schema, **kwargs):
        self.oracle_calls += 1
        gt = _label_from_text(messages[0]["content"])
        key = next(iter(schema.keys()))
        return SimpleNamespace(response={key: gt}, total_cost=self.oracle_cost)


def make_op(cascade):
    api = FakeEmbeddingAPI()
    runner = SimpleNamespace(config={}, api=api, is_cancelled=False)
    config = {
        "name": "is_relevant",
        "type": "filter",
        "prompt": "Doc: {{ input.text }} | label={{ input.gt }}",
        "output": {"schema": {"keep": "bool"}},
        "model": "gpt-4o",
        "cascade": cascade,
    }
    op = FilterOperation(runner, config, "gpt-4o", max_threads=4)
    return op, api


def make_data(n=120, every=3):
    return [{"id": i, "text": f"doc{i}", "gt": (i % every == 0)} for i in range(n)]


def test_is_embedding_model_detection():
    assert _is_embedding_model("text-embedding-3-small")
    assert not _is_embedding_model("gpt-4o-mini")
    assert not _is_embedding_model("definitely-not-a-real-model")


def test_embedding_cascade_keeps_true_positives_without_llm_proxy():
    data = make_data(n=120, every=3)
    op, api = make_op(
        cascade={
            "proxy_model": "text-embedding-3-small",
            "guarantee": "recall",
            "target": 0.9,
            "delta": 0.1,
            "label_budget": 40,
        }
    )
    kept, cost = op.execute(data)

    kept_ids = {r["id"] for r in kept}
    true_positive_ids = {r["id"] for r in data if r["gt"]}

    # Perfectly separable embeddings: every relevant doc is kept.
    assert true_positive_ids <= kept_ids
    # The proxy is the fitted head — zero per-item LLM proxy calls.
    assert api.logprob_calls == 0
    assert api.embedding_calls >= 1
    # Oracle spend stays within the label budget (training rows are
    # memoized, so threshold-search re-draws don't double-bill).
    assert api.oracle_calls <= 40
    assert api.oracle_calls < len(data)
    # Cost = oracle calls only (fake embeddings are free).
    assert cost == pytest.approx(api.oracle_calls * api.oracle_cost)

    # Kept records are original dicts, no filter key leaked.
    for r in kept:
        assert set(r.keys()) == {"id", "text", "gt"}


def test_embedding_cascade_training_rows_keep_oracle_answers():
    data = make_data(n=90, every=2)
    op, api = make_op(
        cascade={
            "proxy_model": "text-embedding-3-small",
            "guarantee": "recall",
            "target": 0.9,
            "label_budget": 30,
        }
    )
    kept, _ = op.execute(data)
    kept_ids = {r["id"] for r in kept}
    # With a ground-truth oracle and separable scores, output should match
    # ground truth exactly on positives (recall) — spot-check both classes.
    assert all(r["id"] in kept_ids for r in data if r["gt"])


def test_embedding_cascade_single_class_falls_back_to_oracle():
    data = [{"id": i, "text": f"doc{i}", "gt": False} for i in range(40)]
    op, api = make_op(
        cascade={
            "proxy_model": "text-embedding-3-small",
            "guarantee": "recall",
            "target": 0.9,
            "label_budget": 20,
        }
    )
    kept, _ = op.execute(data)
    assert kept == []  # nothing relevant, nothing kept
    # Head couldn't be fit (one class) — every distinct row went to the oracle.
    assert api.oracle_calls == len(data)
    assert op.cascade_stats.escalation_rate == 1.0


def test_embedding_cascade_result_is_cached():
    data = make_data(n=60, every=3)
    cascade_cfg = {
        "proxy_model": "text-embedding-3-small",
        "guarantee": "recall",
        "target": 0.9,
        "label_budget": 30,
    }
    op, api = make_op(cascade_cfg)
    kept_first, _ = op.execute(data)
    oracle_after_first = api.oracle_calls

    op2, api2 = make_op(cascade_cfg)
    kept_second, _ = op2.execute(data)
    assert {r["id"] for r in kept_second} == {r["id"] for r in kept_first}
    assert api2.oracle_calls == 0  # served from the cascade cache
