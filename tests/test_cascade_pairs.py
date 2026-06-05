"""Tests for the resolve/equijoin pair-cascade path (`_cascade_match_pairs`).

No real LLM calls: a fake ``runner.api`` plays proxy and oracle, reading each
pair's ground-truth match flag (smuggled through the rendered prompt). Under
the precision guarantee with a perfect proxy, every predicted-positive pair is
a true match (precision), and cost is summed across both models.
"""

import re
from types import SimpleNamespace

import pytest

from docetl.operations.equijoin import EquijoinOperation
from docetl.operations.resolve import ResolveOperation


def _match_from_messages(messages):
    return re.search(r"match=(True|False)", messages[0]["content"]).group(1) == "True"


class FakeAPI:
    def __init__(self):
        self.proxy_calls = 0
        self.oracle_calls = 0
        self.proxy_cost = 0.0001
        self.oracle_cost = 0.01

    def _classify_with_logprob_with_cost(self, model, messages, labels):
        self.proxy_calls += 1
        gt = _match_from_messages(messages)
        return gt, 0.95, self.proxy_cost

    def call_llm(self, model, op_type, messages, schema, **kwargs):
        self.oracle_calls += 1
        gt = _match_from_messages(messages)
        return SimpleNamespace(response={"is_match": gt}, total_cost=self.oracle_cost)

    def parse_llm_response(self, response, schema, **kwargs):
        return [response]


def _build(cls, config):
    """Construct an operation without running __init__/syntax_check (which need
    a full runner). We only exercise the cascade pair method here."""
    api = FakeAPI()
    runner = SimpleNamespace(config={}, api=api, is_cancelled=False)
    op = cls.__new__(cls)
    op.runner = runner
    op.config = config
    op.default_model = "gpt-4o"
    op.console = SimpleNamespace(log=lambda *a, **k: None)
    op.bypass_cache = False
    op.max_threads = 4
    return op, api


def make_resolve(cascade):
    return _build(
        ResolveOperation,
        {
            "name": "dedupe",
            "type": "resolve",
            "comparison_prompt": "A {{ input1.text }} B {{ input2.text }} match={{ input1.match }}",
            "comparison_model": "gpt-4o",
            "cascade": cascade,
        },
    )


def make_equijoin(cascade):
    return _build(
        EquijoinOperation,
        {
            "name": "join",
            "type": "equijoin",
            "comparison_prompt": "L {{ left.text }} R {{ right.text }} match={{ left.match }}",
            "comparison_model": "gpt-4o",
            "cascade": cascade,
        },
    )


def _precision(labels, truth):
    pos = [i for i, lbl in enumerate(labels) if lbl]
    assert pos, "cascade returned no positives"
    return sum(1 for i in pos if truth[i]) / len(pos)


def make_pairs(n=60):
    # Pair i matches iff i is even. The match flag rides on the left item.
    pairs = []
    for i in range(n):
        match = i % 2 == 0
        pairs.append(({"text": f"l{i}", "match": match}, {"text": f"r{i}"}))
    return pairs


CASCADE = {"proxy_model": "gpt-4o-mini", "target": 0.9, "delta": 0.1}


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------
def test_resolve_pairs_precision_predicted_positives_are_true():
    pairs = make_pairs(60)
    op, api = make_resolve(CASCADE)
    labels, cost = op._cascade_match_pairs(pairs, blocking_keys=[])

    assert len(labels) == len(pairs)
    true_match = [p[0]["match"] for p in pairs]
    # Precision guarantee: of the predicted-positive pairs, >= target are true
    # (a few boundary false positives are allowed by the 1-delta guarantee).
    assert _precision(labels, true_match) >= CASCADE["target"] - 0.1
    # The cascade discriminates (doesn't just label everything positive).
    assert not all(labels)

    assert api.proxy_calls == len(pairs)
    expected = api.proxy_calls * api.proxy_cost + api.oracle_calls * api.oracle_cost
    assert cost == pytest.approx(expected, rel=1e-9)
    assert cost > 0


def test_resolve_pairs_empty():
    op, api = make_resolve(CASCADE)
    labels, cost = op._cascade_match_pairs([], blocking_keys=[])
    assert labels == [] and cost == 0.0
    assert api.proxy_calls == 0 and api.oracle_calls == 0


# ---------------------------------------------------------------------------
# equijoin
# ---------------------------------------------------------------------------
def test_equijoin_pairs_precision_predicted_positives_are_true():
    pairs = make_pairs(60)
    op, api = make_equijoin(CASCADE)
    labels, cost = op._cascade_match_pairs(pairs)

    assert len(labels) == len(pairs)
    true_match = [p[0]["match"] for p in pairs]
    assert _precision(labels, true_match) >= CASCADE["target"] - 0.1
    assert not all(labels)

    assert api.proxy_calls == len(pairs)
    expected = api.proxy_calls * api.proxy_cost + api.oracle_calls * api.oracle_cost
    assert cost == pytest.approx(expected, rel=1e-9)


def test_equijoin_pairs_empty():
    op, api = make_equijoin(CASCADE)
    labels, cost = op._cascade_match_pairs([])
    assert labels == [] and cost == 0.0
    assert api.proxy_calls == 0 and api.oracle_calls == 0
