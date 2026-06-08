"""Caching + reporting tests for the model cascade.

An identical re-run (same op config + same data) should reuse the cached result
and make zero new proxy/oracle calls; ``bypass_cache`` forces recomputation.
``cascade_stats`` is exposed on the operation for programmatic reporting. The
cascade cache is isolated to an in-memory fake by the autouse ``cascade_cache``
fixture (see conftest.py), so tests don't touch the real on-disk cache.
"""

import re
from types import SimpleNamespace

from docetl.operations.filter import FilterOperation


def _label_from_messages(messages):
    return re.search(r"label=(True|False)", messages[0]["content"]).group(1) == "True"


class FakeAPI:
    def __init__(self):
        self.proxy_calls = 0
        self.oracle_calls = 0

    def _classify_with_logprob_with_cost(self, model, messages, labels):
        self.proxy_calls += 1
        return _label_from_messages(messages), 0.95, 0.0001

    def call_llm(self, model, op_type, messages, schema, **kwargs):
        self.oracle_calls += 1
        key = next(iter(schema.keys()))
        return SimpleNamespace(
            response={key: _label_from_messages(messages)}, total_cost=0.01
        )


def make_filter(bypass_cache=False):
    api = FakeAPI()
    runner = SimpleNamespace(config={}, api=api, is_cancelled=False)
    config = {
        "name": "is_relevant",
        "type": "filter",
        "prompt": "Doc {{ input.text }} label={{ input.gt }}",
        "output": {"schema": {"keep": "bool"}},
        "model": "gpt-4o",
        "cascade": {"proxy_model": "gpt-4o-mini", "guarantee": "recall", "target": 0.9},
        "bypass_cache": bypass_cache,
    }
    return FilterOperation(runner, config, "gpt-4o", max_threads=4), api


def make_data(n=40):
    return [{"id": i, "text": f"d{i}", "gt": (i % 3 == 0)} for i in range(n)]


def test_identical_rerun_hits_cache(cascade_cache):
    data = make_data()

    op1, api1 = make_filter()
    kept1, cost1 = op1.execute(data)
    assert api1.proxy_calls > 0 and api1.oracle_calls > 0  # first run does work
    assert len(cascade_cache.store) == 1  # result was cached

    # A fresh op with the same config + data must hit the cache: no new calls.
    op2, api2 = make_filter()
    kept2, cost2 = op2.execute(data)
    assert api2.proxy_calls == 0 and api2.oracle_calls == 0
    assert cost2 == cost1
    assert [r["id"] for r in kept2] == [r["id"] for r in kept1]
    # Stats are still reported on a cache hit.
    assert op2.cascade_stats.n_items == len(data)


def test_bypass_cache_recomputes(cascade_cache):
    data = make_data()
    op1, _ = make_filter()
    op1.execute(data)

    op2, api2 = make_filter(bypass_cache=True)
    op2.execute(data)
    # bypass_cache must re-run the proxy/oracle instead of reading the cache.
    assert api2.proxy_calls > 0


def test_different_data_misses_cache(cascade_cache):
    op1, _ = make_filter()
    op1.execute(make_data(40))

    op2, api2 = make_filter()
    op2.execute(make_data(30))  # different dataset -> different key -> recompute
    assert api2.proxy_calls > 0
    assert len(cascade_cache.store) == 2


def test_cascade_stats_exposed(cascade_cache):
    op, _ = make_filter()
    op.execute(make_data())
    s = op.cascade_stats
    assert s.n_items == 40
    assert s.proxy_calls == 40
    assert 0.0 <= s.escalation_rate <= 1.0
    assert s.guarantee == "recall"
