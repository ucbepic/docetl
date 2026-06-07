"""End-to-end-ish tests for the filter model-cascade vertical slice.

No real LLM calls: a fake ``runner.api`` plays both proxy and oracle. The proxy
is perfectly calibrated (returns the ground-truth label with a confident score)
and the oracle returns ground truth, so the recall guarantee should keep every
truly-relevant record. Ground truth is smuggled through the rendered prompt so
the fakes can read it.
"""

import re
from types import SimpleNamespace

import pytest

from docetl.operations.filter import CascadeConfig, FilterOperation


def _label_from_messages(messages):
    text = messages[0]["content"]
    m = re.search(r"label=(True|False)", text)
    return m.group(1) == "True"


class FakeAPI:
    """Stands in for APIWrapper: records calls and answers from the prompt."""

    def __init__(self):
        self.proxy_calls = 0
        self.oracle_calls = 0
        self.proxy_cost = 0.0001
        self.oracle_cost = 0.01

    def _classify_with_logprob_with_cost(self, model, messages, labels):
        self.proxy_calls += 1
        gt = _label_from_messages(messages)
        # Perfectly calibrated, confident proxy.
        return gt, 0.95, self.proxy_cost

    def call_llm(self, model, op_type, messages, schema, **kwargs):
        self.oracle_calls += 1
        gt = _label_from_messages(messages)
        key = next(iter(schema.keys()))
        return SimpleNamespace(response={key: gt}, total_cost=self.oracle_cost)


def make_op(cascade=None, model="gpt-4o"):
    api = FakeAPI()
    runner = SimpleNamespace(config={}, api=api, is_cancelled=False)
    config = {
        "name": "is_relevant",
        "type": "filter",
        "prompt": "Doc: {{ input.text }} | label={{ input.gt }}",
        "output": {"schema": {"keep": "bool"}},
        "model": model,
    }
    if cascade is not None:
        config["cascade"] = cascade
    op = FilterOperation(runner, config, model, max_threads=4)
    return op, api


def make_data(n=60, every=3):
    # gt True for one in `every` records.
    return [{"id": i, "text": f"doc{i}", "gt": (i % every == 0)} for i in range(n)]


# ---------------------------------------------------------------------------
# CascadeConfig validation
# ---------------------------------------------------------------------------
def test_cascade_config_defaults():
    c = CascadeConfig(proxy_model="gpt-4o-mini", target=0.9)
    # guarantee defaults to None so each operator applies its own default
    # (filter -> recall) at run time.
    assert c.guarantee is None
    assert c.delta == 0.05
    assert c.label_budget == 400


def test_cascade_config_rejects_bad_guarantee():
    with pytest.raises(ValueError, match="guarantee"):
        CascadeConfig(proxy_model="m", target=0.9, guarantee="f1")


def test_cascade_config_requires_target_in_range():
    with pytest.raises(ValueError):
        CascadeConfig(proxy_model="m", target=1.5)
    with pytest.raises(ValueError):
        CascadeConfig(proxy_model="m")  # missing target


def test_filter_cascade_with_pdf_or_retriever_rejected():
    # Filter inherits the guard from MapOperation.schema.
    for bad in ("pdf_url_key", "retriever"):
        cfg = {
            "name": "f",
            "type": "filter",
            "prompt": "{{ input.x }}?",
            "output": {"schema": {"keep": "bool"}},
            bad: "something",
            "cascade": {"proxy_model": "gpt-4o-mini", "target": 0.9},
        }
        with pytest.raises(ValueError, match=bad):
            FilterOperation.schema.model_validate(cfg)


# ---------------------------------------------------------------------------
# Cascade execution
# ---------------------------------------------------------------------------
def test_cascade_empty_input_short_circuits():
    op, api = make_op(cascade={"proxy_model": "gpt-4o-mini", "target": 0.9})
    out, cost = op.execute([])
    assert out == []
    assert cost == 0.0
    assert api.proxy_calls == 0 and api.oracle_calls == 0


def test_cascade_recall_keeps_all_true_positives():
    data = make_data(n=60, every=3)
    op, api = make_op(
        cascade={
            "proxy_model": "gpt-4o-mini",
            "guarantee": "recall",
            "target": 0.9,
            "delta": 0.1,
        }
    )
    kept, cost = op.execute(data)

    kept_ids = {r["id"] for r in kept}
    true_positive_ids = {r["id"] for r in data if r["gt"]}

    # Recall guarantee with a perfect proxy: every relevant doc is kept.
    assert true_positive_ids <= kept_ids
    # Kept records are the original dicts, unchanged (no filter key leaked in).
    for r in kept:
        assert set(r.keys()) == {"id", "text", "gt"}

    # Proxy runs on every item; the oracle is consulted at least for calibration.
    assert api.proxy_calls == len(data)
    assert api.oracle_calls >= 1


def test_cascade_cost_is_accumulated_from_both_models():
    data = make_data(n=45, every=4)
    op, api = make_op(
        cascade={"proxy_model": "gpt-4o-mini", "guarantee": "recall", "target": 0.9}
    )
    _, cost = op.execute(data)
    expected = api.proxy_calls * api.proxy_cost + api.oracle_calls * api.oracle_cost
    assert cost == pytest.approx(expected, rel=1e-9)
    assert cost > 0


def test_cascade_progress_ticks_tracker(cascade_cache):
    from docetl.progress.tracker import ProgressTracker, set_active_tracker

    tracker = ProgressTracker()
    set_active_tracker(tracker)
    tracker.op_start("op", "filter", "gpt-4o", total=8)
    try:
        op, api = make_op(
            cascade={
                "proxy_model": "gpt-4o-mini",
                "guarantee": "recall",
                "target": 0.9,
                "label_budget": 5,
            }
        )
        op.execute(make_data(8))
        snap = tracker.snapshot().get("op")
        assert snap.phase is None
        assert api.proxy_calls == 8
        assert api.oracle_calls > 0
        assert snap.completed == api.oracle_calls
        assert snap.total == 5  # label_budget, not n_items
    finally:
        set_active_tracker(None)


def test_format_cascade_plan_lines():
    from docetl.operations.utils.cascade_runner import format_cascade_plan_lines

    lines = format_cascade_plan_lines(
        {
            "proxy_model": "gpt-4o-mini",
            "target": 0.95,
            "label_budget": 20,
        },
        op_type="filter",
        oracle_model="gpt-4o",
    )
    text = "\n".join(lines)
    assert "gpt-4o-mini" in text
    assert "gpt-4o" in text
    assert "recall" in text
    assert "95%" in text
    assert "≤20 oracle labels" in text


def test_build_phase_bypasses_cascade():
    data = make_data(n=12, every=3)
    op, api = make_op(
        cascade={"proxy_model": "gpt-4o-mini", "guarantee": "recall", "target": 0.9}
    )
    # is_build=True must NOT take the cascade path (it would call the fake api
    # in a way the normal map path doesn't); instead it should fall through to
    # the map path, which we don't exercise here -- so just assert the cascade
    # adapters were never invoked.
    try:
        op.execute(data, is_build=True)
    except Exception:
        pass  # normal map path needs a fuller runner; we only assert the branch
    assert api.proxy_calls == 0
