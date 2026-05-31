"""Integration test for the progress tracker against a *real* pipeline run.

Runs a small two-operator pipeline (map + filter) end-to-end through
``DSLRunner`` with real ``gpt-4.1-nano`` calls -- the cheapest OpenAI model --
and asserts that the structured telemetry the TUI consumes reconciles with the
runner's own accounting (per-op counts, total cost, completion).

Requires ``OPENAI_API_KEY`` and network access to api.openai.com; skipped
otherwise so the rest of the suite stays runnable offline.
"""

import os

import pytest

from docetl.progress.tracker import ProgressTracker, set_active_tracker
from docetl.runner import DSLRunner

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY for a real (nano) pipeline run",
)

MODEL = "gpt-4.1-nano"

DATA = [
    {"text": "I was double charged and I want a refund right now."},
    {"text": "The new dashboard is wonderful, thanks so much!"},
    {"text": "My package never arrived and tracking is stuck."},
    {"text": "Could you add a dark mode some day? Low priority."},
    {"text": "We are completely locked out and launching in an hour."},
    {"text": "How do I invite a teammate to my workspace?"},
]


def _config(tmp_path):
    data_path = tmp_path / "input.json"
    import json

    data_path.write_text(json.dumps(DATA))
    return {
        "default_model": MODEL,
        "bypass_cache": True,
        "datasets": {"tickets": {"type": "file", "path": str(data_path)}},
        "operations": [
            {
                "name": "classify",
                "type": "map",
                "prompt": (
                    "Classify this support ticket. Ticket: \"{{ input.text }}\". "
                    "Give a category and a priority (low/medium/high)."
                ),
                "output": {"schema": {"category": "string", "priority": "string"}},
            },
            {
                "name": "urgent_only",
                "type": "filter",
                "prompt": (
                    "Is this ticket urgent and time-sensitive? Ticket: "
                    "\"{{ input.text }}\" priority {{ input.priority }}."
                ),
                "output": {"schema": {"keep": "boolean"}},
            },
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "triage",
                    "input": "tickets",
                    "operations": ["classify", "urgent_only"],
                }
            ],
            "output": {"type": "file", "path": str(tmp_path / "out.json")},
        },
    }


def test_tracker_reconciles_with_real_run(tmp_path):
    runner = DSLRunner(_config(tmp_path), max_threads=4)

    # Wire up the tracker exactly as run_with_tui does, minus the UI.
    tracker = ProgressTracker(concurrency=4)
    runner.progress_tracker = tracker
    runner._tui_active = True
    set_active_tracker(tracker)
    tracker.pipeline_start(runner.list_pipeline_operations())
    try:
        total_cost = runner.load_run_save()
    finally:
        set_active_tracker(None)

    state = tracker.snapshot()
    assert state.finished

    ops = {op.name: op for op in state.ops}
    assert set(ops) == {"triage/classify", "triage/urgent_only"}

    classify = ops["triage/classify"]
    assert classify.status == "done"
    # Every input document was processed by the map op.
    assert classify.total == len(DATA)
    assert classify.completed == len(DATA)
    assert classify.out_count == len(DATA)

    urgent = ops["triage/urgent_only"]
    assert urgent.status == "done"
    # The filter keeps a subset; some of these tickets are clearly urgent.
    assert 0 < urgent.out_count <= len(DATA)

    # Real nano calls cost a little; the tracker's per-op costs reconcile with
    # the runner's own total (which is what the user is billed for).
    assert total_cost > 0
    assert state.total_cost == pytest.approx(total_cost, rel=0.01, abs=1e-6)
    assert sum(op.cost for op in state.ops) == pytest.approx(total_cost, rel=0.01, abs=1e-6)
