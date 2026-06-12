"""Tests for the interactive progress view (issue #487).

Three sections, all offline except the final integration test (which makes real
gpt-4.1-nano calls and is skipped without an API key):

1. the progress tracker (state model, thread-safety)
2. operator profiles (per-type units, provenance, grid sizing)
3. an end-to-end reconciliation check against a real pipeline run
"""

import os
import threading

import pytest

from docetl.progress.events import OpState, RunState
from docetl.progress.tracker import (
    ProgressTracker,
    active_tracker,
    set_active_tracker,
)
from docetl.runner import DSLRunner
from docetl.tui.profiles import get_profile


# =============================================================================
# 1. Progress tracker
# =============================================================================
def test_tracker_lifecycle():
    t = ProgressTracker(concurrency=8)
    t.pipeline_start(
        [
            ("s1", "s1/classify", "map", "gpt-4o-mini"),
            ("s1", "s1/keep", "filter", "gpt-4o-mini"),
        ]
    )
    state = t.snapshot()
    assert [op.name for op in state.ops] == ["s1/classify", "s1/keep"]
    assert all(op.status == "queued" for op in state.ops)

    t.op_start("s1/classify", "map", "gpt-4o-mini", total=10)
    op = state.get("s1/classify")
    assert op.status == "running" and op.total == 10

    for _ in range(7):
        t.tick()
    t.doc_error()
    assert op.completed == 7 and op.errors == 1

    t.op_done(
        "s1/classify",
        cost=0.5,
        prompt_tokens=1000,
        completion_tokens=200,
        outputs=[{"x": i} for i in range(10)],
    )
    assert op.status == "done"
    assert op.completed == 10  # forced to total on completion
    assert op.out_count == 10 and op.tokens == 1200 and op.cost == 0.5
    assert state.total_cost == 0.5 and state.done_ops == 1

    t.pipeline_done()
    assert t.snapshot().finished


def test_tick_caps_at_total():
    t = ProgressTracker()
    t.op_start("op", "map", None, total=3)
    for _ in range(10):
        t.tick()
    assert t.snapshot().get("op").completed == 3


def test_set_phase_resets_to_the_real_unit_count():
    # RichLoopBar reports the actual number of units it will tick (e.g. groups
    # for reduce), overriding the input-doc count guessed in containers.py.
    t = ProgressTracker()
    t.op_start("op", "reduce", None, total=100)  # 100 input docs guessed
    t.tick()
    t.set_phase(5)  # reduce will actually tick 5 groups
    op = t.snapshot().get("op")
    assert op.total == 5 and op.completed == 0


def test_set_phase_accepts_label_for_tui():
    t = ProgressTracker()
    t.op_start("op", "filter", None, total=100)
    t.set_phase(40, label="proxy (gpt-4o-mini)")
    op = t.snapshot().get("op")
    assert op.total == 40 and op.completed == 0 and op.phase == "proxy (gpt-4o-mini)"
    t.clear_phase()
    assert t.snapshot().get("op").phase is None


def test_add_outputs_streams_documents_during_run():
    # Finished docs are appended to the current op's sample as they complete,
    # so the detail pane can show them before the whole op finishes.
    t = ProgressTracker()
    t.op_start("op", "map", None, total=5)
    t.add_outputs([{"x": 1}])
    t.add_outputs([{"x": 2}, {"x": 3}])
    op = t.snapshot().get("op")
    assert [d["x"] for d in op.outputs] == [1, 2, 3]
    op.sample_cap = 3  # never grows past the cap
    t.add_outputs([{"x": 4}])
    assert len(op.outputs) == 3


def test_active_tracker_hook():
    t = ProgressTracker()
    set_active_tracker(t)
    try:
        assert active_tracker() is t
        t.op_start("op", "map", None, total=5)
        from docetl.operations.utils.progress import RichLoopBar

        bar = RichLoopBar.__new__(RichLoopBar)  # simulate RichLoopBar.update path
        bar.tqdm = None
        bar.update(2)
        assert t.snapshot().get("op").completed == 2
    finally:
        set_active_tracker(None)
    assert active_tracker() is None


def test_pipeline_label_from_yaml_path(tmp_path):
    yaml_file = tmp_path / "filteronly.yaml"
    yaml_file.write_text("default_model: gpt-4o-mini\n")
    runner = DSLRunner.from_yaml(str(yaml_file))
    assert runner.pipeline_label() == "filteronly.yaml"


def test_should_use_tui_reads_top_level_flag(monkeypatch):
    # interactive_ui lives at the top level of the config (next to
    # default_model), not under `pipeline`.
    import sys

    from docetl.runner import DSLRunner

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    runner = DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/x.json"}},
        },
        max_threads=2,
    )
    assert runner._should_use_tui() is False  # absent -> off
    runner.config["interactive_ui"] = True
    assert runner._should_use_tui() is True  # top level -> on
    # the old nested location is ignored
    runner.config["interactive_ui"] = False
    runner.config["pipeline"]["interactive_ui"] = True
    assert runner._should_use_tui() is False


def test_runstate_to_dict():
    s = RunState(run_id="abc")
    s.register(OpState("s", "s/op", "map", "m"))
    d = s.to_dict()
    assert d["run_id"] == "abc"
    assert d["ops"][0]["name"] == "s/op"
    assert d["ops"][0]["status"] == "queued"


def test_concurrent_ticks_have_no_lost_updates():
    """Documents complete on many worker threads at once; the tracker must
    count every tick and error exactly once (no lost updates under the lock)."""
    t = ProgressTracker(concurrency=32)
    t.op_start("op", "map", None, total=8 * 500 + 4 * 500)
    barrier = threading.Barrier(12)

    def do(fn):
        barrier.wait()
        for _ in range(500):
            fn()

    threads = [threading.Thread(target=do, args=(t.tick,)) for _ in range(8)]
    threads += [threading.Thread(target=do, args=(t.doc_error,)) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    op = t.snapshot().get("op")
    assert op.completed == 8 * 500 and op.errors == 4 * 500


def test_concurrent_emit_and_snapshot_is_stable():
    """A UI snapshotting on a timer while ops emit must never raise and must
    observe a monotonically non-decreasing completed count."""
    t = ProgressTracker(concurrency=8)
    t.op_start("op", "map", None, total=2000)
    stop = threading.Event()
    seen = []

    def reader():
        last = 0
        while not stop.is_set():
            c = t.snapshot().get("op").completed
            assert c >= last  # never goes backwards
            last = c
            seen.append(c)

    r = threading.Thread(target=reader)
    r.start()
    for _ in range(2000):
        t.tick()
    stop.set()
    r.join()
    assert t.snapshot().get("op").completed == 2000 and seen


# =============================================================================
# 2. Operator profiles (units, provenance, grid sizing)
# =============================================================================
def test_cell_status_synthesis_while_running():
    op = OpState("s", "s/op", "map")
    op.total, op.completed, op.errors, op.status = 100, 40, 3, "running"
    assert op.cell_status(0, running_band=5) == "error"     # first `errors` cells
    assert op.cell_status(3, running_band=5) == "done"      # then done up to completed
    assert op.cell_status(41, running_band=5) == "running"  # a band past the edge
    assert op.cell_status(80, running_band=5) == "queued"   # then queued


def test_grid_count_uses_work_units_while_running_then_outputs():
    op = OpState("s", "s/resolve", "resolve")
    op.status, op.total = "running", 45  # comparisons in flight
    assert op.grid_count == 45  # one cell per work unit while running

    op.status, op.out_count = "done", 6  # collapsed to 6 output records
    assert op.grid_count == 6  # switches to one cell per output document


def test_grid_count_and_cell_status_for_split_fan_out():
    # split fans 1 input doc into many chunks; the grid must show the chunks,
    # all done, even though the work-unit counter only reached 1.
    op = OpState("s", "s/split", "split")
    op.status, op.total, op.completed, op.out_count = "done", 1, 1, 5
    assert op.grid_count == 5
    assert [op.cell_status(i, 0) for i in range(5)] == ["done"] * 5


def test_done_cell_status_marks_errors_first():
    op = OpState("s", "s/op", "map")
    op.status, op.out_count, op.errors = "done", 4, 1
    assert op.cell_status(0, 0) == "error"
    assert op.cell_status(1, 0) == "done"


def test_cell_cascade_role_distinguishes_proxy_and_oracle():
    op = OpState("s", "s/f", "filter")
    op.status = "done"
    op.out_count = 3
    op.cascade_info = {
        "item_escalated": [False, True, False, True, False],
        "item_proxy_scores": [0.9, 0.3, 0.85, 0.4, 0.95],
        "kept_input_indices": [0, 1, 4],
    }
    assert op.cell_cascade_role(0) == "proxy"   # input 0, not escalated
    assert op.cell_cascade_role(1) == "oracle"  # input 1, escalated
    assert op.cell_cascade_role(2) == "proxy"   # input 4, not escalated
    assert op.cell_cascade_role(3) is None      # out of range


def test_cell_cascade_role_map_no_kept_indices():
    """Map cascade: output index == input index (no filtering)."""
    op = OpState("s", "s/m", "map")
    op.status = "done"
    op.out_count = 4
    op.cascade_info = {
        "item_escalated": [False, True, True, False],
        "item_proxy_scores": [0.9, 0.6, 0.55, 0.95],
    }
    assert op.cell_cascade_role(0) == "proxy"
    assert op.cell_cascade_role(1) == "oracle"
    assert op.cell_cascade_role(2) == "oracle"
    assert op.cell_cascade_role(3) == "proxy"
    assert op.cell_cascade_role(4) is None


def test_cell_cascade_role_returns_none_without_cascade():
    op = OpState("s", "s/f", "filter")
    op.status = "done"
    op.out_count = 5
    assert op.cell_cascade_role(0) is None


def test_units_per_operator_type():
    assert get_profile("reduce").unit == "groups"
    assert get_profile("resolve").unit == "comparisons"
    assert get_profile("split").doc_unit == "chunks"
    assert get_profile("filter").doc_unit == "kept docs"
    # unknown / missing types fall back to the default profile.
    assert get_profile("totally_new_op").unit == "docs"
    assert get_profile(None).provenance is None


def test_reduce_provenance_reports_source_count():
    prof = get_profile("reduce")
    op = OpState("s", "s/r", "reduce")
    assert prof.provenance(op, {"_counts_prereduce_r": 7}) == "combined from 7 input documents"
    assert prof.provenance(op, {"_counts_prereduce_r": 1}) == "combined from 1 input document"
    assert prof.provenance(op, {"no": "meta"}) is None


def test_split_provenance_reads_as_chunk_n_of_m_without_uuid():
    prof = get_profile("split")
    op = OpState("s", "s/split", "split")
    op.outputs = [
        {"split_x_id": "be59-uuid", "split_x_chunk_num": 1},
        {"split_x_id": "be59-uuid", "split_x_chunk_num": 2},
        {"split_x_id": "be59-uuid", "split_x_chunk_num": 3},
    ]
    line = prof.provenance(op, op.outputs[1])
    assert line == "chunk 2 of 3"
    assert "be59-uuid" not in line  # the raw parent id is never shown

    # split's chunk-bookkeeping keys AND the full source field it copies onto
    # every chunk (the split_key, "content") are hidden; "<split_key>_chunk" stays.
    doc = {
        "content": "the entire source document repeated on every chunk",
        "content_chunk": "just this chunk",
        "split_x_id": "be59-uuid",
        "split_x_chunk_num": 2,
    }
    assert prof.consumed_keys(doc) == {"content", "split_x_id", "split_x_chunk_num"}


def test_filter_summary_reports_dropped():
    f = OpState("s", "s/f", "filter")
    f.total, f.out_count = 20, 12
    assert "dropped: 8" in "".join(str(x) for x in get_profile("filter").summary(f))


# =============================================================================
# 3. End-to-end reconciliation against a real pipeline run
# =============================================================================
_REAL_DATA = [
    {"text": "I was double charged and I want a refund right now."},
    {"text": "The new dashboard is wonderful, thanks so much!"},
    {"text": "My package never arrived and tracking is stuck."},
    {"text": "Could you add a dark mode some day? Low priority."},
    {"text": "We are completely locked out and launching in an hour."},
    {"text": "How do I invite a teammate to my workspace?"},
]


def _real_config(tmp_path):
    import json

    data_path = tmp_path / "input.json"
    data_path.write_text(json.dumps(_REAL_DATA))
    return {
        "default_model": "gpt-4.1-nano",  # cheapest model, to keep cost negligible
        "bypass_cache": True,
        "datasets": {"tickets": {"type": "file", "path": str(data_path)}},
        "operations": [
            {
                "name": "classify",
                "type": "map",
                "prompt": 'Classify this ticket: "{{ input.text }}". '
                          "Give a category and a priority (low/medium/high).",
                "output": {"schema": {"category": "string", "priority": "string"}},
            },
            {
                "name": "urgent_only",
                "type": "filter",
                "prompt": 'Is this ticket urgent? "{{ input.text }}" '
                          "(priority {{ input.priority }}).",
                "output": {"schema": {"keep": "boolean"}},
            },
        ],
        "pipeline": {
            "steps": [
                {"name": "triage", "input": "tickets",
                 "operations": ["classify", "urgent_only"]}
            ],
            "output": {"type": "file", "path": str(tmp_path / "out.json")},
        },
    }


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY for a real (nano) pipeline run",
)
def test_tracker_reconciles_with_real_run(tmp_path):
    from docetl.runner import DSLRunner

    runner = DSLRunner(_real_config(tmp_path), max_threads=4)

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
    assert classify.total == classify.completed == classify.out_count == len(_REAL_DATA)

    urgent = ops["triage/urgent_only"]
    assert urgent.status == "done"
    assert 0 < urgent.out_count <= len(_REAL_DATA)  # the filter keeps a subset

    # The tracker's per-op costs reconcile with the runner's billed total.
    assert total_cost > 0
    assert state.total_cost == pytest.approx(total_cost, rel=0.01, abs=1e-6)
    assert sum(op.cost for op in state.ops) == pytest.approx(total_cost, rel=0.01, abs=1e-6)


def test_doc_view_resolves_observability_by_bare_op_name():
    """OpState.name is "step/op"; the doc-detail pane must look up
    _observability_<bare op name> (OpState has no .op_name attribute)."""
    pytest.importorskip("textual")
    from docetl.progress.events import OpState
    from docetl.tui.app import DocetlTUI
    from docetl.tui.profiles import get_profile
    from docetl.progress.tracker import ProgressTracker

    app = DocetlTUI(ProgressTracker())
    op = OpState(step="step_extract", name="step_extract/extract", op_type="map")
    doc = {
        "text": "hello",
        "_observability_extract": {"prompt": "the rendered prompt"},
    }
    rows, prompt, provenance = app._doc_view(op, get_profile(op.op_type), 0, doc)
    assert prompt == "the rendered prompt"


def test_live_tui_drive_headless(tmp_path):
    """Drive the real TUI with textual's pilot: run an actual pipeline in the
    worker thread, navigate ops pane -> grid -> doc cells, and quit."""
    pytest.importorskip("textual")
    import asyncio
    import json

    import docetl
    from docetl.progress.tracker import set_active_tracker
    from docetl.runner import DSLRunner
    from docetl.tui.app import DocetlTUI

    async def drive():
        data = [{"x": i} for i in range(24)]
        frame = (
            docetl.from_list(data)
            .code_map("double", code="def transform(doc): return {'y': doc['x'] * 2}")
            .code_filter("keep_big", code="def transform(doc): return doc['y'] >= 10")
        )
        out_path = str(tmp_path / "out.json")
        runner = DSLRunner(frame._build_config(output_path=out_path), max_threads=4)

        from tqdm import tqdm as _tqdm

        _tqdm.set_lock(threading.RLock())
        tracker = ProgressTracker(concurrency=4)
        runner.progress_tracker = tracker
        runner._tui_active = True
        set_active_tracker(tracker)
        tracker.pipeline_start(runner.list_pipeline_operations())

        app = DocetlTUI(tracker, runner)
        try:
            async with app.run_test(size=(120, 40)) as pilot:
                for _ in range(100):
                    await pilot.pause(0.1)
                    if app.result_cost is not None or app.error is not None:
                        break
                assert app.error is None, f"pipeline failed in TUI: {app.error}"
                assert app.result_cost is not None, "pipeline never finished"

                await pilot.press("down")
                await pilot.press("up")
                await pilot.press("enter")  # focus grid
                for key in ("right", "right", "down", "left", "up"):
                    await pilot.press(key)
                await pilot.pause(0.3)
                app.render_all()  # force a detail-pane build at the cursor
                await pilot.press("tab")
                await pilot.press("q")
        finally:
            set_active_tracker(None)

        out = json.load(open(out_path))
        assert len(out) == 19 and all(r["y"] >= 10 for r in out)
        assert all(op.status == "done" for op in tracker.snapshot().ops)

    asyncio.run(drive())
