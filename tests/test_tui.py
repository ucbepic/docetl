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


def test_tick_cost_streams_cost_during_run():
    t = ProgressTracker()
    t.pipeline_start([("s", "s/op", "map", None)])
    t.op_start("s/op", "map", None, total=10)
    t.tick_cost(0.05)
    t.tick_cost(0.10)
    op = t.snapshot().get("s/op")
    assert op.cost == pytest.approx(0.15)
    assert t.snapshot().total_cost == pytest.approx(0.15)
    # Zero or negative deltas are ignored.
    t.tick_cost(0.0)
    t.tick_cost(-1.0)
    assert op.cost == pytest.approx(0.15)
    # op_done overwrites with the final total.
    t.op_done("s/op", cost=0.15, prompt_tokens=100, completion_tokens=50)
    assert op.cost == pytest.approx(0.15)


def test_richloopbar_update_with_cost():
    t = ProgressTracker()
    set_active_tracker(t)
    try:
        t.op_start("op", "map", None, total=5)
        from docetl.operations.utils.progress import RichLoopBar

        bar = RichLoopBar.__new__(RichLoopBar)
        bar.tqdm = None
        bar.update(1, cost=0.03)
        bar.update(1, cost=0.07)
        op = t.snapshot().get("op")
        assert op.completed == 2
        assert op.cost == pytest.approx(0.10)
    finally:
        set_active_tracker(None)


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


def test_ui_mode_routing():
    """The top-level ``ui`` field routes to the correct mode."""
    from docetl.runner import DSLRunner

    base = {
        "default_model": "gpt-4o-mini",
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": "/tmp/x.json"}},
    }

    # Default (absent or "none") → no UI
    runner = DSLRunner(dict(base), max_threads=2)
    assert runner._should_use_tui() is None

    runner.config["ui"] = "none"
    assert runner._should_use_tui() is None

    # "tui" → tui
    runner.config["ui"] = "tui"
    assert runner._should_use_tui() == "tui"

    # "web" → web
    runner.config["ui"] = "web"
    assert runner._should_use_tui() == "web"

    # _tui_active prevents recursion regardless of mode
    runner._tui_active = True
    assert runner._should_use_tui() is None


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


def test_kill_requested_raises_in_progress_bar():
    """When kill_requested is set, RichLoopBar.update() raises PipelineKilled."""
    from docetl.progress.tracker import PipelineKilled

    t = ProgressTracker()
    set_active_tracker(t)
    try:
        t.op_start("op", "map", None, total=10)
        from docetl.operations.utils.progress import RichLoopBar

        bar = RichLoopBar.__new__(RichLoopBar)
        bar.tqdm = None
        bar.update(1)  # fine
        t.kill_requested = True
        with pytest.raises(PipelineKilled):
            bar.update(1)
    finally:
        t.kill_requested = False
        set_active_tracker(None)


def test_feedback_store():
    from docetl.tui.web_reporter import FeedbackStore

    fb = FeedbackStore()
    assert not fb.has_any
    fb.add_pipeline_feedback("too aggressive")
    assert fb.has_any
    fb.add_doc_feedback("classify", 3, {"cat": "billing"}, "should be refund")
    d = fb.to_dict()
    assert len(d["pipeline_feedback"]) == 1
    assert d["pipeline_feedback"][0]["feedback"] == "too aggressive"
    assert len(d["doc_feedback"]) == 1
    assert d["doc_feedback"][0]["operation"] == "classify"
    assert d["doc_feedback"][0]["doc_snapshot"] == {"cat": "billing"}
    assert not d["killed"]

    fb.kill_reason = "bad outputs"
    d = fb.to_dict()
    assert d["killed"]
    assert d["kill_reason"] == "bad outputs"


def test_web_reporter_serves_html():
    """The web server should serve the HTML page and accept feedback POSTs."""
    import json
    import urllib.request
    from http.server import ThreadingHTTPServer

    from docetl.tui.web_reporter import FeedbackStore, _Broadcaster, _make_handler

    t = ProgressTracker()
    t.pipeline_start([("s", "s/op", "map", None)])
    fb = FeedbackStore()
    bc = _Broadcaster(t, fb)
    handler = _make_handler(t, fb, bc)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    import threading
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # GET / should return HTML
        resp = urllib.request.urlopen(f"http://localhost:{port}/")
        html = resp.read().decode()
        assert "DocETL Monitor" in html

        # GET /state should return JSON
        resp = urllib.request.urlopen(f"http://localhost:{port}/state")
        state = json.loads(resp.read())
        assert len(state["ops"]) == 1
        assert state["ops"][0]["status"] == "queued"

        # POST /feedback/pipeline
        req = urllib.request.Request(
            f"http://localhost:{port}/feedback/pipeline",
            data=json.dumps({"text": "needs work"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req)
        assert fb.pipeline_feedback[0]["feedback"] == "needs work"

        # POST /kill
        req = urllib.request.Request(
            f"http://localhost:{port}/kill",
            data=json.dumps({"reason": "bad"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req)
        assert t.kill_requested
        assert fb.kill_reason == "bad"
    finally:
        t.kill_requested = False
        server.shutdown()


def test_log_reporter_emits_progress():
    """The log reporter should emit progress lines for running and done ops."""
    from docetl.tui.log_reporter import _LogReporter

    class FakeConsole:
        def __init__(self):
            self.lines = []
        def log(self, msg):
            self.lines.append(str(msg))

    console = FakeConsole()
    t = ProgressTracker()
    t.pipeline_start([("s", "s/classify", "map", "m")])
    t.op_start("s/classify", "map", "m", total=10)

    reporter = _LogReporter(t, console, interval=60)
    # Emit once with the op running.
    reporter._emit()
    assert any("map:classify" in line for line in console.lines)
    assert any("0/10" in line for line in console.lines)

    # Stream some documents mid-run.
    t.tick(2)
    t.tick_cost(0.10)
    t.add_outputs([{"category": "billing", "priority": "high"}])
    console.lines.clear()
    reporter._emit()
    assert any("50%" in line or "20%" in line for line in console.lines)
    # Document content should appear in the output.
    assert any("category" in line and "billing" in line for line in console.lines)
    assert any("[output]" in line for line in console.lines)

    # A second emit should not repeat the same document.
    console.lines.clear()
    reporter._emit()
    assert not any("billing" in line for line in console.lines)

    t.op_done("s/classify", cost=0.20, prompt_tokens=500, completion_tokens=100,
              outputs=[{"x": i} for i in range(10)])
    t.pipeline_done()
    console.lines.clear()
    reporter._emit()
    assert any("✓ done" in line for line in console.lines)
    assert any("pipeline complete" in line for line in console.lines)


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
