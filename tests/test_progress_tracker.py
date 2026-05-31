"""Unit tests for the progress telemetry foundation (issue #487)."""

from docetl.progress.events import OpState, RunState
from docetl.progress.tracker import (
    ProgressTracker,
    active_tracker,
    set_active_tracker,
)


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

    outputs = [{"x": i} for i in range(10)]
    t.op_done(
        "s1/classify",
        cost=0.5,
        prompt_tokens=1000,
        completion_tokens=200,
        outputs=outputs,
    )
    assert op.status == "done"
    assert op.completed == 10  # forced to total on completion
    assert op.out_count == 10 and op.tokens == 1200 and op.cost == 0.5
    assert state.total_cost == 0.5
    assert state.done_ops == 1

    t.pipeline_done()
    assert t.snapshot().finished


def test_tick_caps_at_total():
    t = ProgressTracker()
    t.op_start("op", "map", None, total=3)
    for _ in range(10):
        t.tick()
    assert t.snapshot().get("op").completed == 3


def test_cell_status_synthesis():
    op = OpState("s", "s/op", "map")
    op.total = 100
    op.completed = 40
    op.errors = 3
    op.status = "running"
    # first `errors` cells are errors
    assert op.cell_status(0, running_band=5) == "error"
    assert op.cell_status(2, running_band=5) == "error"
    # then done up to `completed`
    assert op.cell_status(3, running_band=5) == "done"
    assert op.cell_status(39, running_band=5) == "done"
    # a running band just past the completed edge
    assert op.cell_status(41, running_band=5) == "running"
    # then queued
    assert op.cell_status(80, running_band=5) == "queued"


def test_active_tracker_hook():
    t = ProgressTracker()
    set_active_tracker(t)
    try:
        assert active_tracker() is t
        t.op_start("op", "map", None, total=5)
        # simulate RichLoopBar.update path
        from docetl.operations.utils.progress import RichLoopBar

        # RichLoopBar.update calls active_tracker().tick()
        bar = RichLoopBar.__new__(RichLoopBar)
        bar.tqdm = None
        bar.update(2)
        assert t.snapshot().get("op").completed == 2
    finally:
        set_active_tracker(None)
    assert active_tracker() is None


def test_runstate_to_dict():
    s = RunState(run_id="abc")
    s.register(OpState("s", "s/op", "map", "m"))
    d = s.to_dict()
    assert d["run_id"] == "abc"
    assert d["ops"][0]["name"] == "s/op"
    assert d["ops"][0]["status"] == "queued"
