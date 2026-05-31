"""Capture screenshots of the interactive progress view (issue #487).

Two kinds of shot:

- Real runs: drive the actual ``DSLRunner`` over the demo pipelines in
  ``tui_demo/`` with real ``gpt-4.1-nano`` calls (no stubbed model), for
  map/filter, reduce, split, and resolve. A full set costs well under a cent.
- One synthetic large run (40k documents) to show the zoomed-out heatmap grid,
  which would be wasteful to produce with real calls.

Needs ``OPENAI_API_KEY`` and network access for the real runs. Writes SVG and
PNG files to ``/tmp/docetl_tui/``.

Run: ``uv run python tests/tui_screenshots.py``
"""

import asyncio
import os
import time

import cairosvg

from docetl.progress.events import OpState
from docetl.progress.tracker import ProgressTracker, set_active_tracker
from docetl.runner import DSLRunner
from docetl.tui.app import DocetlTUI, _QuietConsole

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "tui_demo")
OUT = "/tmp/docetl_tui"
os.makedirs(OUT, exist_ok=True)
SIZE = (160, 44)


async def _capture(app: DocetlTUI, pilot, name: str, *, sel_op: int, cursor: int = 0):
    """Point the view at one operation/document and save SVG + PNG."""
    app.sel_op = sel_op
    app.focus_pane = "grid"
    app.cursor = cursor
    app.page = 0
    app.render_all()
    await pilot.pause(0.2)  # let Textual paint before grabbing the frame
    svg = os.path.join(OUT, name + ".svg")
    app.save_screenshot(svg)
    cairosvg.svg2png(url=svg, write_to=os.path.join(OUT, name + ".png"), output_width=1600)


# -- real runs ---------------------------------------------------------------
def _setup(yaml_name: str, max_threads: int):
    runner = DSLRunner.from_yaml(os.path.join(DEMO, yaml_name), max_threads=max_threads)
    runner.config["bypass_cache"] = True  # force real calls so progress streams
    tracker = ProgressTracker(concurrency=min(runner.max_threads or 1, 64))
    runner.progress_tracker = tracker
    runner._tui_active = True
    set_active_tracker(tracker)
    tracker.pipeline_start(runner.list_pipeline_operations())
    runner.console = _QuietConsole()
    return tracker, DocetlTUI(tracker, runner=runner)


async def _wait_finished(tracker, pilot, limit=360):
    for _ in range(limit):
        if tracker.snapshot().finished:
            return
        await pilot.pause(0.5)


async def shoot_done(yaml_name, name, *, sel_op, cursor=0, max_threads=4):
    tracker, app = _setup(yaml_name, max_threads)
    async with app.run_test(size=SIZE) as pilot:
        await _wait_finished(tracker, pilot)
        await _capture(app, pilot, name, sel_op=sel_op, cursor=cursor)
    set_active_tracker(None)
    print(name, "cost $%.4f" % (app.result_cost or 0), "error:", app.error)


async def shoot_midrun(yaml_name, name, *, target_op, min_done, max_threads=2):
    """Capture while ``target_op`` is partway through (shows its live unit)."""
    tracker, app = _setup(yaml_name, max_threads)
    async with app.run_test(size=SIZE) as pilot:
        deadline = time.time() + 120
        while time.time() < deadline and not tracker.snapshot().finished:
            ops = tracker.snapshot().ops
            op = ops[target_op] if target_op < len(ops) else None
            if op and op.status == "running" and op.completed >= min_done:
                break
            await pilot.pause(0.1)
        await _capture(app, pilot, name, sel_op=target_op, cursor=0)
        await _wait_finished(tracker, pilot)
    set_active_tracker(None)
    print(name, "cost $%.4f" % (app.result_cost or 0), "error:", app.error)


# -- synthetic large run (heatmap) -------------------------------------------
def _scale_tracker() -> ProgressTracker:
    """A 40k-document run that triggers the zoomed-out heatmap grid."""
    t = ProgressTracker(concurrency=32)
    ops = [
        OpState("extract", "extract/parse_docs", "map", "gpt-4.1-nano"),
        OpState("extract", "extract/classify", "map", "gpt-4.1-nano"),
        OpState("dedupe", "dedupe/resolve", "resolve", "gpt-4.1-nano"),
    ]
    ops[0].status = "done"
    ops[0].total = ops[0].completed = ops[0].out_count = 40000
    ops[0].cost = 1.24
    ops[0].start_t, ops[0].end_t = 0, 612
    ops[1].status = "running"
    ops[1].total, ops[1].completed, ops[1].errors = 40000, 21850, 140
    ops[1].cost = 0.67
    ops[1].start_t = time.time() - 300
    ops[2].total = 40000
    t.state.ops = ops
    t.state._by_name = {o.name: o for o in ops}
    t.state.started = True
    t.state.start_t = time.time() - 920
    t.state.total_cost = sum(o.cost for o in ops)
    return t


async def shoot_heatmap(name):
    app = DocetlTUI(_scale_tracker(), runner=None)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause(0.3)
        await pilot.press("down")  # select the running op
        await _capture(app, pilot, name, sel_op=1, cursor=0)
    print(name)


async def main():
    os.chdir(DEMO)  # so the pipelines' relative dataset paths resolve
    # map -> map -> filter: live mid-run, then completed with a real document.
    await shoot_midrun("pipeline.yaml", "tui-real-midrun", target_op=0, min_done=5)
    await shoot_done("pipeline.yaml", "tui-real-complete", sel_op=0, cursor=19)
    # reduce: the "groups" unit + per-group provenance.
    await shoot_done("reduce_pipeline.yaml", "tui-reduce-groups", sel_op=1, cursor=0)
    # split: the "chunks" unit + chunk/parent provenance.
    await shoot_done("split_pipeline.yaml", "tui-split-chunks", sel_op=0, cursor=1)
    # resolve: the "comparisons" unit, captured mid-run.
    await shoot_midrun("resolve_pipeline.yaml", "tui-resolve-comparisons",
                       target_op=0, min_done=2, max_threads=1)
    # synthetic scale: the heatmap grid for tens of thousands of documents.
    await shoot_heatmap("tui-scale-heatmap")


if __name__ == "__main__":
    asyncio.run(main())
