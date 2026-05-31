"""Real screenshots of the progress TUI across operator types (issue #487).

Runs the reduce / split / resolve demo pipelines in ``tests/tui_demo/`` inside
the live Textual app with **real gpt-4.1-nano calls** and captures the operator
coverage added on top of map/filter:

- reduce: the "groups" unit + per-group provenance ("merged N source docs")
- split:  the "chunks" unit + chunk/parent provenance
- resolve: the "comparisons" unit, captured mid-run

Requires ``OPENAI_API_KEY`` and network access to api.openai.com.
Run: ``uv run python tests/tui_operator_screenshots.py``  -> /tmp/docetl_tui/*.png
"""

import asyncio
import os
import time

import cairosvg

from docetl.progress.tracker import ProgressTracker, set_active_tracker
from docetl.runner import DSLRunner
from docetl.tui.app import DocetlTUI, _QuietConsole

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "tui_demo")
OUT = "/tmp/docetl_tui"
os.makedirs(OUT, exist_ok=True)
SIZE = (160, 44)


def _setup(yaml_name: str, max_threads: int) -> tuple[DSLRunner, ProgressTracker, DocetlTUI]:
    runner = DSLRunner.from_yaml(os.path.join(DEMO, yaml_name), max_threads=max_threads)
    runner.config["bypass_cache"] = True
    tracker = ProgressTracker(concurrency=min(runner.max_threads or 1, 64))
    runner.progress_tracker = tracker
    runner._tui_active = True
    set_active_tracker(tracker)
    tracker.pipeline_start(runner.list_pipeline_operations())
    runner.console = _QuietConsole()
    return runner, tracker, DocetlTUI(tracker, runner=runner)


async def _shoot(app: DocetlTUI, pilot, name: str, *, sel_op: int, cursor: int = 0,
                 focus_grid: bool = True) -> str:
    app.sel_op = sel_op
    app.focus_pane = "grid" if focus_grid else "ops"
    app.cursor = cursor
    app.page = 0
    app.render_all()
    # Let Textual paint the updated widgets before capturing (the interval
    # timer + message pump need a tick; a synchronous save grabs a stale frame).
    await pilot.pause(0.2)
    png = os.path.join(OUT, name)
    svg = png[:-4] + ".svg"
    app.save_screenshot(svg)
    cairosvg.svg2png(url=svg, write_to=png, output_width=1600)
    return png


async def _wait_finished(tracker, pilot, limit=360):
    for _ in range(limit):
        if tracker.snapshot().finished:
            return
        await pilot.pause(0.5)


async def shoot_done(yaml_name, png_name, *, sel_op, cursor, max_threads=4):
    """Run a pipeline to completion and snapshot one op + document."""
    _, tracker, app = _setup(yaml_name, max_threads)
    async with app.run_test(size=SIZE) as pilot:
        await _wait_finished(tracker, pilot)
        path = await _shoot(app, pilot, png_name, sel_op=sel_op, cursor=cursor)
    set_active_tracker(None)
    print(png_name, "-> cost $%.4f" % (app.result_cost or 0), "| error:", app.error)
    return path


async def shoot_midrun(yaml_name, png_name, *, target_op, min_done, max_threads=1):
    """Snapshot while ``target_op`` is partway through (to show its live unit)."""
    _, tracker, app = _setup(yaml_name, max_threads)
    async with app.run_test(size=SIZE) as pilot:
        deadline = time.time() + 120
        path = None
        while time.time() < deadline:
            ops = tracker.snapshot().ops
            op = ops[target_op] if target_op < len(ops) else None
            if op and op.status == "running" and op.completed >= min_done:
                path = await _shoot(app, pilot, png_name, sel_op=target_op, cursor=0)
                break
            if tracker.snapshot().finished:
                break
            await pilot.pause(0.1)
        if path is None:  # fell through to completion; capture done-state anyway
            path = await _shoot(app, pilot, png_name, sel_op=target_op, cursor=0)
        await _wait_finished(tracker, pilot)
    set_active_tracker(None)
    print(png_name, "-> cost $%.4f" % (app.result_cost or 0), "| error:", app.error)
    return path


async def main():
    os.chdir(DEMO)
    shots = []
    # reduce: inspect the summarize_by_category op (op index 1), first group.
    shots.append(await shoot_done(
        "reduce_pipeline.yaml", "20_reduce_group.png", sel_op=1, cursor=0))
    # split: inspect the split_report op (index 0) -> chunk provenance.
    shots.append(await shoot_done(
        "split_pipeline.yaml", "21_split_chunks.png", sel_op=0, cursor=1))
    # split: the downstream map over chunks (index 1), showing a chunk summary.
    shots.append(await shoot_done(
        "split_pipeline.yaml", "22_split_map.png", sel_op=1, cursor=0))
    # resolve: capture mid-run so the "comparisons" unit is visible.
    shots.append(await shoot_midrun(
        "resolve_pipeline.yaml", "23_resolve_comparisons.png",
        target_op=0, min_done=2, max_threads=1))
    print("wrote:", *[s for s in shots if s])


if __name__ == "__main__":
    asyncio.run(main())
