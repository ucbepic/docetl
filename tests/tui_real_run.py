"""Run a *real* multi-operator pipeline inside the TUI and screenshot it.

This drives the actual ``DSLRunner`` against the demo pipeline in
``tests/tui_demo/`` with **real** ``gpt-4.1-nano`` LLM calls (the cheapest
OpenAI model). Nothing about the LLM layer is stubbed: the tracker is fed by
genuine operation execution through ``containers.py`` + ``RichLoopBar``. We
capture the live screen mid-run, while inspecting a finished document, and at
completion.

Requires ``OPENAI_API_KEY`` in the environment and network access to
``api.openai.com``.

Run: ``uv run python tests/tui_real_run.py``
Outputs SVGs and PNGs under ``/tmp/docetl_tui/``.
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


def _snapshot(app: DocetlTUI, name: str) -> str:
    app.render_all()
    path = os.path.join(OUT, name)
    app.save_screenshot(path)
    return path


async def main():
    # Run from the demo dir so the pipeline's relative dataset paths resolve.
    os.chdir(DEMO)

    # max_threads=2 keeps the run gradual enough to catch a meaningful mid-run
    # frame; bypass_cache forces real LLM calls so progress actually streams.
    runner = DSLRunner.from_yaml(os.path.join(DEMO, "pipeline.yaml"), max_threads=2)
    runner.config["bypass_cache"] = True

    # Replicate run_with_tui's setup, but launch under run_test so we can pause
    # and screenshot the live run programmatically.
    tracker = ProgressTracker(concurrency=min(runner.max_threads or 1, 64))
    runner.progress_tracker = tracker
    runner._tui_active = True
    set_active_tracker(tracker)
    tracker.pipeline_start(runner.list_pipeline_operations())
    runner.console = _QuietConsole()

    app = DocetlTUI(tracker, runner=runner)
    shots = []
    async with app.run_test(size=SIZE) as pilot:
        # Mid-run snapshot: wait until the first op is partway through so the
        # overview shows a live mix of done / running / queued ops.
        deadline = time.time() + 120
        while time.time() < deadline:
            ops = tracker.snapshot().ops
            first = ops[0] if ops else None
            if first and first.completed >= 5 and not tracker.snapshot().finished:
                break
            await pilot.pause(0.15)
        shots.append(_snapshot(app, "10_real_midrun.svg"))
        print("midrun ops:", tracker.snapshot().to_dict()["ops"])

        # Navigate into the grid of the selected op and inspect a document.
        await pilot.press("tab")
        await pilot.press("right")
        await pilot.press("right")
        await pilot.press("down")
        shots.append(_snapshot(app, "11_real_doc.svg"))

        # Wait for completion (poll up to ~3 min) and capture the final state.
        for _ in range(360):
            if tracker.snapshot().finished:
                break
            await pilot.pause(0.5)
        # Land on a completed op's grid + a real document for the final frame.
        app.focus_pane = "ops"
        app.sel_op = 0
        await pilot.press("enter")
        await pilot.press("right")
        await pilot.press("right")
        shots.append(_snapshot(app, "12_real_complete.svg"))

    set_active_tracker(None)
    snap = tracker.snapshot()
    print("finished:", snap.finished, "cost: $%.4f" % (app.result_cost or 0.0))
    print("error:", app.error)
    for s in shots:
        png = s[:-4] + ".png"
        cairosvg.svg2png(url=s, write_to=png, output_width=1600)
        print("png:", png)


if __name__ == "__main__":
    asyncio.run(main())
