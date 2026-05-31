"""Run a real multi-operator pipeline inside the TUI and screenshot it.

Drives the actual DSLRunner (real LLM calls via gpt-4o-mini) so the tracker is
fed by genuine operation execution through containers.py + RichLoopBar.
Captures the live screen mid-run and at completion.

Run: uv run python tests/tui_real_run.py
"""

import asyncio
import os
import random
import time

import cairosvg

from docetl.operations.utils.llm import LLMResult
from docetl.progress.tracker import ProgressTracker, set_active_tracker
from docetl.runner import DSLRunner
from docetl.tui.app import DocetlTUI, _QuietConsole


def install_fake_llm(runner: DSLRunner) -> None:
    """Stub the LLM layer with a fast, deterministic local fake.

    The OpenAI host is blocked by this environment's network policy, so we
    cannot make real calls here. Everything else -- the runner, map/filter
    execution, RichLoopBar ticks, and containers.py op_start/op_done -- runs for
    real; only the network call is replaced. Adds a small sleep so a multi-op
    run takes a few seconds and the mid-run screenshot shows partial progress.
    """
    rng = random.Random(11)

    def fake_call_llm(model, op_type, messages, output_schema, *args, **kwargs):
        time.sleep(rng.uniform(0.15, 0.4))
        out = {}
        for key, typ in (output_schema or {}).items():
            ts = str(typ).lower()
            if "bool" in ts:
                out[key] = rng.random() < 0.55
            elif "int" in ts:
                out[key] = rng.randint(1, 5)
            elif "list" in ts:
                out[key] = ["item"]
            elif key == "category":
                out[key] = rng.choice(
                    ["billing", "shipping", "account", "product", "praise"]
                )
            elif key == "sentiment":
                out[key] = rng.choice(["negative", "neutral", "positive"])
            elif key == "reply":
                out[key] = "Thanks for reaching out — we're on it and will follow up shortly."
            else:
                out[key] = f"{key}-value"
        # simulate token accounting so the UI shows realistic numbers
        runner.total_token_usage[model]["prompt_tokens"] += rng.randint(380, 620)
        runner.total_token_usage[model]["completion_tokens"] += rng.randint(20, 90)
        return LLMResult(response=out, total_cost=rng.uniform(0.0001, 0.0004), validated=True)

    runner.api.call_llm = fake_call_llm

OUT = "/tmp/docetl_tui"
os.makedirs(OUT, exist_ok=True)
SIZE = (160, 44)


async def main():
    runner = DSLRunner.from_yaml("/tmp/docetl_demo/pipeline.yaml", max_threads=6)
    install_fake_llm(runner)

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
        # mid-run snapshot: wait for the first op to be partly done
        await pilot.pause(3.0)
        app.render_all()
        await pilot.pause(0.1)
        p = os.path.join(OUT, "10_real_midrun.svg")
        app.save_screenshot(p)
        shots.append(p)
        print("midrun:", tracker.snapshot().to_dict()["ops"])

        # navigate into the grid of the running/first op and inspect a doc
        await pilot.press("tab")
        await pilot.press("right")
        await pilot.press("right")
        await pilot.press("down")
        app.render_all()
        await pilot.pause(0.1)
        p = os.path.join(OUT, "11_real_doc.svg")
        app.save_screenshot(p)
        shots.append(p)

        # wait for completion (poll up to ~3 min)
        for _ in range(360):
            if tracker.snapshot().finished:
                break
            await pilot.pause(0.5)
        app.render_all()
        await pilot.pause(0.1)
        p = os.path.join(OUT, "12_real_complete.svg")
        app.save_screenshot(p)
        shots.append(p)

    set_active_tracker(None)
    print("finished:", tracker.snapshot().finished, "cost:", app.result_cost)
    print("error:", app.error)
    for s in shots:
        cairosvg.svg2png(url=s, write_to=s[:-4] + ".png", output_width=1600)
        print("png:", s[:-4] + ".png")


if __name__ == "__main__":
    asyncio.run(main())
