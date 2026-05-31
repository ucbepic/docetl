"""Generate screenshots of the progress TUI against synthetic run state.

Run: uv run python tests/tui_screenshots.py
Outputs SVGs under /tmp/docetl_tui/.
"""

import asyncio
import os
import random

from docetl.progress.tracker import ProgressTracker
from docetl.progress.events import OpState
from docetl.tui.app import DocetlTUI

OUT = "/tmp/docetl_tui"
os.makedirs(OUT, exist_ok=True)
SIZE = (160, 44)


def _outputs(op_name: str, n: int, with_prompt: bool) -> list[dict]:
    docs = []
    topics = ["billing dispute", "shipping delay", "refund request", "praise",
              "bug report", "feature request", "account access", "cancellation"]
    for i in range(n):
        d = {
            "id": f"doc_{i}",
            "title": f"Customer ticket #{1000 + i}",
            "category": random.choice(topics),
            "sentiment": random.choice(["positive", "neutral", "negative"]),
            "summary": f"The customer wrote in about a {random.choice(topics)} "
                       f"and the agent resolved it after {random.randint(1, 5)} replies.",
            "priority": random.choice(["low", "medium", "high"]),
        }
        if with_prompt:
            d[f"_observability_{op_name.split('/')[-1]}"] = {
                "prompt": f"Analyze the following support ticket and extract the "
                          f"category, sentiment, and a one-line summary.\n\n"
                          f"Ticket: Customer ticket #{1000 + i} ...",
            }
        docs.append(d)
    return docs


def make_midrun_tracker() -> ProgressTracker:
    """A multi-operator run, mid-flight, moderate doc counts (cells/paged grid)."""
    t = ProgressTracker(concurrency=16)
    ops = [
        OpState("analyze", "analyze/classify_tickets", "map", "gpt-4o-mini"),
        OpState("analyze", "analyze/extract_entities", "map", "gpt-4o-mini"),
        OpState("analyze", "analyze/urgent_only", "filter", "gpt-4o-mini"),
        OpState("summarize", "summarize/resolve_customers", "resolve", "gpt-4o-mini"),
        OpState("summarize", "summarize/per_customer_report", "reduce", "gpt-4o"),
    ]
    # op 0 done
    ops[0].status = "done"; ops[0].total = 1200; ops[0].completed = 1200
    ops[0].out_count = 1200; ops[0].cost = 0.84; ops[0].prompt_tokens = 940_000
    ops[0].completion_tokens = 120_000; ops[0].start_t = 0; ops[0].end_t = 47
    ops[0].outputs = _outputs(ops[0].name, 1200, with_prompt=True)
    # op 1 running
    ops[1].status = "running"; ops[1].total = 1200; ops[1].completed = 742
    ops[1].errors = 6; ops[1].cost = 0.51; ops[1].prompt_tokens = 580_000
    ops[1].completion_tokens = 71_000
    import time
    ops[1].start_t = time.time() - 31
    ops[1].outputs = _outputs(ops[1].name, 742, with_prompt=True)
    # rest queued
    ops[2].total = 1200
    t.state.ops = ops
    t.state._by_name = {o.name: o for o in ops}
    t.state.started = True
    t.state.start_t = time.time() - 78
    t.state.total_cost = sum(o.cost for o in ops)
    return t


def make_scale_tracker() -> ProgressTracker:
    """A large run that triggers the heatmap grid (tens of thousands of docs)."""
    t = ProgressTracker(concurrency=32)
    ops = [
        OpState("extract", "extract/parse_docs", "map", "gpt-4o-mini"),
        OpState("extract", "extract/classify", "map", "gpt-4o-mini"),
        OpState("dedupe", "dedupe/resolve", "resolve", "gpt-4o-mini"),
    ]
    ops[0].status = "done"; ops[0].total = 40000; ops[0].completed = 40000
    ops[0].out_count = 40000; ops[0].cost = 12.40; ops[0].prompt_tokens = 28_000_000
    ops[0].completion_tokens = 3_100_000; ops[0].start_t = 0; ops[0].end_t = 612
    ops[1].status = "running"; ops[1].total = 40000; ops[1].completed = 21850
    ops[1].errors = 140; ops[1].cost = 6.7; ops[1].prompt_tokens = 15_000_000
    ops[1].completion_tokens = 1_700_000
    import time
    ops[1].start_t = time.time() - 300
    ops[2].total = 40000
    t.state.ops = ops
    t.state._by_name = {o.name: o for o in ops}
    t.state.started = True
    t.state.start_t = time.time() - 920
    t.state.total_cost = sum(o.cost for o in ops)
    return t


async def shoot(tracker, name, keys):
    app = DocetlTUI(tracker, runner=None)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause(0.3)
        for k in keys:
            await pilot.press(k)
            await pilot.pause()
        app.render_all()
        await pilot.pause(0.05)
        app.save_screenshot(os.path.join(OUT, name))
    print("wrote", os.path.join(OUT, name))


async def main():
    random.seed(7)
    # 1. mid-run, operations pane focused (overview)
    await shoot(make_midrun_tracker(), "01_overview.svg", [])
    # 2. mid-run, select the running op + enter grid
    await shoot(make_midrun_tracker(), "02_grid_running.svg",
                ["down", "tab"])
    # 3. mid-run, inspect a completed document (op 0 -> grid -> move cursor)
    await shoot(make_midrun_tracker(), "03_doc_detail.svg",
                ["tab", "right", "right", "right", "down", "down"])
    # 4. scale: heatmap grid on the running op
    await shoot(make_scale_tracker(), "04_heatmap.svg",
                ["down", "tab"])


if __name__ == "__main__":
    asyncio.run(main())
