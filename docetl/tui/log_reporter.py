"""Log-based progress reporter for non-interactive environments.

When ``interactive_ui: True`` but stdout is not a TTY (e.g. running from an AI
agent, CI, or a subprocess), the full Textual TUI cannot run.  This module
provides a lightweight alternative that prints structured progress lines to the
console so the caller can still observe operation status, document counts, and
costs as the pipeline executes.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from docetl.progress.tracker import ProgressTracker, set_active_tracker

if TYPE_CHECKING:
    from docetl.runner import DSLRunner


def _fmt_cost(c: float) -> str:
    if c <= 0:
        return "$0"
    if c < 0.01:
        return "<$0.01"
    return f"${c:.2f}"


def _fmt_dur(secs: float) -> str:
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


class _LogReporter:
    """Periodically prints progress lines based on tracker state changes."""

    def __init__(self, tracker: ProgressTracker, console, interval: float = 3.0):
        self._tracker = tracker
        self._console = console
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_op: str | None = None
        self._last_completed: int = -1
        self._last_cost: float = -1.0
        self._reported_done: set[str] = set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._emit()
            self._stop.wait(self._interval)
        self._emit()

    def _emit(self) -> None:
        state = self._tracker.snapshot()
        for op in state.ops:
            label = f"{op.op_type}:{op.name.split('/')[-1]}"

            if op.status == "done" and op.name not in self._reported_done:
                self._reported_done.add(op.name)
                parts = [f"[progress] {label}  ✓ done"]
                if op.total is not None:
                    parts.append(f"{op.completed}/{op.total}")
                if op.out_count is not None and op.out_count != op.total:
                    parts.append(f"→ {op.out_count} output")
                parts.append(_fmt_cost(op.cost))
                parts.append(_fmt_dur(op.elapsed))
                self._console.log("  ".join(parts))
                continue

            if op.status == "running":
                changed = (
                    op.name != self._last_op
                    or op.completed != self._last_completed
                    or abs(op.cost - self._last_cost) > 0.005
                )
                if not changed:
                    continue
                self._last_op = op.name
                self._last_completed = op.completed
                self._last_cost = op.cost

                parts = [f"[progress] {label}"]
                if op.total:
                    pct = int(100 * op.completed / op.total)
                    parts.append(f"{op.completed}/{op.total} ({pct}%)")
                elif op.completed:
                    parts.append(f"{op.completed} done")
                parts.append(_fmt_cost(op.cost))
                if op.elapsed >= 1:
                    parts.append(_fmt_dur(op.elapsed))
                self._console.log("  ".join(parts))

        if state.finished:
            self._console.log(
                f"[progress] pipeline complete  {_fmt_cost(state.total_cost)}  "
                f"{_fmt_dur(state.elapsed)}"
            )


def run_with_log_reporter(runner: "DSLRunner") -> float:
    """Execute the pipeline with log-based progress output.

    Used when ``interactive_ui`` is requested but a full TUI cannot run
    (e.g. non-TTY environment like an AI agent or CI).
    """
    import threading
    from tqdm import tqdm as _tqdm

    _tqdm.set_lock(threading.RLock())

    tracker = ProgressTracker(concurrency=min(runner.max_threads or 1, 64))
    runner.progress_tracker = tracker
    runner._tui_active = True
    set_active_tracker(tracker)

    tracker.pipeline_start(runner.list_pipeline_operations())

    ops = runner.list_pipeline_operations()
    runner.console.log(
        f"[progress] pipeline starting  {len(ops)} operation(s)"
    )

    reporter = _LogReporter(tracker, runner.console)
    reporter.start()

    try:
        cost = runner.load_run_save()
    finally:
        reporter.stop()
        set_active_tracker(None)
        runner.progress_tracker = None
        runner._tui_active = False

    return cost
