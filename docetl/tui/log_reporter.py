"""Log-based progress reporter for non-interactive environments.

When a progress UI is requested but stdout is not a TTY (e.g. running from
CI or a subprocess), the full Textual TUI cannot run.  This module
provides a lightweight alternative that prints structured progress lines **and
document samples** to the console, so a human or AI agent can see what is being
generated in real time and decide whether to intervene (Ctrl-C, edit, etc.).
"""

from __future__ import annotations

import json
import threading
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


_TRUNC = 200


def _trunc(s: str) -> str:
    return s if len(s) <= _TRUNC else s[:_TRUNC] + " …"


def _format_doc(doc: dict) -> str:
    """Render a document dict as a compact, human-readable block."""
    lines = []
    for k, v in doc.items():
        if k.startswith("_"):
            continue
        if isinstance(v, str):
            lines.append(f"  {k}: {_trunc(v)}")
        else:
            lines.append(f"  {k}: {_trunc(json.dumps(v, ensure_ascii=False, default=str))}")
    return "\n".join(lines)


class _LogReporter:
    """Periodically prints progress and document samples from tracker state."""

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
        # Track how many outputs we've already printed per op, so we only
        # print *new* documents each tick.
        self._printed_outputs: dict[str, int] = {}

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

    def _print_new_outputs(self, op) -> None:
        """Print any document samples that arrived since our last check."""
        already = self._printed_outputs.get(op.name, 0)
        docs = op.outputs[already:]
        if not docs:
            return
        label = f"{op.op_type}:{op.name.split('/')[-1]}"
        # Show up to 3 new docs per tick to avoid flooding.
        for doc in docs[:3]:
            self._console.log(f"[output] {label}  document #{already + 1}:\n{_format_doc(doc)}")
            already += 1
        remaining = len(docs) - 3
        if remaining > 0:
            already += remaining
            self._console.log(f"[output] {label}  … and {remaining} more document(s)")
        self._printed_outputs[op.name] = already

    def _emit(self) -> None:
        state = self._tracker.snapshot()
        for op in state.ops:
            label = f"{op.op_type}:{op.name.split('/')[-1]}"

            if op.status == "done" and op.name not in self._reported_done:
                self._reported_done.add(op.name)
                # Print any remaining unprinted outputs before the summary.
                self._print_new_outputs(op)
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
                # Print new document samples first.
                self._print_new_outputs(op)

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

    Used when a progress UI is requested but a full TUI cannot run
    (e.g. non-TTY environment like CI).
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
