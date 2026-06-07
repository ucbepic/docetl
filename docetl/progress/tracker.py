"""Thread-safe progress tracker.

Operations run inside ``ThreadPoolExecutor``s, but the pull-based DAG executes
*one operation at a time* (documents within an operation are parallel; the
operations themselves are sequential). That property lets the tracker keep a
single "current operation", so a generic hook in :class:`RichLoopBar` can tick
the current op without every operation needing bespoke instrumentation.
"""

from __future__ import annotations

import threading
import time
import uuid

from .events import OpState, RunState


class PipelineKilled(Exception):
    """Raised when the user kills the pipeline via the feedback UI."""


class ProgressTracker:
    """Collects structured progress from a running pipeline.

    All mutating methods are guarded by a lock. UIs should call
    :meth:`snapshot` (cheap) on a timer and read the returned :class:`RunState`.
    """

    def __init__(self, concurrency: int = 1) -> None:
        self._lock = threading.RLock()
        self.state = RunState(run_id=uuid.uuid4().hex[:8], concurrency=concurrency)
        self._current: OpState | None = None
        # Subscribers are called (outside the lock) after each mutation so the
        # TUI can request a redraw promptly rather than only polling.
        self._subscribers: list = []
        self.kill_requested = False

    # -- subscription ----------------------------------------------------
    def subscribe(self, callback) -> None:
        self._subscribers.append(callback)

    def _notify(self) -> None:
        for cb in list(self._subscribers):
            try:
                cb()
            except Exception:
                pass

    # -- lifecycle -------------------------------------------------------
    def pipeline_start(self, ops: list[tuple[str, str, str, str | None]]) -> None:
        """Register all operations up front, in pipeline order.

        ``ops`` is a list of ``(step, name, op_type, model)`` tuples.
        """
        with self._lock:
            self.state.ops.clear()
            self.state._by_name.clear()
            for step, name, op_type, model in ops:
                self.state.register(
                    OpState(step=step, name=name, op_type=op_type, model=model)
                )
            self.state.started = True
            self.state.start_t = time.time()
        self._notify()

    def op_start(
        self, name: str, op_type: str, model: str | None, total: int | None
    ) -> None:
        with self._lock:
            op = self.state.get(name)
            if op is None:
                # Operation not pre-registered (e.g. optimizer-injected); add it.
                op = OpState(step=name.split("/")[0], name=name, op_type=op_type, model=model)
                self.state.register(op)
            op.op_type = op_type
            op.model = model
            op.total = total
            op.completed = 0
            op.errors = 0
            op.status = "running"
            op.start_t = time.time()
            self._current = op
        self._notify()

    def set_phase(self, total: int | None) -> None:
        """Reset the current op's progress to a fresh phase of ``total`` units.

        ``RichLoopBar`` calls this when a progress bar starts so the denominator
        matches what is actually being ticked — documents for map/filter, groups
        for reduce, comparisons for resolve/equijoin — rather than the raw
        input-doc count guessed in ``containers.py``. Multi-phase ops (e.g.
        resolve: embed, then compare) call it once per phase; the bar reflects
        the current phase, which is the more useful live signal.
        """
        with self._lock:
            op = self._current
            if op is None:
                return
            op.total = total
            op.completed = 0
            op.errors = 0
        self._notify()

    def tick(self, n: int = 1) -> None:
        """Advance the current operation by ``n`` completed documents."""
        with self._lock:
            op = self._current
            if op is None:
                return
            op.completed += n
            if op.total is not None and op.completed > op.total:
                op.completed = op.total
        self._notify()

    def tick_cost(self, delta: float) -> None:
        """Increment the current operation's cost so the UI can show it live."""
        if delta <= 0:
            return
        with self._lock:
            op = self._current
            if op is None:
                return
            op.cost += delta
            self.state.total_cost = sum(o.cost for o in self.state.ops)
        self._notify()

    def add_outputs(self, items: list[dict]) -> None:
        """Append finished documents to the current op's output sample as they
        complete, so the detail pane can show them mid-run instead of only once
        the whole operation finishes. Capped at ``sample_cap``."""
        if not items:
            return
        with self._lock:
            op = self._current
            if op is None:
                return
            room = op.sample_cap - len(op.outputs)
            if room > 0:
                op.outputs.extend(items[:room])
        self._notify()

    def doc_error(self, n: int = 1) -> None:
        with self._lock:
            if self._current is not None:
                self._current.errors += n
        self._notify()

    def op_done(
        self,
        name: str,
        cost: float,
        prompt_tokens: int,
        completion_tokens: int,
        outputs: list[dict] | None = None,
    ) -> None:
        with self._lock:
            op = self.state.get(name)
            if op is None:
                return
            op.status = "done"
            op.end_t = time.time()
            op.cost = cost
            op.prompt_tokens = prompt_tokens
            op.completion_tokens = completion_tokens
            if outputs is not None:
                op.out_count = len(outputs)
                op.outputs = outputs[: op.sample_cap]
                if op.total is not None:
                    op.completed = op.total
            if self._current is op:
                self._current = None
            self.state.total_cost = sum(o.cost for o in self.state.ops)
        self._notify()

    def pipeline_done(self) -> None:
        with self._lock:
            self.state.finished = True
            self.state.end_t = time.time()
        self._notify()

    # -- reads -----------------------------------------------------------
    def snapshot(self) -> RunState:
        # The TUI reads attributes directly; returning the live object is fine
        # because reads of simple counters are atomic enough for display and we
        # never tear down the structure mid-read.
        return self.state


# Module-global "active" tracker. The runner registers a tracker here while a
# TUI run is in progress so that the generic ``RichLoopBar`` hook can tick the
# current operation without every operation passing the tracker explicitly.
# Only set during interactive runs, so normal runs incur zero overhead.
_ACTIVE_TRACKER: ProgressTracker | None = None


def set_active_tracker(tracker: ProgressTracker | None) -> None:
    global _ACTIVE_TRACKER
    _ACTIVE_TRACKER = tracker


def active_tracker() -> ProgressTracker | None:
    return _ACTIVE_TRACKER
