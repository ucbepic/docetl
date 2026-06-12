"""State model for pipeline progress.

These dataclasses are the in-memory model the UI reads. They are deliberately
plain and JSON-serializable (via :meth:`RunState.to_dict`) so the same model
can later be streamed to the web UI over the existing websocket.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

OpStatus = Literal["queued", "running", "done", "error"]
DocStatus = OpStatus


@dataclass
class OpState:
    """Live state for a single operation in the pipeline."""

    step: str
    name: str
    op_type: str
    model: str | None = None

    status: OpStatus = "queued"
    total: int | None = None  # total input docs (None until known)
    phase: str | None = None  # live sub-phase label (e.g. cascade proxy/oracle)
    completed: int = 0
    errors: int = 0
    out_count: int | None = None  # output docs once finished

    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    cascade_info: dict | None = None
    grid_complete: bool = False  # when True, grid shows all items as done
    grid_total: int | None = None  # overrides total for grid when set

    start_t: float | None = None
    end_t: float | None = None

    # A capped sample of output documents, kept for the detail pane. We never
    # hold the full output set in memory for large runs.
    outputs: list[dict] = field(default_factory=list)
    sample_cap: int = 2000

    @property
    def elapsed(self) -> float:
        if self.start_t is None:
            return 0.0
        return (self.end_t or time.time()) - self.start_t

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def grid_count(self) -> int:
        """Number of cells the document grid should render.

        While the op runs we show one cell per *work unit* — whatever
        ``RichLoopBar`` is ticking (documents for map, groups for reduce,
        comparisons for resolve/equijoin) — so the fill animates against the
        real denominator. Once finished we switch to one cell per *output
        document* so every cell is an inspectable result, even when the output
        count differs from the work-unit count (e.g. split fans 1 doc into many
        chunks; resolve collapses many comparisons into a few clusters).
        """
        if self.status == "done" and self.out_count is not None:
            return self.out_count
        if self.grid_total is not None:
            return self.grid_total
        return self.total or 0

    def cell_cascade_role(self, index: int) -> str | None:
        """Return 'proxy' or 'oracle' for the given cell, or None."""
        if not self.cascade_info:
            return None
        escalated = self.cascade_info.get("item_escalated")
        if not escalated:
            return None
        input_idx = index
        if self.status == "done" and self.out_count is not None:
            kept = self.cascade_info.get("kept_input_indices")
            if kept:
                if index < len(kept):
                    input_idx = kept[index]
                else:
                    return None
            # No kept_input_indices → output index == input index (map cascade)
        if input_idx < len(escalated):
            return "oracle" if escalated[input_idx] else "proxy"
        return None

    def cell_status(self, index: int, running_band: int) -> DocStatus:
        """Synthesize a per-document status for the dot grid.

        We have reliable *counts* for every operation (completed / errors /
        total). We render the first ``errors`` cells as errors, the next block
        as done, a small band as in-flight, and the rest as queued. This gives
        an accurate fill ratio and a lively "working edge" without requiring
        every operation to emit per-document identity.
        """
        if index < self.errors:
            return "error"
        if self.status == "done":
            return "done"
        if self.grid_complete:
            return "done"
        if self.total is None:
            return "queued"
        if index < self.completed:
            return "done"
        if self.status == "running" and index < self.completed + running_band:
            return "running"
        return "queued"


@dataclass
class RunState:
    """Top-level state for one pipeline run."""

    run_id: str
    ops: list[OpState] = field(default_factory=list)
    started: bool = False
    finished: bool = False
    start_t: float | None = None
    end_t: float | None = None
    total_cost: float = 0.0
    concurrency: int = 1  # max worker threads, used for the running band

    def __post_init__(self) -> None:
        self._by_name: dict[str, OpState] = {op.name: op for op in self.ops}

    def get(self, name: str) -> OpState | None:
        return self._by_name.get(name)

    def register(self, op: OpState) -> None:
        self.ops.append(op)
        self._by_name[op.name] = op

    @property
    def elapsed(self) -> float:
        if self.start_t is None:
            return 0.0
        return (self.end_t or time.time()) - self.start_t

    @property
    def done_ops(self) -> int:
        return sum(1 for op in self.ops if op.status == "done")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "finished": self.finished,
            "elapsed": self.elapsed,
            "total_cost": self.total_cost,
            "ops": [
                {
                    "step": op.step,
                    "name": op.name,
                    "type": op.op_type,
                    "model": op.model,
                    "status": op.status,
                    "phase": op.phase,
                    "total": op.total,
                    "completed": op.completed,
                    "errors": op.errors,
                    "out_count": op.out_count,
                    "cost": op.cost,
                    "tokens": op.tokens,
                    "elapsed": op.elapsed,
                    "cascade_info": op.cascade_info,
                }
                for op in self.ops
            ],
        }
