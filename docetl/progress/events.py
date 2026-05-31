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
DocStatus = Literal["queued", "running", "done", "error"]


@dataclass
class OpState:
    """Live state for a single operation in the pipeline."""

    step: str
    name: str
    op_type: str
    model: str | None = None

    status: OpStatus = "queued"
    total: int | None = None  # total input docs (None until known)
    completed: int = 0
    errors: int = 0
    out_count: int | None = None  # output docs once finished

    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

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
    def label(self) -> str:
        return f"{self.op_type}:{self.name}"

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cell_status(self, index: int, running_band: int) -> DocStatus:
        """Synthesize a per-document status for the dot grid.

        We have reliable *counts* for every operation (completed / errors /
        total). We render the first ``errors`` cells as errors, the next block
        as done, a small band as in-flight, and the rest as queued. This gives
        an accurate fill ratio and a lively "working edge" without requiring
        every operation to emit per-document identity.
        """
        if self.total is None:
            return "queued"
        if index < self.errors:
            return "error"
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
                    "total": op.total,
                    "completed": op.completed,
                    "errors": op.errors,
                    "out_count": op.out_count,
                    "cost": op.cost,
                    "tokens": op.tokens,
                    "elapsed": op.elapsed,
                }
                for op in self.ops
            ],
        }
