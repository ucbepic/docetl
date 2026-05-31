"""Structured progress telemetry for DocETL pipeline runs.

This package provides a thread-safe :class:`ProgressTracker` that pipeline
operations emit into as they execute. The tracker maintains a compact
:class:`RunState` (per-operation counters, cost, tokens, and a sample of
outputs) that UIs read to render progress -- currently the interactive
terminal TUI (:mod:`docetl.tui`), and in the future the web UI.

The design intentionally decouples *progress signal* from *console text*: the
old path only had aggregate ``tqdm`` counts, which cannot drive a per-document
view. See ``docs/design/progress-visualization.md``.
"""

from .events import DocStatus, OpState, OpStatus, RunState
from .tracker import ProgressTracker

__all__ = [
    "ProgressTracker",
    "RunState",
    "OpState",
    "OpStatus",
    "DocStatus",
]
