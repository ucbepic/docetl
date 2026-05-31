"""Textual app rendering live pipeline progress as a three-pane dashboard.

Layout (mirrors Claude Code's ``/workflows`` screen):

    ┌ Operations ─┐┌ <selected op> ── grid of dots ─┐┌ Document detail ─┐
    │ steps → ops ││  ● ● ● ◐ · · · · · · · · ·      ││ input / output   │
    │ + counters  ││  paged / heatmap at scale       ││ prompt / status  │
    └─────────────┘└─────────────────────────────────┘└──────────────────┘

Navigation: ↑/↓ move within a pane, Tab switches panes, ←/→ and PgUp/PgDn move
the grid cursor, q quits. The detail pane reflects the document under the grid
cursor.
"""

from __future__ import annotations

import io
import json
import math
import threading
from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from docetl.progress.events import OpState, RunState
from docetl.progress.tracker import ProgressTracker, set_active_tracker
from docetl.tui.profiles import get_profile

if TYPE_CHECKING:
    from docetl.runner import DSLRunner

# Status -> (glyph, rich color)
_DOT = "●"
_STATUS_STYLE = {
    "done": "green",
    "running": "yellow",
    "error": "red",
    "queued": "grey42",
}
_OP_GLYPH = {"done": "✓", "running": "◐", "error": "✗", "queued": "○"}


class DocetlTUI(App):
    """Full-screen progress dashboard for one pipeline run."""

    CSS = """
    Screen { background: $surface; }
    #ops { width: 36; border: round $primary; padding: 0 1; }
    #middle { border: round $primary; padding: 0 1; }
    #detail { width: 52; border: round $accent; padding: 0 1; }
    #grid_title { height: 1; color: $text-muted; }
    #grid { height: 1fr; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, tracker: ProgressTracker, runner: "DSLRunner | None" = None):
        super().__init__()
        self.tracker = tracker
        self.runner = runner
        self.focus_pane = "ops"  # "ops" | "grid"
        self.sel_op = 0
        self.cursor = 0  # cell index within the current grid page
        self.page = 0  # current grid page (0-based)
        # Geometry computed during grid render, read by the key handler.
        self._cols = 1
        self._capacity = 1
        self._mode = "cells"  # "cells" | "paged" | "heatmap"
        self._page_count = 1
        self._page_cells = 0
        self.result_cost: float | None = None
        self.error: BaseException | None = None
        self._worker: threading.Thread | None = None
        # Cache of serialized doc bodies, keyed by (op name, index). Outputs are
        # immutable once captured at op_done, so the detail pane re-renders the
        # selected cell ~7x/s without re-running json.dumps each frame.
        self._doc_json: dict[tuple[str, int], str] = {}

    # -- composition -----------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static(id="ops")
            with Vertical(id="middle"):
                yield Static(id="grid_title")
                yield Static(id="grid")
            yield Static(id="detail")

    def on_mount(self) -> None:
        self.title = "DocETL"
        self.set_interval(0.15, self.render_all)
        if self.runner is not None:
            self._worker = threading.Thread(target=self._run_pipeline, daemon=True)
            self._worker.start()
        self.render_all()

    def on_resize(self, event) -> None:
        # Re-render once the layout has assigned real widget sizes so the grid
        # uses the full middle pane (the first on_mount render runs before layout).
        self.render_all()

    def _run_pipeline(self) -> None:
        try:
            self.result_cost = self.runner.load_run_save()
        except BaseException as exc:  # noqa: BLE001 - surface any failure in the UI
            self.error = exc
            try:
                self.tracker.pipeline_done()
            except Exception:
                pass

    # -- helpers ---------------------------------------------------------
    def _ops(self) -> list[OpState]:
        return self.tracker.snapshot().ops

    def _selected_op(self) -> OpState | None:
        ops = self._ops()
        if not ops:
            return None
        self.sel_op = max(0, min(self.sel_op, len(ops) - 1))
        return ops[self.sel_op]

    # -- key handling ----------------------------------------------------
    def on_key(self, event) -> None:
        key = event.key
        if key in ("q", "escape"):
            self.exit()
            return
        if key == "tab":
            self.focus_pane = "grid" if self.focus_pane == "ops" else "ops"
        elif self.focus_pane == "ops":
            if key == "down":
                self.sel_op += 1
                self.cursor = self.page = 0
            elif key == "up":
                self.sel_op -= 1
                self.cursor = self.page = 0
            elif key in ("right", "enter"):
                self.focus_pane = "grid"
        else:  # grid pane
            if key == "right":
                self.cursor += 1
            elif key == "left":
                if self.cursor == 0:
                    self.focus_pane = "ops"
                else:
                    self.cursor -= 1
            elif key == "down":
                self.cursor += self._cols
            elif key == "up":
                self.cursor = max(0, self.cursor - self._cols)
            elif key in ("pagedown", "space"):
                if self.page < self._page_count - 1:
                    self.page += 1
                    self.cursor = 0
            elif key == "pageup":
                if self.page > 0:
                    self.page -= 1
                    self.cursor = 0
            self.cursor = max(0, min(self.cursor, max(0, self._page_cells - 1)))
        event.stop()
        self.render_all()

    # -- rendering -------------------------------------------------------
    def render_all(self) -> None:
        state = self.tracker.snapshot()
        try:
            self.query_one("#ops", Static).update(self._render_ops(state))
            self.query_one("#grid_title", Static).update(self._render_grid_title(state))
            self.query_one("#grid", Static).update(self._render_grid(state))
            self.query_one("#detail", Static).update(self._render_detail(state))
        except Exception:
            # Widgets may not be mounted on the very first tick.
            pass

    def _render_ops(self, state: RunState) -> Panel:
        head = Text()
        head.append("DocETL pipeline\n", style="bold")
        head.append(
            f"{state.done_ops}/{len(state.ops)} ops · ", style="grey70"
        )
        head.append(f"${state.total_cost:.2f} · ", style="green")
        head.append(f"{_fmt_dur(state.elapsed)}\n", style="grey70")
        if self.error is not None:
            head.append("✗ failed\n", style="bold red")
        elif state.finished:
            head.append("✓ complete\n", style="bold green")

        body = Text()
        last_step = None
        for i, op in enumerate(state.ops):
            if op.step != last_step:
                body.append(f"\n{op.step}\n", style="bold cyan")
                last_step = op.step
            selected = i == self.sel_op
            glyph = _OP_GLYPH[op.status]
            gstyle = _STATUS_STYLE[op.status]
            line = Text()
            line.append(f" {glyph} ", style=gstyle)
            label = f"{op.op_type}:{op.name.split('/')[-1]}"
            line.append(_trunc(label, 22), style="bold" if selected else "")
            # progress / counts
            if op.total:
                pct = int(100 * op.completed / op.total)
                line.append(f"  {op.completed}/{op.total} ", style="grey70")
                if op.status == "running":
                    line.append(f"{pct}%", style="yellow")
            if op.errors:
                line.append(f"  !{op.errors}", style="red")
            if selected:
                line.stylize("reverse")
            body.append_text(line)
            body.append("\n")
            # second line: cost / tokens / time for finished or running ops
            if op.status in ("done", "running") and (op.cost or op.tokens):
                sub = Text("     ")
                sub.append(f"${op.cost:.3f}", style="green")
                if op.tokens:
                    sub.append(f" · {_fmt_k(op.tokens)} tok", style="grey50")
                sub.append(f" · {_fmt_dur(op.elapsed)}", style="grey50")
                body.append_text(sub)
                body.append("\n")

        title = "Operations" + (" ◀" if self.focus_pane == "ops" else "")
        return Panel(Group(head, body), title=title, border_style="cyan")

    def _render_grid_title(self, state: RunState) -> Text:
        op = self._selected_op()
        if op is None:
            return Text("")
        prof = get_profile(op.op_type)
        t = Text()
        t.append(f"{op.op_type}:{op.name.split('/')[-1]}", style="bold")
        if op.status == "done" and op.out_count is not None:
            t.append(f"  {op.out_count:,} {prof.doc_unit}", style="grey70")
        elif op.total:
            t.append(f"  {op.completed:,}/{op.total:,} {prof.unit}", style="grey70")
        t.append(f"  ·  {op.status}", style=_STATUS_STYLE[op.status])
        if self._mode == "heatmap":
            t.append("  ·  heatmap (each cell = bucket)", style="magenta")
        elif self._mode == "paged":
            t.append(
                f"  ·  page {self.page + 1}/{self._page_count} [PgDn]", style="grey50"
            )
        return t

    def _render_grid(self, state: RunState) -> Text:
        op = self._selected_op()
        if op is None or not op.grid_count:
            self._page_cells = 0
            return Text("(no documents yet)", style="grey42")

        size = self.query_one("#grid", Static).size
        width = max(10, size.width)
        height = max(4, size.height)
        cols = max(1, width // 2)
        capacity = cols * height
        total = op.grid_count
        self._cols = cols
        self._capacity = capacity

        running_band = state.concurrency if op.status == "running" else 0

        # Choose a display mode that stays legible at scale.
        if total <= capacity:
            self._mode, self._page_count = "cells", 1
        elif total <= capacity * 4:
            self._mode = "paged"
            self._page_count = math.ceil(total / capacity)
        else:
            self._mode, self._page_count = "heatmap", 1

        self.page = max(0, min(self.page, self._page_count - 1))
        out = Text()

        if self._mode == "heatmap":
            bucket = math.ceil(total / capacity)
            self._page_cells = capacity
            for cell in range(capacity):
                lo = cell * bucket
                if lo >= total:
                    break
                hi = min(total, lo + bucket)
                n = hi - lo
                done = max(0, min(n, op.completed - lo))
                err = max(0, min(n, op.errors - lo))
                frac = done / n if n else 0
                style = _heat_style(frac, err > 0)
                glyph = _DOT if frac > 0 else "·"
                if cell == self.cursor and self.focus_pane == "grid":
                    out.append(glyph, style=style + " reverse")
                else:
                    out.append(glyph, style=style)
                out.append(" ")
                if (cell + 1) % cols == 0:
                    out.append("\n")
            return out

        # cells / paged: one dot per document
        start = self.page * capacity
        end = min(total, start + capacity)
        self._page_cells = end - start
        for i, idx in enumerate(range(start, end)):
            st = op.cell_status(idx, running_band)
            style = _STATUS_STYLE[st]
            glyph = _DOT if st != "queued" else "·"
            if i == self.cursor and self.focus_pane == "grid":
                out.append(glyph, style=style + " reverse")
            else:
                out.append(glyph, style=style)
            out.append(" ")
            if (i + 1) % cols == 0:
                out.append("\n")
        return out

    def _render_detail(self, state: RunState) -> Panel:
        op = self._selected_op()
        if op is None:
            return Panel(Text(""), title="Detail", border_style="magenta")

        if self.focus_pane != "grid":
            # Operation-level summary.
            body = Text()
            body.append(f"{op.op_type}:{op.name.split('/')[-1]}\n\n", style="bold")
            body.append(f"step:    {op.step}\n", style="grey70")
            body.append(f"model:   {op.model}\n", style="grey70")
            body.append(f"status:  {op.status}\n", style=_STATUS_STYLE[op.status])
            prof = get_profile(op.op_type)
            if op.total:
                body.append(
                    f"{prof.unit}:  {op.completed:,}/{op.total:,}\n", style="grey70"
                )
            if op.out_count is not None:
                body.append(
                    f"output:  {op.out_count:,} {prof.doc_unit}\n", style="grey70"
                )
            body.append(f"errors:  {op.errors}\n", style="red" if op.errors else "grey70")
            body.append(f"cost:    ${op.cost:.4f}\n", style="green")
            body.append(f"tokens:  {op.tokens:,}\n", style="grey70")
            body.append(f"elapsed: {_fmt_dur(op.elapsed)}\n", style="grey70")
            if prof.summary is not None:
                for line in prof.summary(op):
                    body.append_text(line)
            body.append("\nTab → grid, then ↑↓←→ to inspect documents.", style="dim")
            return Panel(body, title="Operation", border_style="magenta")

        # Document-level detail under the grid cursor.
        if self._mode == "heatmap":
            bucket = math.ceil(op.grid_count / self._capacity)
            lo = self.cursor * bucket
            hi = min(op.grid_count, lo + bucket)
            body = Text()
            body.append(f"Bucket {self.cursor}\n\n", style="bold")
            body.append(f"docs {lo:,}–{hi - 1:,} ({hi - lo} docs)\n", style="grey70")
            done = max(0, min(hi - lo, op.completed - lo))
            body.append(f"{done}/{hi - lo} complete\n", style="green")
            body.append("\n(zoom not available in heatmap mode)", style="dim")
            return Panel(body, title="Detail", border_style="magenta")

        abs_idx = self.page * self._capacity + self.cursor
        return Panel(
            self._render_doc(op, abs_idx), title=f"Document #{abs_idx}", border_style="magenta"
        )

    def _render_doc(self, op: OpState, idx: int) -> Group:
        if idx >= op.grid_count:
            return Group(Text("(no document)", style="grey42"))
        if not op.outputs or idx >= len(op.outputs):
            st = op.cell_status(idx, op_runband(op))
            return Group(
                Text(f"status: {st}\n", style=_STATUS_STYLE.get(st, "grey42")),
                Text(
                    "Output not yet available.\n"
                    "(captured when the operation finishes;"
                    " large runs keep a sample of the first 2,000 docs.)",
                    style="dim",
                ),
            )
        doc = op.outputs[idx]
        obs_key = f"_observability_{op.name.split('/')[-1]}"
        prompt = None
        display = {k: v for k, v in doc.items() if not k.startswith("_observability_")}
        if obs_key in doc and isinstance(doc[obs_key], dict):
            prompt = doc[obs_key].get("prompt")

        cache_key = (op.name, idx)
        body = self._doc_json.get(cache_key)
        if body is None:
            body = _trunc(json.dumps(display, indent=2, default=str), 1400)
            self._doc_json[cache_key] = body

        out = Text()
        out.append("status: done\n\n", style="green")
        out.append("output\n", style="bold underline")
        out.append(body + "\n")
        parts = [out]
        # Operator-specific provenance (reduce source counts, split chunk/parent).
        prof = get_profile(op.op_type)
        if prof.provenance is not None:
            parts.extend(prof.provenance(doc))
        if prompt:
            p = Text()
            p.append("\nprompt\n", style="bold underline")
            p.append(_trunc(str(prompt), 1000), style="grey70")
            parts.append(p)
        return Group(*parts)


def op_runband(op: OpState) -> int:
    return 0


# -- formatting helpers --------------------------------------------------
def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " …"


def _fmt_dur(secs: float) -> str:
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _fmt_k(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


def _heat_style(frac: float, has_error: bool) -> str:
    if has_error:
        return "red"
    # interpolate grey (done=0) -> bright green (done=1)
    r = int(80 - 80 * frac)
    g = int(80 + 120 * frac)
    b = int(80 - 80 * frac)
    return f"rgb({max(0, r)},{min(255, g)},{max(0, b)})"


def run_with_tui(runner: "DSLRunner") -> float:
    """Set up the tracker, launch the TUI, and run the pipeline within it.

    Returns the total pipeline cost. Console output from the runner is silenced
    during the TUI so it doesn't corrupt the full-screen display.
    """
    tracker = ProgressTracker(concurrency=min(runner.max_threads or 1, 64))
    runner.progress_tracker = tracker
    runner._tui_active = True
    set_active_tracker(tracker)

    # Pre-register all operations so they show as "queued" immediately.
    tracker.pipeline_start(runner.list_pipeline_operations())

    # Silence the runner's console for the duration of the TUI.
    saved_console = runner.console
    quiet = _QuietConsole()
    runner.console = quiet

    app = DocetlTUI(tracker, runner=runner)
    try:
        app.run()
    finally:
        set_active_tracker(None)
        runner.console = saved_console
        runner.progress_tracker = None
        runner._tui_active = False

    if app.error is not None:
        raise app.error
    return app.result_cost or 0.0


class _QuietConsole:
    """Minimal stand-in for the runner console that swallows output.

    Implements the handful of methods the runner/operations call so nothing
    leaks onto the terminal while the TUI owns the screen.
    """

    def __init__(self) -> None:
        self.file = io.StringIO()

    def log(self, *a, **k):
        pass

    def print(self, *a, **k):
        pass

    def rule(self, *a, **k):
        pass

    def status(self, *a, **k):
        return _NullStatus()

    def __getattr__(self, _name):
        return lambda *a, **k: None


class _NullStatus:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def start(self):
        pass

    def stop(self):
        pass

    def update(self, *a, **k):
        pass
