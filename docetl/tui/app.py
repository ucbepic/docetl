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
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static

from docetl.operations.utils.cascade_runner import describe_cascade_stats
from docetl.progress.events import OpState, RunState
from docetl.progress.tracker import ProgressTracker, set_active_tracker
from docetl.tui.profiles import get_profile

if TYPE_CHECKING:
    from docetl.runner import DSLRunner

# Status -> (glyph, rich color)
_DOT = "●"
_EMPTY = "○"  # not-yet-done cell: an outline circle the same size as a filled one
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
    #ops { width: 40; border: round $primary; padding: 0 1; }
    #middle { border: round $primary; padding: 0 1; }
    #detail { width: 52; border: round $accent; padding: 0 1; }
    #detail_body { width: 1fr; }
    #ops, #middle, #detail { border-title-align: center; border-title-color: $text-muted; }
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
        # Cache of extracted doc views, keyed by (op name, index). Outputs are
        # immutable once captured at op_done, so the detail pane re-renders the
        # selected cell ~7x/s without re-extracting/truncating fields each frame.
        self._doc_view_cache: dict[tuple[str, int], tuple] = {}

    # -- composition -----------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static(id="ops")
            with Vertical(id="middle"):
                yield Static(id="grid_title")
                yield Static(id="grid")
            with VerticalScroll(id="detail"):
                yield Static(id="detail_body")

    def on_mount(self) -> None:
        self.title = self._pipeline_label()
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
    def _pipeline_label(self) -> str:
        if self.runner is not None:
            return self.runner.pipeline_label()
        return "DocETL pipeline"

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
            ops = self.query_one("#ops", Static)
            ops.update(self._render_ops(state))
            ops.border_title = "Operations" + (" ◀" if self.focus_pane == "ops" else "")

            op = self._selected_op()
            middle = self.query_one("#middle")
            middle.border_title = (
                f"{op.op_type}:{op.name.split('/')[-1]}" if op else "Documents"
            ) + (" ◀" if self.focus_pane == "grid" else "")
            self.query_one("#grid_title", Static).update(self._render_grid_title(state))
            self.query_one("#grid", Static).update(self._render_grid(state))

            detail_container = self.query_one("#detail", VerticalScroll)
            detail_body = self.query_one("#detail_body", Static)
            title, body = self._render_detail(state)
            detail_body.update(body)
            detail_container.border_title = title
        except Exception:
            # Widgets may not be mounted on the very first tick.
            pass

    def _render_ops(self, state: RunState) -> Group:
        head = Text()
        head.append(f"{self._pipeline_label()}\n", style="bold")
        head.append(f"{state.done_ops}/{len(state.ops)} ops", style="grey70")
        head.append("   ")
        head.append(_fmt_cost(state.total_cost), style="green")
        head.append("   ")
        head.append(f"{_fmt_dur(state.elapsed)}\n", style="grey70")
        if self.error is not None:
            head.append("✗ failed\n", style="bold red")
            head.append(
                _trunc(f"{type(self.error).__name__}: {self.error}", 240) + "\n",
                style="red",
            )
            head.append("press q to quit and see the full traceback\n", style="dim")
        elif state.finished:
            head.append("✓ complete\n", style="bold green")
            head.append("press q to exit\n", style="bold yellow")

        body = Text()
        last_step = None
        for i, op in enumerate(state.ops):
            if op.step != last_step:
                body.append(f"\n{op.step}\n", style="bold cyan")
                last_step = op.step
            selected = i == self.sel_op
            glyph = _OP_GLYPH[op.status]
            gstyle = _STATUS_STYLE[op.status]
            # Line 1: status glyph + op label. The selection bar lives here only,
            # so it reads as a clean highlighted row rather than a ragged block.
            line = Text()
            line.append(f" {glyph} ", style=gstyle)
            line.append(
                _trunc(f"{op.op_type}:{op.name.split('/')[-1]}", 30),
                style="bold" if selected else "",
            )
            if selected:
                line.stylize("reverse")
            body.append_text(line)
            body.append("\n")
            # Lines below the op label: compact stats.
            ci = op.cascade_info
            if ci:
                # Proxy sub-line
                proxy_line = Text("    ")
                proxy_line.append("proxy", style="green")
                proxy_line.append(f"  {ci['proxy_calls']} scored", style="grey70")
                pc = ci.get("proxy_cost", 0)
                if pc:
                    proxy_line.append(f"  {_fmt_cost(pc)}", style="green")
                body.append_text(proxy_line)
                body.append("\n")

                # Oracle sub-line: live progress during run, final counts when done
                oracle_line = Text("    ")
                oracle_line.append("oracle", style="cyan")
                if op.status == "done":
                    oracle_line.append(f"  {ci['oracle_calls']}", style="grey70")
                elif op.total:
                    oracle_line.append(f"  {op.completed}/{op.total}", style="grey70")
                    oracle_line.append(f" {int(100 * op.completed / op.total)}%", style="yellow")
                budget = ci.get("label_budget")
                if budget:
                    oracle_line.append(f" (budget {budget})", style="grey42")
                oc = ci.get("oracle_cost", 0)
                if oc:
                    oracle_line.append(f"  {_fmt_cost(oc)}", style="green")
                if op.elapsed >= 1:
                    oracle_line.append(f"  {_fmt_dur(op.elapsed)}", style="grey54")
                body.append_text(oracle_line)
                body.append("\n")
            else:
                frags: list[Text] = []
                if op.phase:
                    frags.append(Text(op.phase, style="cyan"))
                if op.total:
                    f = Text(f"{op.completed}/{op.total}", style="grey70")
                    if op.status == "running":
                        f.append(f" {int(100 * op.completed / op.total)}%", style="yellow")
                    frags.append(f)
                if op.cost:
                    frags.append(Text(_fmt_cost(op.cost), style="green"))
                if op.status in ("done", "running") and op.elapsed >= 1:
                    frags.append(Text(_fmt_dur(op.elapsed), style="grey54"))
                if op.errors:
                    frags.append(Text(f"!{op.errors}", style="red"))
                if frags:
                    sub = Text("    ")
                    for j, f in enumerate(frags):
                        if j:
                            sub.append("   ")
                        sub.append_text(f)
                    body.append_text(sub)
                    body.append("\n")

        return Group(head, body)

    def _render_grid_title(self, state: RunState) -> Text:
        op = self._selected_op()
        if op is None:
            return Text("")
        prof = get_profile(op.op_type)
        t = Text()
        if op.status == "done" and op.out_count is not None:
            t.append(f"{op.out_count:,} {prof.doc_unit}", style="grey70")
        elif op.total:
            t.append(f"{op.completed:,}/{op.total:,} {prof.unit}", style="grey70")
        else:
            t.append("? docs", style="grey62")  # count not known yet
        if op.phase:
            t.append(f"   {op.phase}", style="cyan")
        t.append("   ")
        t.append(op.status, style=_STATUS_STYLE[op.status])
        if self._mode == "heatmap":
            t.append("     heatmap (each cell = a bucket of docs)", style="magenta")
        elif self._mode == "paged":
            t.append(
                f"     page {self.page + 1}/{self._page_count} [PgDn]", style="grey50"
            )
        if op.cascade_info and op.cascade_info.get("item_escalated"):
            t.append("   ")
            t.append(_DOT, style="green")
            t.append(" proxy ", style="dim")
            t.append(_DOT, style="cyan")
            t.append(" oracle", style="dim")
        return t

    def _render_grid(self, state: RunState) -> Text:
        # Reset paging state first so a no-grid op can't leave a stale
        # "page 1/3" in the title from a previously-selected operation.
        self._mode, self._page_count, self._page_cells = "cells", 1, 0
        op = self._selected_op()
        if op is None:
            return Text("")

        # Only draw the per-document grid when we actually know a document count:
        # the op has finished (we have its outputs), or its live progress counts
        # documents. Otherwise show a "?" rather than an empty or misleading grid.
        prof = get_profile(op.op_type)
        if not self._grid_shows_docs(op):
            out = Text()
            out.append("?\n\n", style="bold yellow")
            if op.status == "done":
                out.append("no documents", style="grey42")
            elif not prof.grid_is_docs:
                out.append(
                    f"document count not known yet — {op.op_type} reports progress\n"
                    f"as {prof.unit}; documents appear once it finishes",
                    style="grey54",
                )
            else:
                out.append("document count not known yet", style="grey54")
            return out

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
                glyph = _DOT if frac > 0 else _EMPTY
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
            glyph = _DOT if st != "queued" else _EMPTY
            if st == "done":
                role = op.cell_cascade_role(idx)
                if role == "oracle":
                    style = "cyan"
            if i == self.cursor and self.focus_pane == "grid":
                out.append(glyph, style=style + " reverse")
            else:
                out.append(glyph, style=style)
            out.append(" ")
            if (i + 1) % cols == 0:
                out.append("\n")
        return out

    def _grid_shows_docs(self, op: OpState) -> bool:
        """Whether the document grid can be drawn for this op: it has finished
        (so we have outputs) or its live progress counts documents."""
        return bool(op.grid_count) and (
            op.status == "done" or get_profile(op.op_type).grid_is_docs
        )

    def _render_detail(self, state: RunState) -> tuple[str, Group | Text]:
        op = self._selected_op()
        if op is None:
            return "Detail", Text("")

        # Show the operation summary when the ops pane is focused, or when there
        # are no enumerable documents to inspect (the grid is a "?").
        if self.focus_pane != "grid" or not self._grid_shows_docs(op):
            # Operation-level summary.
            body = Text()
            body.append(f"step:    {op.step}\n", style="grey70")
            body.append(f"model:   {op.model}\n", style="grey70")
            body.append(f"status:  {op.status}\n", style=_STATUS_STYLE[op.status])
            if op.phase:
                body.append(f"phase:   {op.phase}\n", style="cyan")
            prof = get_profile(op.op_type)
            if op.total:
                body.append(
                    f"{prof.unit}:  {op.completed:,}/{op.total:,}\n", style="grey70"
                )
            if op.out_count is not None:
                body.append(
                    f"output:  {op.out_count:,} {prof.doc_unit}\n", style="grey70"
                )
            body.append(
                f"errors:  {op.errors}\n", style="red" if op.errors else "grey70"
            )
            body.append(f"cost:    ${op.cost:.4f}\n", style="green")
            body.append(f"tokens:  {op.tokens:,}\n", style="grey70")
            body.append(f"elapsed: {_fmt_dur(op.elapsed)}\n", style="grey70")
            if prof.summary is not None:
                for line in prof.summary(op):
                    body.append_text(line)
            if op.cascade_info:
                body.append_text(_render_cascade_info(op.cascade_info))
            body.append("\nTab → grid, then ↑↓←→ to inspect documents.", style="dim")
            return "Operation", body

        # Document-level detail under the grid cursor.
        if self._mode == "heatmap":
            bucket = math.ceil(op.grid_count / self._capacity)
            lo = self.cursor * bucket
            hi = min(op.grid_count, lo + bucket)
            body = Text()
            body.append(f"docs {lo:,}–{hi - 1:,} ({hi - lo} docs)\n", style="grey70")
            done = max(0, min(hi - lo, op.completed - lo))
            body.append(f"{done}/{hi - lo} complete\n", style="green")
            body.append("\n(zoom not available in heatmap mode)", style="dim")
            return f"Bucket {self.cursor}", body

        abs_idx = self.page * self._capacity + self.cursor
        return f"Document #{abs_idx}", self._render_doc(op, abs_idx)

    def _render_doc(self, op: OpState, idx: int) -> Group:
        if idx >= op.grid_count:
            return Group(Text("(no document)", style="grey42"))
        if not op.outputs or idx >= len(op.outputs):
            st = op.cell_status(idx, op_runband(op))
            return Group(
                _kv("status", st, value_style=_STATUS_STYLE.get(st, "grey42")),
                Text(
                    "\nOutput not yet available — captured when the operation\n"
                    "finishes (large runs keep a sample of the first 2,000 docs).",
                    style="grey50",
                ),
            )

        doc = op.outputs[idx]
        prof = get_profile(op.op_type)
        rows, prompt, provenance = self._doc_view(op, prof, idx, doc)

        out = Text()
        role = op.cell_cascade_role(idx)
        if role == "oracle":
            out.append_text(_kv("status", "done — oracle decided", value_style="cyan"))
        elif role == "proxy":
            out.append_text(_kv("status", "done — proxy decided", value_style="green"))
        else:
            out.append_text(_kv("status", "done", value_style="green"))
        out.append("\n")
        if op.cascade_info:
            out.append_text(_render_cascade_doc(op.cascade_info, idx))
            out.append("\n")
        for key, value in rows:
            # short scalars read best inline; long / multi-line values get their
            # own indented block so nothing wraps awkwardly against the label.
            if len(value) <= 36 and "\n" not in value:
                out.append_text(_kv(key, value))
            else:
                out.append(f"{key}\n", style="bold #7dcfff")
                for line in value.splitlines() or [""]:
                    out.append(f"  {line}\n", style="grey85")
        parts = [out]
        if provenance:
            parts.append(_section("provenance", provenance))
        if prompt:
            parts.append(_section("prompt", _trunc(prompt, 1000), style="grey70"))
        return Group(*parts)

    def _doc_view(self, op, prof, idx, doc):
        """Build (display rows, prompt, provenance) for one document, cached.

        Outputs are immutable after ``op_done``, so the per-cell field extraction
        (which truncates long values) is computed once rather than every frame.
        """
        cache_key = (op.name, idx)
        cached = self._doc_view_cache.get(cache_key)
        if cached is not None:
            return cached

        name = op.name.split("/")[-1]
        obs = doc.get(f"_observability_{name}")
        prompt = None
        if isinstance(obs, dict):
            prompt = obs.get("prompt")
            if (
                prompt is None
                and isinstance(obs.get("prompts"), list)
                and obs["prompts"]
            ):
                prompt = obs["prompts"][0]

        consumed = prof.consumed_keys(doc) if prof.consumed_keys else set()
        rows = []
        for k, v in doc.items():
            if k.startswith("_") or k in consumed:
                continue  # internal bookkeeping / surfaced as provenance instead
            value = (
                v
                if isinstance(v, str)
                else json.dumps(v, ensure_ascii=False, default=str)
            )
            rows.append((k, _trunc(str(value), 280)))

        provenance = prof.provenance(op, doc) if prof.provenance else None
        result = (rows, prompt, provenance)
        self._doc_view_cache[cache_key] = result
        return result


def op_runband(op: OpState) -> int:
    return 0


# -- formatting helpers --------------------------------------------------
def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " …"


def _kv(key: str, value: str, value_style: str = "grey85") -> Text:
    """One ``key: value`` row — a muted label and its value, no JSON braces."""
    t = Text()
    t.append(f"{key}: ", style="bold #7dcfff")
    t.append(f"{value}\n", style=value_style)
    return t


def _section(label: str, value: str, style: str = "grey70") -> Text:
    """A titled block: a bold header line over its (un-bolded) value."""
    t = Text()
    t.append(f"\n{label}\n", style="bold")
    t.append(value, style=style)
    return t


def _fmt_cost(c: float) -> str:
    """Compact cost: avoids the misleading ``$0.000`` for sub-cent runs."""
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


def _render_cascade_info(info: dict) -> Text:
    """Compact cascade stats block for the operation detail pane.

    During the oracle phase, only partial data is available (proxy stats +
    histogram). After the cascade finishes, all fields are present.
    """
    t = Text()
    t.append("\ncascade\n", style="bold magenta")
    proxy_cost = info.get("proxy_cost", 0)
    oracle_cost = info.get("oracle_cost", 0)
    guarantee = info.get("guarantee", "")
    is_calibrated = guarantee in ("precision", "recall")

    t.append(
        f"  proxy   {info['proxy_model']}  "
        f"{info['proxy_calls']:,} scored",
        style="cyan",
    )
    if proxy_cost:
        t.append(f"  ${proxy_cost:.4f}", style="cyan")
    t.append("\n")

    if "oracle_calls" in info:
        descs = describe_cascade_stats(info)
        t.append(
            f"  oracle  {info['oracle_model']}  "
            f"{descs['oracle_desc']}  ${oracle_cost:.4f}\n",
            style="cyan",
        )
    elif info.get("oracle_model"):
        budget = info.get("label_budget", "?")
        t.append(
            f"  oracle  {info['oracle_model']}  sampling… (budget {budget})\n",
            style="cyan",
        )

    if "target" in info:
        target = info["target"]
        t.append(f"  {guarantee} ≥ {target:.0%}", style="yellow")
        t.append(f"  δ={info.get('delta', '?')}\n", style="grey70")

    if "oracle_calls" in info:
        td = descs["threshold_desc"]
        if "n/a" not in td:
            t.append(f"  threshold  {td}\n", style="yellow")
        elif is_calibrated:
            t.append(f"  threshold  {td}\n", style="dim")

        if is_calibrated:
            pass  # result_desc already covers calibration items
        elif "escalation_rate" in info:
            esc = info["escalation_rate"]
            t.append(f"  escalation {esc:.0%}", style="red" if esc >= 0.5 else "green")
            served = info["served_by_proxy"]
            t.append(f"  ({served:,} served by proxy)\n", style="grey70")
    if info.get("cached"):
        t.append("  (cached)\n", style="dim")

    escalated = info.get("item_escalated")
    if escalated:
        kept = info.get("kept_input_indices")
        if kept:
            n_oracle = sum(1 for idx in kept if idx < len(escalated) and escalated[idx])
            n_proxy = len(kept) - n_oracle
            label = "kept"
        else:
            n_oracle = sum(escalated)
            n_proxy = len(escalated) - n_oracle
            label = "items"
        t.append(f"  {label}  ", style="dim")
        t.append(_DOT, style="green")
        t.append(f" {n_proxy} proxy", style="green")
        t.append("  ", style="dim")
        t.append(_DOT, style="cyan")
        t.append(f" {n_oracle} oracle\n", style="cyan")

    score_hist = info.get("score_hist")
    if score_hist:
        t.append_text(_render_score_bar(score_hist, info.get("threshold")))

    return t


def _render_cascade_doc(cascade_info: dict, output_idx: int) -> Text:
    """Per-document cascade info for the detail pane."""
    t = Text()

    proxy_scores = cascade_info.get("item_proxy_scores", [])
    proxy_labels = cascade_info.get("item_proxy_labels", [])
    escalated = cascade_info.get("item_escalated", [])
    kept = cascade_info.get("kept_input_indices")
    input_idx = output_idx
    if kept and output_idx < len(kept):
        input_idx = kept[output_idx]

    has_data = False
    if input_idx < len(proxy_scores):
        has_data = True
        score = proxy_scores[input_idx]
        t.append("cascade\n", style="bold magenta")

        if input_idx < len(proxy_labels):
            proxy_label = bool(proxy_labels[input_idx])
        else:
            proxy_label = score > 0.5
        confidence = score if proxy_label else (1.0 - score)
        t.append("  proxy said:  ", style="dim")
        t.append(f"{proxy_label}", style="green" if proxy_label else "red")
        t.append(f"  ({confidence:.0%})\n", style="grey70")

    if input_idx < len(escalated) and escalated[input_idx]:
        if not has_data:
            t.append("cascade\n", style="bold magenta")
        t.append("  kept by:     ", style="dim")
        t.append("oracle", style="cyan")
        t.append(" said ", style="dim")
        t.append("True\n", style="green")
    elif has_data:
        t.append("  kept by:     ", style="dim")
        t.append("proxy", style="green")
        t.append(" said ", style="dim")
        t.append("True\n", style="green")

    threshold = cascade_info.get("threshold")
    if threshold is not None and threshold >= 0.01:
        t.append("  threshold:   ", style="dim")
        t.append(f"{threshold:.3f}\n", style="yellow")

    return t


_HIST_CHARS = " ▁▂▃▄▅▆▇█"


def _render_score_bar(hist: list[int], threshold: float | None) -> Text:
    """Compact proxy score distribution with threshold marker.

    Renders as two lines:
      scores  ▁▃▅▇█▇▅▃▁▁▂▃▅▇█▆▃▂▁
              0        ↑t=0.72    1
    """
    t = Text()
    n_bins = len(hist)
    peak = max(hist) or 1
    pad = "  "
    label = "scores  "

    t.append(pad + label, style="dim")
    for i, count in enumerate(hist):
        level = int(count / peak * (len(_HIST_CHARS) - 1))
        if count > 0 and level == 0:
            level = 1
        bin_center = (i + 0.5) / n_bins
        if threshold is not None and threshold >= 0.01 and bin_center >= threshold:
            t.append(_HIST_CHARS[level], style="green")
        else:
            t.append(_HIST_CHARS[level], style="bright_cyan" if level > 0 else "grey42")
    t.append("\n")

    axis_pad = pad + " " * len(label)
    if threshold is not None and threshold >= 0.01:
        pos = int(threshold * n_bins)
        pos = max(0, min(pos, n_bins - 1))
        marker = f"↑t={threshold:.2f}"
        before = max(0, pos)
        after = max(0, n_bins - before - len(marker))
        t.append(axis_pad, style="dim")
        t.append(" " * before, style="dim")
        t.append(marker, style="yellow")
        t.append(" " * after + "\n", style="dim")
    else:
        t.append(axis_pad, style="dim")
        t.append("0" + " " * (n_bins - 2) + "1\n", style="grey42")

    return t


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
    # tqdm lazily builds a *multiprocessing* write-lock on first use, which
    # spawns a helper process; under Textual's control of the terminal that
    # fork_exec fails ("bad value(s) in fds_to_keep") and kills the run. Pin
    # tqdm to a plain thread lock so any progress bar created while the
    # pipeline runs in the worker thread is safe.
    import threading

    from tqdm import tqdm as _tqdm

    _tqdm.set_lock(threading.RLock())

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

    # Back in the normal terminal: print a short summary of the finished run.
    from rich.panel import Panel

    cost = app.result_cost or 0.0
    try:
        out_path = runner.get_output_path(require=False)
    except Exception:
        out_path = None
    summary = f"Cost: [green]${cost:.4f}[/green]"
    if out_path:
        summary += f"\nOutput: [dim]{out_path}[/dim]"
    saved_console.log(Panel(summary, title="Run complete"))
    return cost


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
