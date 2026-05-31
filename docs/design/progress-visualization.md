# Design: Interactive Progress Visualization (Issue #487)

Status: **Proposal / for review** — no implementation yet.

## 1. Goal

Replace today's scrolling wall of log text with a full-screen, interactive
progress view for a pipeline run — inspired by Claude Code's `/workflows` screen:

- A left panel listing pipeline **steps → operations** with live status, progress
  counts, cost, tokens, and elapsed time.
- A middle panel showing a **grid of dots**, one per document, that turns green as
  each document completes (queued / running / done / error).
- A right **detail panel**: click a dot to inspect that document's input, output,
  the LLM prompt that produced it, cost/latency, errors, and its **provenance**
  (which input docs / operation produced it).
- Full keyboard + mouse navigation (↑/↓, ←/→, Enter, click).

The hard constraint from the issue: a run can have **tens of thousands of
documents**, so the UI must stay legible at that scale and never depend on
rendering every dot.

We target **both** surfaces: an interactive terminal TUI and the existing web UI.

## 2. Current state (what we build on)

| Concern | Today | File |
| --- | --- | --- |
| Run entry | `docetl run f.yaml` → `DSLRunner.load_run_save()` | `docetl/cli.py:202`, `docetl/runner.py:491` |
| Op execution | Pull-based DAG; each `OpContainer.next()` wraps op in `console.status("Running …")` and logs one `✓` line | `docetl/containers.py:424` |
| Per-doc progress | `RichLoopBar` (a `tqdm` bar) updated once per batch in a `ThreadPoolExecutor` | `docetl/operations/utils/progress.py`, `docetl/operations/map.py:572` |
| Console | `ThreadSafeConsole` buffers rendered text to a `StringIO` when `USE_FRONTEND=true`; plain Rich console to stdout otherwise | `docetl/console.py` |
| Web streaming | Websocket polls `runner.console.file.getvalue()` every 0.5s and ships **rendered ANSI text** to the browser | `server/app/routes/pipeline.py:378` |
| Web rendering | Next.js renders that ANSI text via `@agbishop/react-ansi-18` | `website/src/contexts/WebSocketContext.tsx`, `website/src/components/PipelineGui.tsx` |
| Checkpoints | Each op's full output persisted as JSON | `runner.py:_save_checkpoint` |
| Observability | `enable_observability: true` stores the LLM prompt on each output as `_observability_<op>` | `docetl/operations/map.py:441` |

**Gaps:**
1. **No structured progress signal.** Progress is text only (spinner + tqdm counts
   + log lines). The web UI literally re-renders terminal text.
2. **No per-document status events.** `RichLoopBar` knows only `n/total`; nothing
   emits "doc X finished with status/cost/tokens".
3. **No document identity or provenance.** `add_uuid` adds an id only if the user
   wires it; `rank` uses a throwaway `_docetl_id`. The runner keeps no
   input→output lineage.

The dot-grid cannot be bolted onto the text stream. The linchpin is a structured
telemetry layer that both UIs consume. **Foundation first.**

## 3. Architecture

```
                ┌─────────────────────────────────────────┐
   operations ──┤  ProgressTracker  (thread-safe event bus) │
   (map/filter/ └───────────────┬─────────────────┬─────────┘
    reduce/…)                    │                 │
                        in-process subscriber   JSON serializer
                                 │                 │
                        ┌────────▼──────┐   ┌──────▼─────────────┐
                        │ Textual TUI   │   │ Websocket: events  │──► Next.js grid UI
                        │ (docetl run)  │   │ (server)           │
                        └───────────────┘   └────────────────────┘
```

- Operations already hold `self.runner` (`base.py:71`), so they emit via
  `self.runner.progress_tracker.emit(...)`. No signature changes to operations.
- The tracker is the single source of truth. The terminal TUI subscribes
  in-process; the server serializes events to JSON over the existing websocket.

### 3.1 Event schema

Emitted as lightweight dataclasses in-process; serialized to JSON for the web.

```python
# docetl/progress/events.py
DocStatus = Literal["queued", "running", "done", "error", "filtered"]

@dataclass
class PipelineStart:   run_id: str; steps: list[StepInfo]; t: float
@dataclass
class OpStart:         run_id; step: str; op_name: str; op_type: str
                       model: str | None; total_docs: int | None; t: float
@dataclass
class DocDone:         run_id; op_name: str; doc_id: str
                       status: DocStatus; cost: float; tokens: TokenUsage | None
                       latency: float; error: str | None; t: float
@dataclass
class OpDone:          run_id; op_name; in_count; out_count
                       cost; tokens; elapsed; t: float
@dataclass
class StepDone:        run_id; step; cost; t: float
@dataclass
class PipelineDone:    run_id; cost; elapsed; t: float
```

Notes:
- `DocDone` is the high-frequency event. For 1:1 ops (map/filter) it is per
  document; for aggregating ops (reduce/resolve/equijoin) we emit at group/batch
  granularity (one `DocDone` per output group) plus an `OpDone` summary. This keeps
  event volume bounded and matches what those ops can meaningfully report.
- The tracker keeps a compact in-memory model (`RunState`) that the UI reads:
  per-op counters (`queued/running/done/error`), per-doc status, and lineage.
  Raw input/output bodies are **not** held in memory — they are read lazily from
  checkpoints on demand (§3.3).

### 3.2 Provenance model

Introduce a runner-managed internal id, `__docetl_id`, assigned to every record at
**scan time** (leaf of the DAG). It is:
- copied through 1:1 ops (map, filter-pass),
- propagated as a parent-list through fan-out/fan-in ops,
- **stripped before final save** (same pattern `rank` already uses with
  `_docetl_id`, `rank.py:884`), so it never leaks into user output.

The tracker records lineage edges per op:

| Op type | Lineage semantics |
| --- | --- |
| map / parallel_map | 1 input → 1 output (id preserved) |
| split | 1 input → N output chunks (children carry `parent_id`) |
| filter | 1 input → 0 or 1 output (record drops as `filtered`) |
| reduce / resolve | N inputs → 1 output (output stores `source_ids`) |
| equijoin | (left_id, right_id) → 1 output |
| gather / unnest | recombination tracked via existing `doc_id` keys |

A document's provenance chain is then a reverse walk over these edges — answering
"this output came from input docs A, B via op `resolve_x`".

### 3.3 Document inspection (input/output/prompt)

No new bulk storage. The detail panel assembles a doc view from sources that
already exist:
- **Output body** + intermediate: per-op checkpoint JSON (`_save_checkpoint`),
  loaded lazily and indexed by `__docetl_id`.
- **Prompt / LLM I/O**: the `_observability_<op>` field, surfaced when
  `enable_observability` is set (we will default it on under the TUI, configurable).
- **Input body**: the upstream op's checkpoint via the lineage edge.

## 4. Terminal TUI (Textual)

New optional dependency: **`textual`** (Rich's sibling library). Plain Rich
`Live`+`Layout` cannot do real arrow-key/mouse navigation; Textual is the right
tool and integrates with the existing Rich renderables.

- Entry: `docetl run f.yaml --tui` (auto-enabled when stdout is a TTY; `--no-tui`
  forces today's plain logging; non-TTY / CI always falls back). The runner runs in
  a worker thread (mirrors the server's `asyncio.to_thread(runner.load_run_save)`
  at `pipeline.py:396`); events flow to the UI via a thread-safe queue.

### 4.1 Layout (three panes, like `/workflows`)

```
┌ Steps ────────┐┌ Operation: map:extract ─ 1,240/10,000 ┐┌ Document d4f1 ───────┐
│▶ 1 extract 12%││ ✓✓✓✓✓✓◐◐◐○○○○○○○○○○○○○○○○○○○○○○○○○○○○○ ││ status: done         │
│  2 dedupe   ⏸ ││ ✓✓✓✓✓✓✓✓✓◐◐○○○○○○○○○○○○○○○○○○○○○○○○○○○ ││ cost: $0.004         │
│               ││ … showing 1–2,000 of 10,000  [PgDn] … ││ ─ input ─ output ─   │
│ cost $1.20    ││                                       ││ ─ prompt ─ provenance│
└───────────────┘└───────────────────────────────────────┘└──────────────────────┘
  ↑/↓ select op    ←/→ page · Enter inspect · / filter      o open · q quit
```

- **Left** — `Tree`/`ListView` of steps→ops with status glyph, `done/total` %,
  rolling cost & tokens, elapsed. ↑/↓ selects; selection drives the middle pane.
- **Middle** — virtualized dot grid for the selected op. Cell color = status
  (○ queued, ◐ running, ✓ done green, ✗ error red, · filtered dim). A cursor moves
  with arrows; Enter/click selects a doc.
- **Right** — detail for the selected doc: status, cost, latency, tokens; tabbed
  Input / Output / Prompt / Provenance; error traceback if any.

### 4.2 Scaling to tens of thousands of docs (the issue's core concern)

1. **Per-op grids**, never one global grid — bounds what's on screen.
2. **Virtualized viewport**: render only visible cells; footer shows
   "1–2,000 of 40,000"; PgUp/PgDn pages. Live counters mean you never need to see
   every dot.
3. **Heatmap/aggregate mode** above a threshold (e.g. >5k docs): each cell
   aggregates a bucket of docs, shaded by completion ratio; zoom in to expand.
4. **Filters** (`/`): errors-only, running-only, or jump-to-doc-by-id — so triage
   doesn't require scanning.

### 4.3 Keys

`↑/↓` move within pane · `←/→` switch pane / page grid · `Enter`/click inspect ·
`o` dump doc JSON to a temp file / `$EDITOR` · `/` filter · `e` errors-only ·
`q` quit (run continues / or prompt to cancel).

## 5. Web UI parity

- Server: add an event-stream mode to the websocket. Instead of (or alongside)
  `{"type":"output", data: <ansi text>}`, send `{"type":"event", data: <Event JSON>}`.
  Keep the text channel for backward compatibility (the current ANSI viewer keeps
  working during migration).
- Client: a new React view in `website/` mirroring the 3-pane layout. The dot grid
  uses a canvas or a virtualized grid (e.g. existing virtualization deps) to handle
  10k+ cells; a detail drawer fetches doc bodies via a new
  `GET /doc/{run_id}/{op}/{doc_id}` endpoint backed by checkpoints.
- Reuse Radix UI primitives already in `website/package.json`.

## 6. File-by-file change list

**New**
- `docetl/progress/__init__.py`
- `docetl/progress/events.py` — event dataclasses + JSON (de)serialization.
- `docetl/progress/tracker.py` — `ProgressTracker` (thread-safe), `RunState`,
  lineage store, subscriber interface.
- `docetl/tui/app.py` + widgets (`steps_panel.py`, `doc_grid.py`, `detail_panel.py`).
- `tests/test_progress_tracker.py`, `tests/test_provenance.py`.
- `website/src/components/RunProgress/*` (grid, steps list, detail drawer).

**Modified**
- `docetl/runner.py` — construct `self.progress_tracker`; emit `pipeline/step`
  lifecycle in `load_run_save` / `StepBoundary`; assign `__docetl_id` at scan;
  strip it in `save()`.
- `docetl/containers.py` — emit `OpStart`/`OpDone` around `OpContainer.next()`'s
  execute block (`containers.py:524`); record lineage from input/output counts.
- `docetl/operations/utils/progress.py` — `RichLoopBar.update()` optionally emits a
  `DocDone`; add a per-item emit hook for the batch loops.
- `docetl/operations/map.py` / `filter.py` / `reduce.py` / `resolve.py` /
  `equijoin.py` — emit per-doc/per-group `DocDone` with id, cost, tokens, latency,
  error; populate lineage edges. (map/filter first; others coarser.)
- `docetl/cli.py` — `--tui/--no-tui` flag on `run`; launch TUI around the runner.
- `docetl/console.py` — no behavior change; TUI uses its own Textual render path.
- `server/app/routes/pipeline.py` — add structured-event channel to
  `websocket_run_pipeline`; new doc-fetch endpoint.
- `website/src/contexts/WebSocketContext.tsx` / `PipelineGui.tsx` — parse `event`
  messages and route to the new components.
- `pyproject.toml` — add `textual` (optional `[tui]` extra) so headless installs
  stay slim.

## 7. Phasing

1. **Phase 0 — Foundation**: events, tracker, `__docetl_id` + provenance, emit from
   map/filter. Unit-tested, no UI. Lowest risk; unblocks everything.
2. **Phase 1 — Terminal TUI**: Textual app + 3 panes + scaling modes.
3. **Phase 2 — Coarse-op coverage**: reduce/resolve/equijoin/split lineage.
4. **Phase 3 — Web parity**: event channel + React grid + doc endpoint.

## 8. Testing

- Unit: tracker thread-safety under concurrent emits; provenance edges for
  split/filter/reduce/equijoin.
- Integration: run a small real pipeline (OpenAI keys available) — e.g. a map over
  ~20 docs — and snapshot the event stream; assert counts/costs reconcile with
  `total_cost`.
- Manual: drive the TUI via the `run`/`verify` skill; screenshot at small and
  synthetic-large (40k stub docs) scale to validate the scaling modes.

## 9. Risks & open questions

- **New dependency** (`textual`): mitigated via an optional extra + TTY-gated
  auto-enable and a clean fallback to current logging.
- **Threading**: tracker must be lock-guarded; UI consumes via a queue, never
  touching op threads directly.
- **No regressions**: existing stdout logging and the current ANSI web stream must
  keep working; the TUI and event channel are additive/opt-in.
- **Event volume** at 100k+ docs: cap `DocDone` granularity (batch-level fallback)
  and coalesce on the UI side.
- Open: should `enable_observability` default **on** under the TUI (richer detail
  pane, slightly more memory/storage) or stay opt-in? Recommend on-under-TUI,
  configurable.
