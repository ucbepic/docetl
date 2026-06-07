"""Web-based progress + feedback UI for non-interactive environments.

Starts a lightweight HTTP server alongside the pipeline so a human can open a
browser, watch outputs stream in, give per-document or pipeline-level feedback,
and kill the pipeline if needed.  The agent reads feedback from stdout and from
a JSON file written on exit.

No external web framework required — uses Python's built-in http.server.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from docetl.progress.tracker import PipelineKilled, ProgressTracker, set_active_tracker

if TYPE_CHECKING:
    from docetl.runner import DSLRunner


# ---------------------------------------------------------------------------
# Feedback store
# ---------------------------------------------------------------------------
class FeedbackStore:
    """Thread-safe store for human feedback collected via the web UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self.doc_feedback: list[dict] = []
        self.pipeline_feedback: list[dict] = []
        self.kill_reason: str | None = None

    def add_doc_feedback(self, op_name: str, doc_index: int, doc_snapshot: dict, text: str):
        with self._lock:
            self.doc_feedback.append({
                "operation": op_name,
                "doc_index": doc_index,
                "doc_snapshot": {k: v for k, v in doc_snapshot.items() if not k.startswith("_")},
                "feedback": text,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

    def add_pipeline_feedback(self, text: str):
        with self._lock:
            self.pipeline_feedback.append({
                "feedback": text,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "pipeline_feedback": list(self.pipeline_feedback),
                "doc_feedback": list(self.doc_feedback),
                "killed": self.kill_reason is not None,
                "kill_reason": self.kill_reason,
            }

    @property
    def has_any(self) -> bool:
        with self._lock:
            return bool(self.doc_feedback or self.pipeline_feedback or self.kill_reason)


# ---------------------------------------------------------------------------
# SSE state broadcaster
# ---------------------------------------------------------------------------
class _Broadcaster:
    """Pushes state snapshots to SSE subscribers."""

    def __init__(self, tracker: ProgressTracker, feedback: FeedbackStore):
        self._tracker = tracker
        self._feedback = feedback
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sent_docs: dict[str, int] = {}  # op_name -> docs already sent

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=50)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            self._push()
            self._stop.wait(1.0)
        self._push()  # final push

    def _push(self):
        state = self._tracker.snapshot()
        event = self._build_event(state)
        with self._lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass

    def _build_event(self, state) -> dict:
        ops = []
        all_docs = []
        for op in state.ops:
            ops.append({
                "name": op.name,
                "op_type": op.op_type,
                "model": op.model,
                "status": op.status,
                "total": op.total,
                "completed": op.completed,
                "errors": op.errors,
                "out_count": op.out_count,
                "cost": op.cost,
                "elapsed": op.elapsed,
            })
            for i, doc in enumerate(op.outputs):
                all_docs.append({
                    "op_name": op.name,
                    "op_type": op.op_type,
                    "doc_index": i,
                    "fields": {k: _trunc(v) for k, v in doc.items() if not k.startswith("_")},
                })

        return {
            "ops": ops,
            "all_docs": all_docs,
            "total_cost": state.total_cost,
            "elapsed": state.elapsed,
            "finished": state.finished,
            "feedback_count": len(self._feedback.doc_feedback) + len(self._feedback.pipeline_feedback),
        }


def _trunc(v, n=500) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + " …"


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
def _make_handler(tracker: ProgressTracker, feedback: FeedbackStore, broadcaster: _Broadcaster):

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # silence request logs

        def do_GET(self):
            if self.path == "/":
                self._serve_html()
            elif self.path == "/events":
                self._serve_sse()
            elif self.path == "/state":
                self._json_response(broadcaster._build_event(tracker.snapshot()))
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            if self.path == "/feedback/doc":
                feedback.add_doc_feedback(
                    body.get("op_name", ""),
                    body.get("doc_index", 0),
                    body.get("doc_snapshot", {}),
                    body.get("text", ""),
                )
                self._json_response({"ok": True})

            elif self.path == "/feedback/pipeline":
                feedback.add_pipeline_feedback(body.get("text", ""))
                self._json_response({"ok": True})

            elif self.path == "/kill":
                reason = body.get("reason", "")
                feedback.kill_reason = reason
                tracker.kill_requested = True
                self._json_response({"ok": True})

            else:
                self.send_error(404)

        def _json_response(self, data: dict):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_html(self):
            body = _HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = broadcaster.subscribe()
            try:
                while True:
                    try:
                        event = q.get(timeout=30)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                broadcaster.unsubscribe(q)

    return Handler


# ---------------------------------------------------------------------------
# HTML page (embedded)
# ---------------------------------------------------------------------------
_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocETL Monitor</title>
<style>
  :root {
    --background: hsl(211 40% 99%);
    --foreground: hsl(211 5% 0%);
    --card: hsl(211 25% 97%);
    --card-foreground: hsl(211 5% 10%);
    --popover: hsl(211 40% 99%);
    --popover-foreground: hsl(211 100% 0%);
    --primary: hsl(211 100% 50%);
    --primary-foreground: hsl(0 0% 100%);
    --secondary: hsl(211 30% 70%);
    --muted: hsl(173 30% 92%);
    --muted-foreground: hsl(211 5% 35%);
    --accent: hsl(173 30% 90%);
    --accent-foreground: hsl(211 5% 10%);
    --destructive: hsl(0 100% 30%);
    --border: hsl(211 30% 82%);
    --ring: hsl(211 100% 50%);
    --radius: 0.5rem;
    --chart-2: hsl(173 58% 39%);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 14px; background: var(--background); color: var(--foreground);
    display: flex; flex-direction: column; height: 100vh;
  }

  /* Top bar */
  .topbar {
    background: white; border-bottom: 1px solid var(--border);
    padding: 8px 16px; display: flex; align-items: center; gap: 12px;
    box-shadow: 0 1px 3px 0 rgb(0 0 0 / .05); flex-shrink: 0;
  }
  .topbar-title { font-size: 15px; font-weight: 600; color: var(--foreground); }
  .topbar-sep { width: 1px; height: 20px; background: var(--border); }
  .topbar-stat { font-size: 13px; color: var(--muted-foreground); }
  .topbar-stat b { font-weight: 600; color: var(--foreground); }
  .topbar-cost b { color: hsl(152 69% 31%); }
  .topbar-spacer { flex: 1; }
  .topbar-fb { display: flex; gap: 6px; }
  .topbar-fb input {
    width: 280px; background: var(--background); border: 1px solid var(--border);
    border-radius: var(--radius); color: var(--foreground); padding: 5px 10px;
    font-family: inherit; font-size: 13px; transition: border-color .15s, box-shadow .15s;
  }
  .topbar-fb input:focus {
    border-color: var(--primary); outline: none;
    box-shadow: 0 0 0 2px hsl(211 100% 50% / .12);
  }
  .btn {
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: var(--radius); font-family: inherit; font-size: 13px;
    font-weight: 500; cursor: pointer; white-space: nowrap;
    padding: 5px 14px; transition: background .15s, border-color .15s;
    border: 1px solid var(--border); background: white; color: var(--foreground);
  }
  .btn:hover { background: var(--accent); }
  .btn-primary {
    background: var(--primary); color: var(--primary-foreground);
    border-color: var(--primary);
  }
  .btn-primary:hover { background: hsl(211 100% 42%); }
  .btn-destructive {
    border-color: var(--destructive); color: var(--destructive);
  }
  .btn-destructive:hover { background: hsl(0 100% 30% / .06); }

  /* Operations strip */
  .ops-strip {
    background: var(--card); border-bottom: 1px solid var(--border);
    padding: 10px 16px; display: flex; gap: 6px; flex-wrap: wrap;
    align-items: center; flex-shrink: 0;
  }
  .op-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px 4px 8px; border-radius: 999px; font-size: 12px;
    font-weight: 500; border: 1px solid var(--border); background: white;
    position: relative; overflow: hidden;
  }
  .op-chip[data-status="done"] { border-color: hsl(152 69% 55%); background: hsl(152 69% 97%); }
  .op-chip[data-status="running"] { border-color: hsl(211 100% 70%); background: hsl(211 100% 97%); }
  .op-chip[data-status="error"] { border-color: var(--destructive); background: hsl(0 100% 97%); }
  .op-dot {
    width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
  }
  .op-dot.queued { background: var(--border); }
  .op-dot.running { background: var(--primary); animation: pulse 1.5s infinite; }
  .op-dot.done { background: hsl(152 69% 40%); }
  .op-dot.error { background: var(--destructive); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .4; } }
  .op-name { color: var(--foreground); }
  .op-pct { color: var(--muted-foreground); font-size: 11px; }
  .op-cost { color: hsl(152 69% 31%); font-size: 11px; font-weight: 600; }
  .op-progress {
    position: absolute; bottom: 0; left: 0; height: 2px;
    background: var(--primary); transition: width .4s ease;
  }

  /* Main content area */
  .main { flex: 1; overflow: hidden; display: flex; flex-direction: column; }

  /* Tabs */
  .tabs {
    display: flex; gap: 0; padding: 0 16px; background: white;
    border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  .tab {
    padding: 8px 16px; font-size: 13px; font-weight: 500;
    color: var(--muted-foreground); cursor: pointer; border-bottom: 2px solid transparent;
    transition: color .15s, border-color .15s; background: none; border-top: none;
    border-left: none; border-right: none; font-family: inherit;
  }
  .tab:hover { color: var(--foreground); }
  .tab.active { color: var(--foreground); border-bottom-color: var(--primary); }

  /* Table view */
  .table-wrap { flex: 1; overflow: auto; }
  .data-table {
    width: 100%; border-collapse: collapse; font-size: 13px;
    table-layout: fixed;
  }
  .data-table th {
    position: sticky; top: 0; z-index: 2; background: var(--card);
    text-align: left; font-weight: 500; color: var(--muted-foreground);
    border-bottom: 1px solid var(--border); vertical-align: top;
    padding: 0;
  }
  .col-header { padding: 6px 10px; }
  .col-header-name {
    display: flex; align-items: center; gap: 4px; font-size: 12px;
    cursor: pointer; user-select: none;
  }
  .col-header-name:hover { color: var(--foreground); }
  .sort-icon { width: 12px; height: 12px; opacity: .5; flex-shrink: 0; }
  .col-header-name:hover .sort-icon { opacity: .8; }
  .col-stats {
    font-size: 10px; color: var(--muted-foreground); margin-top: 2px;
    display: flex; justify-content: space-between;
  }
  .col-histogram { height: 48px; margin-top: 4px; display: flex; align-items: flex-end; gap: 1px; }
  .hist-bar {
    flex: 1; background: var(--chart-2); border-radius: 2px 2px 0 0;
    min-width: 0; transition: opacity .15s; cursor: default; position: relative;
  }
  .hist-bar:hover { opacity: .75; }

  .data-table td {
    padding: 8px 10px; border-bottom: 1px solid hsl(211 30% 92%);
    vertical-align: top; color: var(--card-foreground); overflow: hidden;
    text-overflow: ellipsis; max-height: 120px;
  }
  .data-table tr { transition: background .1s; }
  .data-table tbody tr:hover { background: hsl(211 40% 97%); }
  .data-table tbody tr.selected { background: hsl(211 60% 95%); }

  .cell-text { white-space: pre-wrap; word-break: break-word; line-height: 1.5; }
  .cell-num { font-variant-numeric: tabular-nums; }

  .col-idx { width: 44px; text-align: center; color: var(--muted-foreground); font-size: 12px; }
  .col-op { width: 120px; }
  .col-fb { width: 44px; text-align: center; }
  .fb-icon {
    width: 28px; height: 28px; border-radius: var(--radius); border: 1px solid transparent;
    background: none; cursor: pointer; display: inline-flex; align-items: center;
    justify-content: center; color: var(--muted-foreground); transition: all .15s;
    font-family: inherit; padding: 0;
  }
  .fb-icon:hover { border-color: var(--border); background: var(--accent); color: var(--primary); }
  .fb-icon.has-fb { color: hsl(152 69% 40%); }

  /* Feedback row */
  .fb-row td { padding: 0 !important; border-bottom: 1px solid var(--border) !important; }
  .fb-inline {
    display: flex; gap: 8px; padding: 8px 10px; background: hsl(211 40% 98%);
    align-items: center;
  }
  .fb-inline input {
    flex: 1; border: 1px solid var(--border); border-radius: var(--radius);
    padding: 5px 10px; font-family: inherit; font-size: 13px; background: white;
    color: var(--foreground); transition: border-color .15s, box-shadow .15s;
  }
  .fb-inline input:focus {
    border-color: var(--primary); outline: none;
    box-shadow: 0 0 0 2px hsl(211 100% 50% / .12);
  }
  .fb-sent-msg { color: hsl(152 69% 31%); font-size: 12px; font-weight: 500; padding: 10px; }

  /* Histogram view */
  .viz-panel { flex: 1; overflow: auto; padding: 16px; }
  .viz-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
  .viz-card {
    background: white; border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px; box-shadow: 0 1px 3px 0 rgb(0 0 0 / .05);
  }
  .viz-card-title { font-size: 13px; font-weight: 600; color: var(--foreground); margin-bottom: 2px; }
  .viz-card-sub { font-size: 11px; color: var(--muted-foreground); margin-bottom: 10px; }
  .viz-chart { height: 100px; display: flex; align-items: flex-end; gap: 2px; }
  .viz-bar {
    flex: 1; background: var(--chart-2); border-radius: 3px 3px 0 0;
    min-width: 0; position: relative; cursor: default; transition: opacity .15s;
  }
  .viz-bar:hover { opacity: .7; }
  .viz-labels { display: flex; justify-content: space-between; margin-top: 4px; }
  .viz-label { font-size: 10px; color: var(--muted-foreground); }

  /* Tooltip */
  .tt {
    position: fixed; z-index: 100; pointer-events: none;
    background: white; border: 1px solid var(--border);
    border-radius: var(--radius); padding: 6px 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,.1); font-size: 12px;
    color: var(--foreground); max-width: 250px; opacity: 0;
    transition: opacity .12s;
  }
  .tt.show { opacity: 1; }
  .tt-label { color: var(--muted-foreground); font-size: 11px; }
  .tt-val { font-weight: 600; }

  /* Status bar */
  .statusbar {
    background: white; border-top: 1px solid var(--border);
    padding: 5px 16px; font-size: 12px; color: var(--muted-foreground);
    display: flex; gap: 16px; flex-shrink: 0; align-items: center;
  }
  .status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    display: inline-block; margin-right: 4px;
  }
  .status-dot.live { background: hsl(152 69% 40%); animation: pulse 1.5s infinite; }
  .status-dot.done { background: hsl(152 69% 40%); }
  .status-dot.off { background: var(--destructive); }

  /* Completion banner */
  .complete-banner {
    background: hsl(152 69% 97%); border: 1px solid hsl(152 69% 75%);
    border-radius: var(--radius); padding: 12px 16px; margin: 12px 16px 0;
    display: flex; align-items: center; gap: 10px; font-size: 13px;
  }
  .complete-banner b { color: hsl(152 69% 28%); }

  .hidden { display: none !important; }
</style>
</head>
<body>

<div class="topbar">
  <span class="topbar-title">DocETL</span>
  <div class="topbar-sep"></div>
  <span class="topbar-stat topbar-cost">Cost: <b id="h-cost">$0</b></span>
  <span class="topbar-stat">Time: <b id="h-time">0s</b></span>
  <span class="topbar-stat">Ops: <b id="h-ops">0/0</b></span>
  <span class="topbar-spacer"></span>
  <div class="topbar-fb">
    <input type="text" id="pipeline-fb-input"
           placeholder="Pipeline feedback…"
           onkeydown="if(event.key==='Enter')sendPipelineFeedback()">
    <button class="btn btn-primary" onclick="sendPipelineFeedback()">Send</button>
  </div>
  <button class="btn btn-destructive" id="kill-btn" onclick="killPipeline()">Stop Pipeline</button>
</div>

<div class="ops-strip" id="ops-strip"></div>

<div class="main">
  <div class="tabs">
    <button class="tab active" data-tab="table" onclick="switchTab('table')">Table</button>
    <button class="tab" data-tab="visualize" onclick="switchTab('visualize')">Visualize</button>
  </div>

  <div id="complete-banner" class="complete-banner hidden">
    <b>Pipeline Complete</b>
    <span id="complete-summary"></span>
  </div>

  <div id="tab-table" class="table-wrap">
    <table class="data-table" id="data-table">
      <thead id="table-head"></thead>
      <tbody id="table-body"></tbody>
    </table>
  </div>

  <div id="tab-visualize" class="viz-panel hidden">
    <div class="viz-grid" id="viz-grid"></div>
  </div>
</div>

<div class="statusbar">
  <span><span class="status-dot live" id="status-dot"></span> <span id="f-status">Connecting…</span></span>
  <span id="f-rows">0 rows</span>
  <span id="f-feedback">Feedback: 0</span>
</div>

<div class="tt" id="tooltip"></div>

<script>
let allDocs = [];
let columns = [];
let columnStats = {};
let sortCol = null;
let sortDir = 'asc';
let expandedRow = -1;
let finished = false;
let killed = false;
let seenDocKeys = new Set();
let currentTab = 'table';

function fmtCost(c) {
  if (c <= 0) return '$0';
  if (c < 0.01) return '<$0.01';
  return '$' + c.toFixed(2);
}
function fmtDur(s) {
  s = Math.round(s);
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60), r = s % 60;
  if (m < 60) return m + 'm ' + r + 's';
  return Math.floor(m / 60) + 'h ' + (m % 60) + 'm';
}
function escHtml(s) {
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}

/* --- Operations strip --- */
function updateOps(ops) {
  const doneCount = ops.filter(o => o.status === 'done').length;
  document.getElementById('h-ops').textContent = doneCount + '/' + ops.length;
  const strip = document.getElementById('ops-strip');
  strip.innerHTML = '';
  ops.forEach(op => {
    const chip = document.createElement('div');
    chip.className = 'op-chip';
    chip.dataset.status = op.status;
    let pctText = '';
    let pctWidth = 0;
    if (op.total) {
      pctWidth = Math.round(100 * op.completed / op.total);
      pctText = pctWidth + '%';
    }
    let inner = '<span class="op-dot ' + op.status + '"></span>';
    inner += '<span class="op-name">' + op.op_type + ':' + op.name.split('/').pop() + '</span>';
    if (pctText) inner += '<span class="op-pct">' + pctText + '</span>';
    if (op.cost > 0) inner += '<span class="op-cost">' + fmtCost(op.cost) + '</span>';
    if (op.status === 'running' && op.total) {
      inner += '<span class="op-progress" style="width:' + pctWidth + '%"></span>';
    }
    chip.innerHTML = inner;
    strip.appendChild(chip);
  });
}

/* --- Column stats (matching playground logic) --- */
function computeColumnStats(colKey) {
  const vals = allDocs.map(d => d.fields[colKey]).filter(v => v != null);
  if (!vals.length) return null;

  const first = vals[0];
  let type = 'string-words';
  if (typeof first === 'number') type = 'number';
  else if (typeof first === 'boolean') type = 'boolean';
  else if (Array.isArray(first)) type = 'array';
  else if (typeof first === 'string') {
    const sample = vals.slice(0, 5).filter(v => typeof v === 'string');
    type = sample.length > 0 && sample.every(v => !/\s/.test(v.trim())) ? 'string-chars' : 'string-words';
  }

  const numeric = vals.map(v => {
    if (typeof v === 'boolean') return v ? 1 : 0;
    if (typeof v === 'number') return v;
    if (Array.isArray(v)) return v.length;
    if (typeof v === 'string') return type === 'string-chars' ? v.trim().length : v.trim().split(/\s+/).length;
    return JSON.stringify(v).split(/\s+/).length;
  });

  const min = Math.min(...numeric);
  const max = Math.max(...numeric);
  const avg = numeric.reduce((a, b) => a + b, 0) / numeric.length;

  const valueCounts = {};
  vals.forEach(v => {
    const key = typeof v === 'object' ? JSON.stringify(v) : String(v);
    valueCounts[key] = (valueCounts[key] || 0) + 1;
  });
  const distinctCount = Object.keys(valueCounts).length;
  const isLowCardinality = distinctCount < vals.length * 0.5;

  const sortedValueCounts = Object.entries(valueCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([value, count]) => ({ value, count }));

  if (type === 'boolean') {
    return { type, min, max, avg, distribution: [numeric.filter(v => v === 0).length, numeric.filter(v => v === 1).length],
             bucketSize: 1, distinctCount, totalCount: vals.length, isLowCardinality: true, sortedValueCounts };
  }
  if (min === max) {
    return { type, min, max, avg, distribution: [numeric.length], bucketSize: 1,
             distinctCount, totalCount: vals.length, isLowCardinality, sortedValueCounts };
  }

  const nBuckets = 7;
  const bucketSize = type === 'number' ? (max - min) / nBuckets : Math.ceil((max - min) / nBuckets);
  const distribution = new Array(nBuckets).fill(0);
  numeric.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / bucketSize), nBuckets - 1);
    distribution[idx]++;
  });

  return { type, min, max, avg, distribution, bucketSize, distinctCount, totalCount: vals.length,
           isLowCardinality, sortedValueCounts };
}

function recomputeStats() {
  columnStats = {};
  columns.forEach(col => { columnStats[col] = computeColumnStats(col); });
}

/* --- Histogram rendering --- */
function unitLabel(type) {
  return { number: '', array: ' items', boolean: '', 'string-chars': ' chars', 'string-words': ' words' }[type] || '';
}

function renderHistogram(stats, container, tall) {
  const h = tall ? 100 : 48;
  const barClass = tall ? 'viz-bar' : 'hist-bar';
  container.innerHTML = '';
  container.style.height = h + 'px';
  if (!stats) return;

  if (stats.isLowCardinality) {
    const items = stats.sortedValueCounts.slice(0, 10);
    const maxC = Math.max(...items.map(d => d.count));
    items.forEach(d => {
      const bar = document.createElement('div');
      bar.className = barClass;
      bar.style.height = Math.max(2, (d.count / maxC) * h) + 'px';
      bar.dataset.ttLabel = d.value;
      bar.dataset.ttVal = d.count + ' (' + ((d.count / stats.totalCount) * 100).toFixed(1) + '%)';
      container.appendChild(bar);
    });
  } else {
    const maxC = Math.max(...stats.distribution);
    const u = unitLabel(stats.type);
    stats.distribution.forEach((count, i) => {
      const bar = document.createElement('div');
      bar.className = barClass;
      bar.style.height = Math.max(2, (count / maxC) * h) + 'px';
      const lo = Math.round(stats.min + i * stats.bucketSize);
      const hi = Math.round(stats.min + (i + 1) * stats.bucketSize);
      bar.dataset.ttLabel = lo + ' – ' + hi + u;
      bar.dataset.ttVal = count + ' (' + ((count / stats.totalCount) * 100).toFixed(1) + '%)';
      container.appendChild(bar);
    });
  }
}

/* --- Tooltip --- */
const tooltip = document.getElementById('tooltip');
document.addEventListener('mouseover', e => {
  const bar = e.target.closest('[data-tt-label]');
  if (bar) {
    tooltip.innerHTML = '<div class="tt-label">' + escHtml(bar.dataset.ttLabel) + '</div><div class="tt-val">' + escHtml(bar.dataset.ttVal) + '</div>';
    tooltip.classList.add('show');
  }
});
document.addEventListener('mousemove', e => {
  if (tooltip.classList.contains('show')) {
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top = (e.clientY - 8) + 'px';
  }
});
document.addEventListener('mouseout', e => {
  if (e.target.closest('[data-tt-label]')) tooltip.classList.remove('show');
});

/* --- Table rendering --- */
function discoverColumns() {
  const seen = new Set();
  const cols = [];
  allDocs.forEach(doc => {
    for (const k of Object.keys(doc.fields)) {
      if (!seen.has(k)) { seen.add(k); cols.push(k); }
    }
  });
  return cols;
}

function getSortedIndices() {
  const indices = allDocs.map((_, i) => i);
  if (!sortCol) return indices;
  indices.sort((a, b) => {
    let va = allDocs[a].fields[sortCol];
    let vb = allDocs[b].fields[sortCol];
    if (va == null) return -1;
    if (vb == null) return 1;
    if (typeof va === 'number' && typeof vb === 'number') return sortDir === 'asc' ? va - vb : vb - va;
    va = String(va); vb = String(vb);
    return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
  });
  return indices;
}

function renderTableHead() {
  const thead = document.getElementById('table-head');
  thead.innerHTML = '';
  const tr = document.createElement('tr');

  // Row number column
  const thIdx = document.createElement('th');
  thIdx.className = 'col-idx';
  thIdx.innerHTML = '<div class="col-header"><span style="font-size:11px">#</span></div>';
  tr.appendChild(thIdx);

  // Operation column
  const thOp = document.createElement('th');
  thOp.className = 'col-op';
  thOp.innerHTML = '<div class="col-header"><span class="col-header-name" style="font-size:12px">operation</span></div>';
  tr.appendChild(thOp);

  // Data columns
  columns.forEach(col => {
    const th = document.createElement('th');
    const stats = columnStats[col];
    const isSorted = sortCol === col;
    const arrow = !isSorted ? '↕' : (sortDir === 'asc' ? '↑' : '↓');

    let statsHtml = '';
    if (stats) {
      if (stats.isLowCardinality) {
        statsHtml = '<div class="col-stats"><span>' + stats.distinctCount + ' distinct</span></div>';
      } else {
        statsHtml = '<div class="col-stats"><span>' + Math.round(stats.min) + '</span><span>avg ' + Math.round(stats.avg) + '</span><span>' + Math.round(stats.max) + '</span></div>';
      }
    }

    th.innerHTML =
      '<div class="col-header">' +
        '<div class="col-header-name" onclick="toggleSort(\'' + escHtml(col) + '\')">' +
          '<span class="sort-icon">' + arrow + '</span>' +
          '<span>' + escHtml(col) + '</span>' +
        '</div>' +
        statsHtml +
        '<div class="col-histogram" id="hist-' + escHtml(col) + '"></div>' +
      '</div>';
    tr.appendChild(th);
  });

  // Feedback column
  const thFb = document.createElement('th');
  thFb.className = 'col-fb';
  thFb.innerHTML = '<div class="col-header"><span style="font-size:11px">fb</span></div>';
  tr.appendChild(thFb);

  thead.appendChild(tr);

  // Render histograms
  columns.forEach(col => {
    const container = document.getElementById('hist-' + col);
    if (container) renderHistogram(columnStats[col], container, false);
  });
}

function renderTableBody() {
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  const sorted = getSortedIndices();

  sorted.forEach(idx => {
    const doc = allDocs[idx];
    const tr = document.createElement('tr');
    if (idx === expandedRow) tr.classList.add('selected');

    // Row number
    const tdIdx = document.createElement('td');
    tdIdx.className = 'col-idx';
    tdIdx.textContent = doc.doc_index + 1;
    tr.appendChild(tdIdx);

    // Operation
    const tdOp = document.createElement('td');
    tdOp.className = 'col-op';
    tdOp.innerHTML = '<span style="font-size:12px;color:var(--muted-foreground)">' + escHtml(doc.op_type + ':' + doc.op_name.split('/').pop()) + '</span>';
    tr.appendChild(tdOp);

    // Data cells
    columns.forEach(col => {
      const td = document.createElement('td');
      const val = doc.fields[col];
      if (val == null) {
        td.innerHTML = '<span style="color:var(--muted-foreground);font-style:italic">—</span>';
      } else if (typeof val === 'number') {
        td.className = 'cell-num';
        td.textContent = val;
      } else {
        td.className = 'cell-text';
        const s = typeof val === 'string' ? val : JSON.stringify(val);
        td.textContent = s.length > 300 ? s.slice(0, 300) + ' …' : s;
      }
      tr.appendChild(td);
    });

    // Feedback button
    const tdFb = document.createElement('td');
    tdFb.className = 'col-fb';
    const fbBtn = document.createElement('button');
    fbBtn.className = 'fb-icon' + (doc._fbSent ? ' has-fb' : '');
    fbBtn.innerHTML = doc._fbSent ? '✓' : '✎';
    fbBtn.title = doc._fbSent ? 'Feedback sent' : 'Give feedback';
    fbBtn.onclick = (e) => { e.stopPropagation(); toggleFeedbackRow(idx); };
    tdFb.appendChild(fbBtn);
    tr.appendChild(tdFb);

    tr.onclick = () => toggleFeedbackRow(idx);
    tbody.appendChild(tr);

    // Feedback expansion row
    if (idx === expandedRow) {
      const fbTr = document.createElement('tr');
      fbTr.className = 'fb-row';
      const fbTd = document.createElement('td');
      fbTd.colSpan = columns.length + 3;
      if (doc._fbSent) {
        fbTd.innerHTML = '<div class="fb-sent-msg">✓ Feedback sent: "' + escHtml(doc._fbText) + '"</div>';
      } else {
        fbTd.innerHTML =
          '<div class="fb-inline">' +
            '<input type="text" placeholder="Feedback on this output…" id="fb-input-' + idx + '" ' +
              'onkeydown="if(event.key===\'Enter\')sendDocFeedback(' + idx + ')">' +
            '<button class="btn" onclick="sendDocFeedback(' + idx + ')">Send</button>' +
          '</div>';
      }
      fbTr.appendChild(fbTd);
      tbody.appendChild(fbTr);
      // Focus input
      setTimeout(() => {
        const inp = document.getElementById('fb-input-' + idx);
        if (inp) inp.focus();
      }, 0);
    }
  });
}

function toggleSort(col) {
  if (sortCol === col) {
    sortDir = sortDir === 'asc' ? 'desc' : 'asc';
  } else {
    sortCol = col;
    sortDir = 'asc';
  }
  renderTableHead();
  renderTableBody();
}

function toggleFeedbackRow(idx) {
  expandedRow = expandedRow === idx ? -1 : idx;
  renderTableBody();
}

/* --- Visualize tab --- */
function renderVizPanel() {
  const grid = document.getElementById('viz-grid');
  grid.innerHTML = '';
  columns.forEach(col => {
    const stats = columnStats[col];
    if (!stats) return;
    const card = document.createElement('div');
    card.className = 'viz-card';
    const u = unitLabel(stats.type);
    let sub = stats.isLowCardinality
      ? stats.distinctCount + ' distinct values · ' + stats.totalCount + ' total'
      : 'min ' + Math.round(stats.min) + u + ' · avg ' + Math.round(stats.avg) + u + ' · max ' + Math.round(stats.max) + u;
    card.innerHTML =
      '<div class="viz-card-title">' + escHtml(col) + '</div>' +
      '<div class="viz-card-sub">' + sub + '</div>' +
      '<div class="viz-chart" id="vizchart-' + escHtml(col) + '"></div>' +
      '<div class="viz-labels" id="vizlabels-' + escHtml(col) + '"></div>';
    grid.appendChild(card);

    const chartEl = document.getElementById('vizchart-' + col);
    renderHistogram(stats, chartEl, true);

    const labelsEl = document.getElementById('vizlabels-' + col);
    if (stats.isLowCardinality) {
      const top = stats.sortedValueCounts.slice(0, 3);
      labelsEl.innerHTML = top.map(d => '<span class="viz-label">' + escHtml(d.value.length > 12 ? d.value.slice(0, 12) + '…' : d.value) + '</span>').join('');
    } else {
      labelsEl.innerHTML = '<span class="viz-label">' + Math.round(stats.min) + u + '</span><span class="viz-label">' + Math.round(stats.max) + u + '</span>';
    }
  });
}

/* --- Tabs --- */
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.getElementById('tab-table').classList.toggle('hidden', tab !== 'table');
  document.getElementById('tab-visualize').classList.toggle('hidden', tab !== 'visualize');
  if (tab === 'visualize') renderVizPanel();
}

/* --- Data sync --- */
function syncDocs(docs) {
  let changed = false;
  docs.forEach(doc => {
    const key = doc.op_name + ':' + doc.doc_index;
    if (!seenDocKeys.has(key)) {
      seenDocKeys.add(key);
      allDocs.push(doc);
      changed = true;
    }
  });
  if (!changed) return;

  const newCols = discoverColumns();
  const colsChanged = newCols.length !== columns.length || newCols.some((c, i) => c !== columns[i]);
  columns = newCols;
  recomputeStats();

  renderTableHead();
  renderTableBody();
  if (currentTab === 'visualize') renderVizPanel();

  document.getElementById('f-rows').textContent = allDocs.length + ' row' + (allDocs.length === 1 ? '' : 's');

  // Auto-scroll to bottom
  const tw = document.getElementById('tab-table');
  tw.scrollTop = tw.scrollHeight;
}

/* --- Feedback --- */
function sendDocFeedback(idx) {
  const input = document.getElementById('fb-input-' + idx);
  const text = input.value.trim();
  if (!text) return;
  const doc = allDocs[idx];
  fetch('/feedback/doc', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ op_name: doc.op_name, doc_index: doc.doc_index, doc_snapshot: doc.fields, text: text })
  });
  doc._fbSent = true;
  doc._fbText = text;
  renderTableBody();
}

function sendPipelineFeedback() {
  const input = document.getElementById('pipeline-fb-input');
  const text = input.value.trim();
  if (!text) return;
  fetch('/feedback/pipeline', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ text: text })
  });
  input.value = '';
  input.placeholder = 'Sent! Type more…';
}

function killPipeline() {
  const reason = prompt('Reason for stopping (optional):') || '';
  if (killed) return;
  killed = true;
  fetch('/kill', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ reason: reason })
  });
  const btn = document.getElementById('kill-btn');
  btn.textContent = 'Stopping…';
  btn.disabled = true;
  btn.style.opacity = '0.5';
}

/* --- SSE --- */
const evtSource = new EventSource('/events');
evtSource.onmessage = function(e) {
  const data = JSON.parse(e.data);
  updateOps(data.ops);
  syncDocs(data.all_docs);
  document.getElementById('h-cost').textContent = fmtCost(data.total_cost);
  document.getElementById('h-time').textContent = fmtDur(data.elapsed);
  document.getElementById('f-feedback').textContent = 'Feedback: ' + data.feedback_count;
  document.getElementById('f-status').textContent = data.finished ? 'Complete' : 'Running';

  if (data.finished && !finished) {
    finished = true;
    const dot = document.getElementById('status-dot');
    dot.classList.remove('live');
    dot.classList.add('done');
    const banner = document.getElementById('complete-banner');
    banner.classList.remove('hidden');
    document.getElementById('complete-summary').textContent = fmtCost(data.total_cost) + ' · ' + fmtDur(data.elapsed) + ' · ' + allDocs.length + ' outputs';
    document.getElementById('kill-btn').classList.add('hidden');
  }
};
evtSource.onerror = function() {
  document.getElementById('f-status').textContent = 'Disconnected';
  const dot = document.getElementById('status-dot');
  dot.classList.remove('live');
  dot.classList.add('off');
};
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_with_web_ui(runner: "DSLRunner") -> float:
    """Run the pipeline with a web-based progress + feedback UI.

    Starts an HTTP server on a free port, prints the URL, runs the pipeline,
    and writes any collected feedback to ``_docetl_feedback.json`` in the
    working directory (and to stdout).
    """
    import threading

    from tqdm import tqdm as _tqdm

    _tqdm.set_lock(threading.RLock())

    tracker = ProgressTracker(concurrency=min(runner.max_threads or 1, 64))
    runner.progress_tracker = tracker
    runner._tui_active = True
    set_active_tracker(tracker)

    tracker.pipeline_start(runner.list_pipeline_operations())

    feedback = FeedbackStore()
    broadcaster = _Broadcaster(tracker, feedback)
    broadcaster.start()

    handler_cls = _make_handler(tracker, feedback, broadcaster)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://localhost:{port}"
    runner.console.log(f"[bold blue]Live monitor:[/bold blue] [link={url}]{url}[/link]")
    runner.console.log(
        "[dim]Open in a browser to watch outputs and give feedback. "
        "The pipeline is running…[/dim]"
    )

    cost = 0.0
    killed = False
    try:
        cost = runner.load_run_save()
    except PipelineKilled:
        killed = True
        runner.console.log("[bold red]Pipeline killed by user.[/bold red]")
        try:
            tracker.pipeline_done()
        except Exception:
            pass
    finally:
        broadcaster.stop()
        set_active_tracker(None)
        runner.progress_tracker = None
        runner._tui_active = False

    # Write feedback
    if feedback.has_any:
        fb_data = feedback.to_dict()
        fb_path = os.path.join(os.getcwd(), "_docetl_feedback.json")
        with open(fb_path, "w") as f:
            json.dump(fb_data, f, indent=2)
        runner.console.log(f"\n[bold]Human feedback collected:[/bold]")
        if feedback.pipeline_feedback:
            for item in feedback.pipeline_feedback:
                runner.console.log(f"  [pipeline] {item['feedback']}")
        if feedback.doc_feedback:
            for item in feedback.doc_feedback:
                runner.console.log(
                    f"  [doc] {item['operation']} #{item['doc_index']}: {item['feedback']}"
                )
        if killed and feedback.kill_reason:
            runner.console.log(f"  [kill reason] {feedback.kill_reason}")
        runner.console.log(f"[dim]Full feedback written to {fb_path}[/dim]")

    # Keep the server alive briefly so the browser can show the final state.
    time.sleep(2)
    server.shutdown()

    if killed:
        cost = runner.total_cost

    return cost
