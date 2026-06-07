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
        new_docs = []
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
            already = self._sent_docs.get(op.name, 0)
            for i, doc in enumerate(op.outputs[already:], start=already):
                new_docs.append({
                    "op_name": op.name,
                    "op_type": op.op_type,
                    "doc_index": i,
                    "fields": {k: _trunc(v) for k, v in doc.items() if not k.startswith("_")},
                })
            self._sent_docs[op.name] = len(op.outputs)

        return {
            "ops": ops,
            "new_docs": new_docs,
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
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 13px;
         background: #0d1117; color: #c9d1d9; display: flex; flex-direction: column; height: 100vh; }

  header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 20px;
           display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  header h1 { font-size: 15px; color: #58a6ff; font-weight: 600; white-space: nowrap; }
  .stats { color: #8b949e; font-size: 12px; display: flex; gap: 16px; }
  .stats .cost { color: #3fb950; }

  .pipeline-fb { display: flex; gap: 8px; flex: 1; min-width: 300px; }
  .pipeline-fb input { flex: 1; background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
                        color: #c9d1d9; padding: 6px 10px; font-family: inherit; font-size: 12px; }
  .pipeline-fb input:focus { border-color: #58a6ff; outline: none; }

  .btn { background: #21262d; border: 1px solid #30363d; border-radius: 6px; color: #c9d1d9;
         padding: 6px 14px; cursor: pointer; font-family: inherit; font-size: 12px; white-space: nowrap; }
  .btn:hover { background: #30363d; }
  .btn.danger { border-color: #f85149; color: #f85149; }
  .btn.danger:hover { background: #f8514922; }
  .btn.primary { border-color: #58a6ff; color: #58a6ff; }

  .main { display: flex; flex: 1; overflow: hidden; }

  .sidebar { width: 260px; min-width: 200px; background: #161b22; border-right: 1px solid #30363d;
             padding: 12px; overflow-y: auto; }
  .sidebar h2 { font-size: 11px; text-transform: uppercase; color: #8b949e; margin-bottom: 8px; letter-spacing: 0.5px; }
  .op { padding: 8px 10px; border-radius: 6px; margin-bottom: 4px; }
  .op:hover { background: #1c2128; }
  .op .name { font-weight: 600; color: #c9d1d9; }
  .op .meta { font-size: 11px; color: #8b949e; margin-top: 2px; }
  .op .meta .cost { color: #3fb950; }
  .op.running { border-left: 3px solid #d29922; }
  .op.done { border-left: 3px solid #3fb950; }
  .op.error { border-left: 3px solid #f85149; }
  .op.queued { border-left: 3px solid #30363d; }

  .content { flex: 1; overflow-y: auto; padding: 16px 20px; }
  .content h2 { font-size: 11px; text-transform: uppercase; color: #8b949e; margin-bottom: 12px; letter-spacing: 0.5px; }

  .doc-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
              margin-bottom: 12px; overflow: hidden; }
  .doc-header { padding: 10px 14px; background: #1c2128; border-bottom: 1px solid #30363d;
                display: flex; justify-content: space-between; align-items: center; }
  .doc-header .label { font-weight: 600; color: #58a6ff; font-size: 12px; }
  .doc-header .op-tag { font-size: 11px; color: #8b949e; background: #21262d;
                        padding: 2px 8px; border-radius: 10px; }
  .doc-fields { padding: 12px 14px; }
  .field { margin-bottom: 6px; }
  .field .key { color: #79c0ff; font-weight: 600; }
  .field .val { color: #c9d1d9; white-space: pre-wrap; word-break: break-word; }
  .doc-fb { padding: 10px 14px; border-top: 1px solid #30363d; display: flex; gap: 8px; }
  .doc-fb input { flex: 1; background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
                  color: #c9d1d9; padding: 5px 10px; font-family: inherit; font-size: 12px; }
  .doc-fb input:focus { border-color: #58a6ff; outline: none; }
  .fb-sent { color: #3fb950; font-size: 11px; padding: 6px 0; }

  .banner { background: #1c2128; border: 1px solid #30363d; border-radius: 8px;
            padding: 16px 20px; margin-bottom: 16px; text-align: center; }
  .banner.done { border-color: #3fb950; }
  .banner.killed { border-color: #f85149; }
  .banner .big { font-size: 18px; font-weight: 600; margin-bottom: 4px; }

  footer { background: #161b22; border-top: 1px solid #30363d; padding: 8px 20px;
           font-size: 11px; color: #8b949e; display: flex; gap: 20px; }
</style>
</head>
<body>

<header>
  <h1>DocETL Monitor</h1>
  <div class="stats">
    <span>Cost: <span class="cost" id="h-cost">$0</span></span>
    <span>Time: <span id="h-time">0s</span></span>
    <span>Ops: <span id="h-ops">0/0</span></span>
  </div>
  <div class="pipeline-fb">
    <input type="text" id="pipeline-fb-input" placeholder="Pipeline-level feedback (e.g. 'prompts are too aggressive')">
    <button class="btn primary" onclick="sendPipelineFeedback()">Send</button>
  </div>
  <button class="btn danger" id="kill-btn" onclick="killPipeline()">Kill Pipeline</button>
</header>

<div class="main">
  <div class="sidebar">
    <h2>Operations</h2>
    <div id="ops-list"></div>
  </div>
  <div class="content" id="content">
    <h2>Document Outputs</h2>
    <div id="docs-list"></div>
  </div>
</div>

<footer>
  <span id="f-status">Connecting…</span>
  <span id="f-feedback">Feedback: 0</span>
</footer>

<script>
const opsList = document.getElementById('ops-list');
const docsList = document.getElementById('docs-list');
let allDocs = [];
let finished = false;
let killed = false;

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
  const h = Math.floor(m / 60);
  return h + 'h ' + (m % 60) + 'm';
}

function updateOps(ops) {
  const doneCount = ops.filter(o => o.status === 'done').length;
  document.getElementById('h-ops').textContent = doneCount + '/' + ops.length;

  opsList.innerHTML = '';
  ops.forEach(op => {
    const el = document.createElement('div');
    el.className = 'op ' + op.status;
    const glyph = {done:'✓', running:'◐', error:'✗', queued:'○'}[op.status] || '○';
    let meta = '';
    if (op.total) {
      const pct = Math.round(100 * op.completed / op.total);
      meta += op.completed + '/' + op.total + ' (' + pct + '%)';
    }
    if (op.cost > 0) meta += (meta ? '  ' : '') + '<span class="cost">' + fmtCost(op.cost) + '</span>';
    if (op.elapsed >= 1) meta += (meta ? '  ' : '') + fmtDur(op.elapsed);
    if (op.errors) meta += '  <span style="color:#f85149">!' + op.errors + '</span>';
    el.innerHTML = '<div class="name">' + glyph + ' ' + op.op_type + ':' + op.name.split('/').pop() + '</div>' +
                   (meta ? '<div class="meta">' + meta + '</div>' : '');
    opsList.appendChild(el);
  });
}

function addDocs(docs) {
  docs.forEach(doc => {
    allDocs.push(doc);
    const idx = allDocs.length - 1;
    const card = document.createElement('div');
    card.className = 'doc-card';
    card.id = 'doc-' + idx;

    let fieldsHtml = '';
    for (const [k, v] of Object.entries(doc.fields)) {
      const val = typeof v === 'string' ? v : JSON.stringify(v);
      fieldsHtml += '<div class="field"><span class="key">' + escHtml(k) + ': </span><span class="val">' + escHtml(val) + '</span></div>';
    }

    card.innerHTML =
      '<div class="doc-header">' +
        '<span class="label">Document #' + (doc.doc_index + 1) + '</span>' +
        '<span class="op-tag">' + doc.op_type + ':' + doc.op_name.split('/').pop() + '</span>' +
      '</div>' +
      '<div class="doc-fields">' + fieldsHtml + '</div>' +
      '<div class="doc-fb" id="fb-row-' + idx + '">' +
        '<input type="text" placeholder="Feedback on this output…" id="fb-input-' + idx + '" onkeydown="if(event.key===\'Enter\')sendDocFeedback(' + idx + ')">' +
        '<button class="btn" onclick="sendDocFeedback(' + idx + ')">Send</button>' +
      '</div>';
    docsList.appendChild(card);
  });
  // Auto-scroll to bottom
  const content = document.getElementById('content');
  content.scrollTop = content.scrollHeight;
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function sendDocFeedback(idx) {
  const input = document.getElementById('fb-input-' + idx);
  const text = input.value.trim();
  if (!text) return;
  const doc = allDocs[idx];
  fetch('/feedback/doc', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({op_name: doc.op_name, doc_index: doc.doc_index, doc_snapshot: doc.fields, text: text})
  });
  const row = document.getElementById('fb-row-' + idx);
  row.innerHTML = '<span class="fb-sent">✓ Feedback sent: "' + escHtml(text) + '"</span>';
}

function sendPipelineFeedback() {
  const input = document.getElementById('pipeline-fb-input');
  const text = input.value.trim();
  if (!text) return;
  fetch('/feedback/pipeline', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: text})
  });
  input.value = '';
  input.placeholder = '✓ Sent! Type more feedback…';
}

function killPipeline() {
  const reason = prompt('Reason for killing the pipeline (optional):') || '';
  if (killed) return;
  killed = true;
  fetch('/kill', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({reason: reason})
  });
  document.getElementById('kill-btn').textContent = 'Killing…';
  document.getElementById('kill-btn').disabled = true;
}

// SSE connection
const evtSource = new EventSource('/events');
evtSource.onmessage = function(e) {
  const data = JSON.parse(e.data);
  updateOps(data.ops);
  if (data.new_docs.length) addDocs(data.new_docs);
  document.getElementById('h-cost').textContent = fmtCost(data.total_cost);
  document.getElementById('h-time').textContent = fmtDur(data.elapsed);
  document.getElementById('f-feedback').textContent = 'Feedback: ' + data.feedback_count;
  document.getElementById('f-status').textContent = data.finished ? 'Pipeline complete' : 'Running';

  if (data.finished && !finished) {
    finished = true;
    const banner = document.createElement('div');
    banner.className = 'banner done';
    banner.innerHTML = '<div class="big" style="color:#3fb950">✓ Pipeline Complete</div>' +
                       '<div>' + fmtCost(data.total_cost) + '  ' + fmtDur(data.elapsed) + '</div>';
    docsList.prepend(banner);
    document.getElementById('kill-btn').style.display = 'none';
  }
};
evtSource.onerror = function() {
  document.getElementById('f-status').textContent = 'Disconnected';
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
