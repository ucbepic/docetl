"""Web-based progress + feedback UI for non-interactive environments.

Supports two modes:

1. **Inline** (default) — the pipeline process starts its own HTTP server,
   runs the pipeline, and waits for human feedback before exiting.

2. **Persistent server** — a long-lived ``docetl serve`` process runs
   independently.  Each ``docetl run`` pushes state updates to it via HTTP.
   The agent polls ``GET /feedback/poll`` for new feedback at any time.
   The server survives across multiple pipeline runs.

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
from urllib.error import URLError
from urllib.request import Request, urlopen

from docetl.progress.tracker import PipelineKilled, ProgressTracker, set_active_tracker

if TYPE_CHECKING:
    from docetl.runner import DSLRunner

_PORT_FILE = ".docetl_server_port"
_FEEDBACK_LOG = ".docetl_feedback.log"


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
        self._agent_messages: list[dict] = []
        self._agent_msg_counter = 0
        # Truncate log at start of each session
        with open(_FEEDBACK_LOG, "w") as f:
            pass

    def _log(self, line: str):
        try:
            with open(_FEEDBACK_LOG, "a") as f:
                f.write(line + "\n")
                f.flush()
        except OSError:
            pass

    def add_agent_message(self, text: str, msg_type: str = "info",
                          actions: list[str] | None = None) -> int:
        with self._lock:
            self._agent_msg_counter += 1
            msg = {"id": self._agent_msg_counter, "text": text, "type": msg_type,
                   "actions": actions or [],
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self._agent_messages.append(msg)
            return self._agent_msg_counter

    def respond_to_message(self, msg_id: int, action: str):
        with self._lock:
            for m in self._agent_messages:
                if m["id"] == msg_id:
                    m["response"] = action
                    m["actions"] = []
                    break
        print(f"[TOAST:response] id={msg_id} action={action}", flush=True)

    def get_agent_messages_since(self, since_id: int) -> list[dict]:
        with self._lock:
            return [m for m in self._agent_messages if m["id"] > since_id]

    def add_doc_feedback(self, op_name: str, doc_index: int, doc_snapshot: dict, text: str):
        entry = {
            "operation": op_name,
            "doc_index": doc_index,
            "doc_snapshot": {k: v for k, v in doc_snapshot.items() if not k.startswith("_")},
            "feedback": text,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self._lock:
            self.doc_feedback.append(entry)
        line = f"[FEEDBACK:doc] op={op_name} doc_index={doc_index} | {text}"
        print(line, flush=True)
        self._log(line)

    def remove_doc_feedback(self, op_name: str, doc_index: int, feedback_index: int):
        with self._lock:
            matches = [i for i, f in enumerate(self.doc_feedback)
                       if f["operation"] == op_name and f["doc_index"] == doc_index]
            if 0 <= feedback_index < len(matches):
                del self.doc_feedback[matches[feedback_index]]
        line = f"[FEEDBACK:delete] op={op_name} doc_index={doc_index} fb_index={feedback_index}"
        print(line, flush=True)
        self._log(line)

    def add_pipeline_feedback(self, text: str):
        entry = {
            "feedback": text,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self._lock:
            self.pipeline_feedback.append(entry)
        line = f"[FEEDBACK:pipeline] {text}"
        print(line, flush=True)
        self._log(line)

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
    """Pushes state snapshots to SSE subscribers.

    Two modes:
    * **local** — ``tracker`` is set; a background thread polls it every second.
    * **remote** — ``tracker`` is ``None``; the pipeline pushes state via
      ``accept_state()``.
    """

    def __init__(self, tracker: ProgressTracker | None, feedback: FeedbackStore):
        self._tracker = tracker
        self._feedback = feedback
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sent_docs: dict[str, int] = {}
        self._reset_counter = 0
        self._last_event: dict = {}

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
        if self._tracker is None:
            return  # remote mode — no polling thread needed
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

    def accept_state(self, event: dict):
        """Accept a state dict pushed by a remote pipeline process."""
        event["feedback_count"] = len(self._feedback.doc_feedback) + len(self._feedback.pipeline_feedback)
        event["doc_feedback"] = [
            {"op_name": f["operation"], "doc_index": f["doc_index"], "feedback": f["feedback"]}
            for f in self._feedback.doc_feedback
        ]
        event["agent_messages"] = self._feedback.get_agent_messages_since(0)
        event["reset_token"] = self._reset_counter
        self._last_event = event
        self._broadcast(event)

    def rebroadcast(self):
        """Re-broadcast the last known state with fresh feedback/messages.

        In remote mode, the pipeline's push loop drives SSE updates. Once the
        pipeline finishes (or between pushes), new feedback and toasts are stored
        but never delivered to SSE subscribers. Call this after any mutation
        (add feedback, post toast, etc.) so the browser sees the change immediately.
        """
        if not self._last_event:
            return
        event = dict(self._last_event)
        event["feedback_count"] = len(self._feedback.doc_feedback) + len(self._feedback.pipeline_feedback)
        event["doc_feedback"] = [
            {"op_name": f["operation"], "doc_index": f["doc_index"], "feedback": f["feedback"]}
            for f in self._feedback.doc_feedback
        ]
        event["agent_messages"] = self._feedback.get_agent_messages_since(0)
        event["reset_token"] = self._reset_counter
        self._broadcast(event)

    def _loop(self):
        while not self._stop.is_set():
            self._push()
            self._stop.wait(0.5)
        self._push()  # final push

    def _push(self):
        state = self._tracker.snapshot()
        event = self._build_event(state)
        self._last_event = event
        self._broadcast(event)

    def _broadcast(self, event: dict):
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

        all_docs = []
        for op in state.ops:
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
            "doc_feedback": [{"op_name": f["operation"], "doc_index": f["doc_index"], "feedback": f["feedback"]}
                             for f in self._feedback.doc_feedback],
            "agent_messages": self._feedback.get_agent_messages_since(0),
            "reset_token": self._reset_counter,
        }


def _trunc(v, n=500) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + " …"


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
def _make_handler(tracker: ProgressTracker | None, feedback: FeedbackStore, broadcaster: _Broadcaster):

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # silence request logs

        def do_GET(self):
            if self.path == "/":
                self._serve_html()
            elif self.path == "/events":
                self._serve_sse()
            elif self.path == "/state":
                if tracker is not None:
                    self._json_response(broadcaster._build_event(tracker.snapshot()))
                else:
                    self._json_response(broadcaster._last_event or {})
            elif self.path.startswith("/messages"):
                since = 0
                if "?since=" in self.path:
                    try:
                        since = int(self.path.split("?since=")[1])
                    except ValueError:
                        pass
                self._json_response({"messages": feedback.get_agent_messages_since(since)})
            elif self.path.startswith("/feedback/poll"):
                self._json_response(feedback.to_dict())
            elif self.path == "/health":
                self._json_response({"ok": True})
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
                broadcaster.rebroadcast()
                self._json_response({"ok": True})

            elif self.path == "/feedback/doc/delete":
                feedback.remove_doc_feedback(
                    body.get("op_name", ""),
                    body.get("doc_index", 0),
                    body.get("feedback_index", 0),
                )
                broadcaster.rebroadcast()
                self._json_response({"ok": True})

            elif self.path == "/feedback/pipeline":
                feedback.add_pipeline_feedback(body.get("text", ""))
                broadcaster.rebroadcast()
                self._json_response({"ok": True})

            elif self.path == "/kill":
                reason = body.get("reason", "")
                feedback.kill_reason = reason
                feedback._log(f"[FEEDBACK:kill] {reason}")
                if tracker is not None:
                    tracker.kill_requested = True
                broadcaster.rebroadcast()
                self._json_response({"ok": True})

            elif self.path == "/message":
                msg_id = feedback.add_agent_message(
                    body.get("text", ""),
                    body.get("type", "info"),
                    body.get("actions"),
                )
                broadcaster.rebroadcast()
                self._json_response({"ok": True, "id": msg_id})

            elif self.path == "/message/respond":
                feedback.respond_to_message(
                    body.get("id", 0),
                    body.get("action", ""),
                )
                broadcaster.rebroadcast()
                self._json_response({"ok": True})

            elif self.path == "/state/push":
                broadcaster.accept_state(body)
                self._json_response({"ok": True})

            elif self.path == "/reset":
                if tracker is not None:
                    for op in tracker.snapshot().ops:
                        op.outputs.clear()
                    tracker._finished = False
                with feedback._lock:
                    feedback.doc_feedback.clear()
                    feedback.pipeline_feedback.clear()
                    feedback.kill_reason = None
                with open(_FEEDBACK_LOG, "w"):
                    pass
                broadcaster._reset_counter += 1
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
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            q = broadcaster.subscribe()
            # Send the last known state immediately so the browser catches up
            # instead of waiting up to 1s for the next push.
            if broadcaster._last_event:
                self.wfile.write(f"data: {json.dumps(broadcaster._last_event)}\n\n".encode())
                self.wfile.flush()
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
    background: white;
    padding: 8px 16px; display: flex; align-items: center; gap: 12px;
    box-shadow: 0 1px 3px 0 rgb(0 0 0 / .06); flex-shrink: 0;
  }
  .topbar-title { font-size: 15px; font-weight: 600; color: var(--foreground); }
  .topbar-sep { width: 1px; height: 20px; background: hsl(211 20% 90%); }
  .topbar-stat { font-size: 13px; color: var(--muted-foreground); }
  .topbar-stat b { font-weight: 600; color: var(--foreground); }
  .topbar-cost b { color: hsl(152 69% 31%); }
  .topbar-spacer { flex: 1; }
  .topbar-fb { display: flex; gap: 6px; }
  .topbar-fb input {
    width: 280px; background: var(--background); border: none;
    border-radius: var(--radius); color: var(--foreground); padding: 5px 10px;
    font-family: inherit; font-size: 13px; transition: box-shadow .15s;
  }
  .topbar-fb input:focus {
    outline: none;
    box-shadow: 0 0 0 2px hsl(211 100% 50% / .15);
  }
  .btn {
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: var(--radius); font-family: inherit; font-size: 13px;
    font-weight: 500; cursor: pointer; white-space: nowrap;
    padding: 5px 14px; transition: background .15s, border-color .15s;
    border: none; background: var(--card); color: var(--foreground);
  }
  .btn:hover { background: var(--accent); }
  .btn-primary {
    background: var(--primary); color: var(--primary-foreground);
  }
  .btn-primary:hover { background: hsl(211 100% 42%); }
  .btn-destructive {
    color: var(--destructive); background: hsl(0 100% 30% / .06);
  }
  .btn-destructive:hover { background: hsl(0 100% 30% / .1); }

  /* Operations strip */
  .ops-strip {
    background: white;
    padding: 0 16px; display: flex; gap: 0; align-items: stretch; flex-shrink: 0;
    overflow-x: auto; box-shadow: 0 1px 2px 0 rgb(0 0 0 / .04);
  }
  .op-item {
    display: flex; align-items: center; gap: 6px;
    padding: 8px 14px; font-size: 12px; font-weight: 500;
    position: relative; white-space: nowrap; cursor: pointer;
    border-radius: 6px; transition: background .15s;
  }
  .op-item:hover { background: hsl(211 40% 95%); }
  .op-item.op-selected { background: hsl(211 50% 93%); }
  .op-dot {
    width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
  }
  .op-dot.queued { background: hsl(211 20% 78%); }
  .op-dot.running { background: var(--primary); animation: pulse 1.5s infinite; }
  .op-dot.done { background: hsl(152 69% 40%); }
  .op-dot.error { background: var(--destructive); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .4; } }
  .op-name { color: var(--foreground); }
  .op-detail { color: var(--muted-foreground); font-size: 11px; }
  .op-cost { color: hsl(152 69% 31%); font-size: 11px; font-weight: 600; }
  .op-arrow {
    color: var(--foreground); font-size: 18px; display: flex; align-items: center;
    padding: 0 6px;
  }
  .op-bar {
    position: absolute; bottom: 0; left: 0; height: 2px;
    background: var(--primary); transition: width .6s ease;
    border-radius: 0 1px 1px 0;
  }

  /* Main content area */
  .main { flex: 1; overflow: hidden; display: flex; flex-direction: column; }

  /* Tabs */
  .tabs {
    display: flex; gap: 0; padding: 0 16px; background: white;
    flex-shrink: 0;
  }
  .tab {
    padding: 8px 16px; font-size: 13px; font-weight: 500;
    color: var(--muted-foreground); cursor: pointer; border: none;
    border-bottom: 2px solid transparent;
    transition: color .15s, border-color .15s; background: none; font-family: inherit;
  }
  .tab:hover { color: var(--foreground); }
  .tab.active { color: var(--foreground); border-bottom-color: var(--primary); }

  /* Table view */
  .table-wrap {
    flex: 1; overflow: auto; position: relative;
    background:
      linear-gradient(white 30%, transparent),
      linear-gradient(transparent, white 70%) 0 100%,
      radial-gradient(farthest-side at 50% 0, rgba(0,0,0,.08), transparent),
      radial-gradient(farthest-side at 50% 100%, rgba(0,0,0,.08), transparent) 0 100%;
    background-repeat: no-repeat;
    background-size: 100% 30px, 100% 30px, 100% 8px, 100% 8px;
    background-attachment: local, local, scroll, scroll;
  }
  .data-table {
    width: 100%; border-collapse: collapse; font-size: 13px;
    table-layout: fixed;
  }
  .data-table th {
    position: sticky; top: 0; z-index: 2; background: var(--card);
    text-align: left; font-weight: 500; color: var(--muted-foreground);
    vertical-align: top; padding: 0; overflow: hidden;
  }
  .col-resize {
    position: absolute; right: 0; top: 0; bottom: 0; width: 5px;
    cursor: col-resize; z-index: 3; user-select: none;
  }
  .col-resize:hover, .col-resize.active { background: var(--primary); opacity: .4; }
  .col-header { padding: 6px 10px; }
  .col-header-name {
    display: flex; align-items: center; gap: 4px; font-size: 12px;
    cursor: pointer; user-select: none;
  }
  .col-header-name:hover { color: var(--foreground); }
  .sort-icon { opacity: .4; flex-shrink: 0; font-size: 10px; }
  .col-header-name:hover .sort-icon { opacity: .7; }
  .col-stats {
    font-size: 10px; color: var(--muted-foreground); margin-top: 2px;
    display: flex; justify-content: space-between;
  }
  .col-histogram { margin-top: 4px; }
  .col-histogram svg { display: block; width: 100%; }

  .data-table td {
    padding: 8px 10px;
    vertical-align: top; color: var(--card-foreground); overflow: hidden;
    text-overflow: ellipsis; max-height: 120px;
  }
  .data-table tr { transition: background .1s; cursor: pointer; }
  .data-table tbody tr:hover { background: hsl(211 40% 97%); }

  .cell-text { white-space: pre-wrap; word-break: break-word; line-height: 1.5; }
  .cell-num { font-variant-numeric: tabular-nums; }

  .col-idx { width: 54px; text-align: center; color: var(--muted-foreground); font-size: 12px; }

  /* Feedback dot indicator */
  .fb-dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: hsl(152 69% 40%); margin-right: 4px; vertical-align: middle;
    flex-shrink: 0;
  }
  .idx-cell-inner { display: flex; align-items: center; justify-content: center; gap: 2px; }

  /* Column filter input */
  .col-filter-wrap {
    margin-top: 4px; display: flex; align-items: center; gap: 2px;
  }
  .col-filter {
    width: 100%; border: none; border-radius: 3px;
    padding: 3px 6px; font-family: inherit; font-size: 11px;
    background: hsl(211 30% 96%); color: var(--foreground);
    transition: background .15s, box-shadow .15s;
  }
  .col-filter:focus {
    outline: none; background: white;
    box-shadow: 0 0 0 2px hsl(211 100% 50% / .12);
  }
  .col-filter-clear {
    background: none; border: none; cursor: pointer; font-size: 13px;
    color: var(--muted-foreground); padding: 0 2px; line-height: 1;
    flex-shrink: 0; display: none;
  }
  .col-filter-clear.active { display: inline-block; }
  .col-filter-clear:hover { color: var(--foreground); }
  .filter-toggle {
    background: none; border: none; cursor: pointer; font-size: 11px;
    color: var(--muted-foreground); padding: 0; margin-left: 4px;
    opacity: .5; transition: opacity .15s;
  }
  .filter-toggle:hover { opacity: 1; }
  .filter-toggle.active { opacity: 1; color: var(--primary); }

  /* Row selected highlight */
  .row-selected, .data-table tbody tr.row-selected:hover {
    background: hsl(211 60% 95%);
  }

  /* Detail side panel */
  .detail-panel {
    position: fixed; top: 0; right: 0; bottom: 0; width: 400px;
    background: white; z-index: 40;
    box-shadow: -4px 0 20px rgba(0,0,0,.1);
    transform: translateX(100%);
    transition: transform .25s ease;
    display: flex; flex-direction: column;
    overflow: hidden;
  }
  .detail-panel.open { transform: translateX(0); }
  .detail-header {
    display: flex; align-items: center; gap: 8px;
    padding: 14px 16px; flex-shrink: 0;
  }
  .detail-header-title { font-size: 15px; font-weight: 600; flex: 1; display: flex; align-items: center; gap: 6px; }
  .detail-row-of { font-size: 12px; color: var(--muted-foreground); font-weight: 400; }
  .detail-hotkey-hint { font-size: 10px; color: var(--muted-foreground); opacity: .6; margin-left: 4px; pointer-events: none; }
  .detail-nav-btn {
    background: var(--card); border: none; border-radius: var(--radius);
    cursor: pointer; padding: 4px 10px; font-size: 14px;
    color: var(--muted-foreground); transition: background .15s;
  }
  .detail-nav-btn:hover { background: var(--accent); color: var(--foreground); }
  .detail-nav-btn:disabled { opacity: .3; cursor: default; }
  .detail-close {
    background: none; border: none; cursor: pointer;
    font-size: 20px; line-height: 1; color: var(--muted-foreground);
    padding: 0 4px; font-family: inherit;
  }
  .detail-close:hover { color: var(--foreground); }
  .detail-body {
    flex: 1; overflow-y: auto; padding: 16px;
  }
  .detail-field { margin-bottom: 14px; }
  .detail-field-key {
    font-size: 11px; font-weight: 500; color: var(--muted-foreground);
    margin-bottom: 3px; text-transform: uppercase; letter-spacing: .03em;
  }
  .detail-field-val {
    font-size: 13px; color: var(--foreground); line-height: 1.5;
    white-space: pre-wrap; word-break: break-word;
  }
  .detail-op-tag {
    display: inline-block; font-size: 11px; color: var(--muted-foreground);
    background: var(--card); border-radius: 3px; padding: 2px 8px;
    margin-bottom: 12px;
  }
  .detail-fb {
    padding: 0 16px 10px; flex-shrink: 0;
  }
  .detail-fb-label {
    font-size: 11px; font-weight: 500; color: var(--muted-foreground);
    margin-bottom: 6px; text-transform: uppercase; letter-spacing: .03em;
  }
  .detail-fb-card {
    background: hsl(152 60% 96%); border-radius: var(--radius);
    padding: 10px 12px; margin-bottom: 8px;
    display: flex; align-items: flex-start; gap: 8px;
  }
  .detail-fb-text {
    font-size: 13px; color: hsl(152 69% 26%); line-height: 1.5;
    flex: 1; white-space: pre-wrap; word-break: break-word;
  }
  .detail-fb-delete {
    background: none; border: none; cursor: pointer;
    color: hsl(152 30% 60%); font-size: 16px; line-height: 1;
    padding: 0 2px; flex-shrink: 0; transition: color .15s;
  }
  .detail-fb-delete:hover { color: var(--destructive); }
  .detail-fb-row { display: flex; gap: 6px; align-items: flex-end; }
  .detail-fb-input {
    flex: 1; border: none; border-radius: var(--radius);
    padding: 8px 12px; font-family: inherit; font-size: 13px;
    background: var(--background); color: var(--foreground);
    transition: box-shadow .15s; resize: none; overflow: hidden;
    min-height: 38px; max-height: 120px; line-height: 1.5;
  }
  .detail-fb-input:focus {
    outline: none;
    box-shadow: 0 0 0 2px hsl(211 100% 50% / .12);
  }

  /* Histogram view */
  .viz-panel { flex: 1; overflow: auto; padding: 16px; }
  .viz-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
  .viz-card {
    background: white; border: none; border-radius: var(--radius);
    padding: 14px; box-shadow: 0 1px 4px 0 rgb(0 0 0 / .07);
  }
  .viz-card-title { font-size: 13px; font-weight: 600; color: var(--foreground); margin-bottom: 2px; }
  .viz-card-sub { font-size: 11px; color: var(--muted-foreground); margin-bottom: 10px; }
  .viz-chart { }
  .viz-chart svg { display: block; width: 100%; }

  /* Tooltip */
  .tt {
    position: fixed; z-index: 100; pointer-events: none;
    background: white; border: none;
    border-radius: var(--radius); padding: 6px 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,.12); font-size: 12px;
    color: var(--foreground); max-width: 250px; opacity: 0;
    transition: opacity .12s;
  }
  .tt.show { opacity: 1; }
  .tt-label { color: var(--muted-foreground); font-size: 11px; }
  .tt-val { font-weight: 600; }

  /* Status bar */
  .statusbar {
    background: white;
    padding: 5px 16px; font-size: 12px; color: var(--muted-foreground);
    display: flex; gap: 16px; flex-shrink: 0; align-items: center;
    box-shadow: 0 -1px 3px 0 rgb(0 0 0 / .04);
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
    background: hsl(152 69% 97%); border: none;
    border-radius: var(--radius); padding: 12px 16px; margin: 12px 16px 0;
    display: flex; align-items: center; gap: 10px; font-size: 13px;
  }
  .complete-banner b { color: hsl(152 69% 28%); }

  /* Toast notifications */
  .toast-container {
    position: fixed; top: 60px; right: 16px; z-index: 50;
    display: flex; flex-direction: column; gap: 8px; max-width: 380px;
  }
  .toast {
    background: white; border: none; border-radius: var(--radius);
    padding: 10px 14px; box-shadow: 0 4px 16px rgba(0,0,0,.12);
    font-size: 13px; color: var(--foreground); line-height: 1.4;
    animation: toastIn .3s ease-out;
    display: flex; gap: 8px; align-items: flex-start;
  }
  .toast.info { }
  .toast.success { }
  .toast.warning { }
  .toast-body { flex: 1; }
  .toast-label { font-size: 11px; font-weight: 600; color: var(--muted-foreground); margin-bottom: 2px; }
  .toast-text { }
  .toast-text.truncated { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .toast-expand {
    font-size: 11px; color: var(--primary); cursor: pointer; border: none;
    background: none; font-family: inherit; padding: 2px 0; margin-top: 2px;
  }
  .toast-expand:hover { text-decoration: underline; }
  .toast-actions {
    display: flex; gap: 6px; margin-top: 8px;
  }
  .toast-action {
    padding: 4px 12px; font-size: 12px; font-weight: 500; border-radius: var(--radius);
    border: none; cursor: pointer; font-family: inherit; transition: background .15s;
  }
  .toast-action.confirm {
    background: var(--primary); color: white;
  }
  .toast-action.confirm:hover { background: hsl(211 100% 42%); }
  .toast-action.dismiss {
    background: var(--card); color: var(--muted-foreground);
  }
  .toast-action.dismiss:hover { background: hsl(211 20% 90%); }
  .toast-responded { font-size: 11px; color: var(--muted-foreground); font-style: italic; margin-top: 4px; }
  .toast-dismiss-x {
    background: none; border: none; cursor: pointer; color: var(--muted-foreground);
    font-size: 16px; line-height: 1; padding: 0; font-family: inherit;
  }
  .toast-dismiss-x:hover { color: var(--foreground); }
  @keyframes toastIn { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: none; } }

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

<div class="toast-container" id="toasts"></div>
<div class="tt" id="tooltip"></div>

<!-- Detail side panel -->
<div class="detail-panel" id="detail-panel">
  <div class="detail-header">
    <span class="detail-header-title" id="detail-title">Row</span>
    <button class="detail-nav-btn" id="detail-prev" onclick="navigateRow(-1)" title="Previous row (← arrow)">&#8592;</button>
    <button class="detail-nav-btn" id="detail-next" onclick="navigateRow(1)" title="Next row (→ arrow)">&#8594;</button>
    <button class="detail-close" onclick="closeRowDetail()" title="Close (Esc)">&#215;</button>
  </div>
  <div class="detail-fb" id="detail-fb"></div>
  <div class="detail-body" id="detail-body"></div>
</div>

<script>
let allDocs = [];
let columns = [];
let columnStats = {};
let sortCol = null;
let sortDir = 'asc';
let finished = false;
let killed = false;
let seenDocKeys = new Set();
let currentTab = 'table';
let selectedRow = null;
let columnFilters = {};
let filterVisible = {};
let lastResetToken = 0;
let selectedOp = null;
let userPickedOp = false;
let lastOps = [];

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
function autoGrowTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

/* --- Operations strip --- */
function updateOps(ops) {
  lastOps = ops;
  const doneCount = ops.filter(o => o.status === 'done').length;
  document.getElementById('h-ops').textContent = doneCount + '/' + ops.length;

  // Auto-select last op with outputs if none selected yet
  if (selectedOp === null) {
    for (let i = ops.length - 1; i >= 0; i--) {
      if (ops[i].status === 'done' || ops[i].status === 'running') {
        selectedOp = ops[i].name;
        break;
      }
    }
  }

  const strip = document.getElementById('ops-strip');
  strip.innerHTML = '';
  ops.forEach((op, idx) => {
    if (idx > 0) {
      const arrow = document.createElement('div');
      arrow.className = 'op-arrow';
      arrow.innerHTML = '&#8594;';
      strip.appendChild(arrow);
    }
    const el = document.createElement('div');
    el.className = 'op-item' + (selectedOp === op.name ? ' op-selected' : '');
    el.style.cursor = 'pointer';
    el.onclick = function() { selectOp(op.name); };
    let pctWidth = 0;
    let parts = '<span class="op-dot ' + op.status + '"></span>';
    parts += '<span class="op-name">' + op.op_type + ':' + op.name.split('/').pop() + '</span>';
    if (op.total) {
      pctWidth = Math.round(100 * op.completed / op.total);
      parts += '<span class="op-detail">' + op.completed + '/' + op.total + '</span>';
    }
    if (op.cost > 0) parts += '<span class="op-cost">' + fmtCost(op.cost) + '</span>';
    if (op.elapsed >= 1) parts += '<span class="op-detail">' + fmtDur(op.elapsed) + '</span>';
    if (op.status === 'running' && op.total) {
      parts += '<span class="op-bar" style="width:' + pctWidth + '%"></span>';
    }
    el.innerHTML = parts;
    strip.appendChild(el);
  });
}

function selectOp(opName) {
  userPickedOp = true;
  selectedOp = opName;
  columnFilters = {};
  filterVisible = {};
  sortCol = null;
  selectedRow = null;
  document.getElementById('detail-panel').classList.remove('open');
  columns = discoverColumns();
  recomputeStats();
  renderTableHead();
  renderTableBody();
  if (currentTab === 'visualize') renderVizPanel();
  updateOps(lastOps);
  document.getElementById('f-rows').textContent = getVisibleDocs().length + ' row' + (getVisibleDocs().length === 1 ? '' : 's');
}

function getVisibleDocs() {
  return allDocs.filter(d => d.op_name === selectedOp);
}

/* --- Column stats (matching playground logic) --- */
function computeColumnStats(colKey) {
  const vals = allDocs.filter(d => !selectedOp || d.op_name === selectedOp).map(d => d.fields[colKey]).filter(v => v != null);
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

/* --- Histogram rendering (SVG) --- */
function unitLabel(type) {
  return { number: '', array: ' items', boolean: '', 'string-chars': ' chars', 'string-words': ' words' }[type] || '';
}

function renderHistogram(stats, container, tall) {
  container.innerHTML = '';
  if (!stats) return;

  const barH = tall ? 90 : 40;
  const labelH = tall ? 16 : 12;
  const totalH = barH + labelH + 2;
  const pad = tall ? 4 : 2;
  const barR = tall ? 3 : 2;
  const chartColor = 'hsl(173,58%,39%)';
  const hoverColor = 'hsl(173,58%,32%)';
  const labelColor = 'hsl(211,5%,35%)';
  const labelSize = tall ? 10 : 8;

  let items, labels;
  if (stats.isLowCardinality) {
    items = stats.sortedValueCounts.slice(0, 10);
    labels = items.map(d => d.value.length > 8 ? d.value.slice(0, 7) + '…' : d.value);
  } else {
    const u = unitLabel(stats.type);
    items = stats.distribution.map((count, i) => ({
      count,
      label: Math.round(stats.min + i * stats.bucketSize),
      fullLabel: Math.round(stats.min + i * stats.bucketSize) + '–' + Math.round(stats.min + (i + 1) * stats.bucketSize) + u
    }));
    labels = items.map(d => String(d.label));
  }

  const n = items.length;
  if (!n) return;
  const maxC = Math.max(...items.map(d => d.count));
  if (maxC === 0) return;

  const svgW = container.clientWidth || 200;
  const gap = Math.max(1, Math.round(svgW / n * 0.15));
  const barW = Math.max(4, (svgW - gap * (n - 1)) / n);

  let svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + svgW + '" height="' + totalH + '" viewBox="0 0 ' + svgW + ' ' + totalH + '">';

  items.forEach((d, i) => {
    const x = i * (barW + gap);
    const h = Math.max(1, (d.count / maxC) * barH);
    const y = barH - h;
    const pct = ((d.count / stats.totalCount) * 100).toFixed(1);
    const ttLabel = stats.isLowCardinality ? d.value : d.fullLabel;
    const ttVal = d.count.toLocaleString() + ' (' + pct + '%)';

    svg += '<rect x="' + x + '" y="' + y + '" width="' + barW + '" height="' + h + '" rx="' + barR + '" fill="' + chartColor + '" data-tt-label="' + escHtml(ttLabel) + '" data-tt-val="' + escHtml(ttVal) + '" style="cursor:pointer"><title>' + escHtml(ttLabel) + ': ' + escHtml(ttVal) + '</title></rect>';

    // Label
    if (tall || i % Math.ceil(n / 5) === 0) {
      const lbl = labels[i] || '';
      svg += '<text x="' + (x + barW / 2) + '" y="' + (barH + labelH) + '" text-anchor="middle" fill="' + labelColor + '" font-size="' + labelSize + '" font-family="-apple-system,BlinkMacSystemFont,sans-serif">' + escHtml(lbl) + '</text>';
    }
  });

  svg += '</svg>';
  container.innerHTML = svg;
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

/* --- Column filters --- */
function setColumnFilter(col, value) {
  if (value) {
    columnFilters[col] = value;
  } else {
    delete columnFilters[col];
  }
  renderTableBody();
  if (selectedRow !== null) renderDetailPanel();
}

function getFilteredIndices() {
  const indices = [];
  allDocs.forEach((doc, i) => {
    if (selectedOp && doc.op_name !== selectedOp) return;
    indices.push(i);
  });
  const activeFilters = Object.entries(columnFilters);
  if (!activeFilters.length) return indices;
  return indices.filter(idx => {
    const doc = allDocs[idx];
    return activeFilters.every(([col, query]) => {
      const val = doc.fields[col];
      if (val == null) return false;
      const s = typeof val === 'string' ? val : JSON.stringify(val);
      return s.toLowerCase().includes(query.toLowerCase());
    });
  });
}

/* --- Table rendering --- */
function discoverColumns() {
  const seen = new Set();
  const cols = [];
  allDocs.forEach(doc => {
    if (selectedOp && doc.op_name !== selectedOp) return;
    for (const k of Object.keys(doc.fields)) {
      if (!seen.has(k)) { seen.add(k); cols.push(k); }
    }
  });
  return cols;
}

function getSortedIndices() {
  const indices = getFilteredIndices();
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

  // Data columns (no operation column)
  columns.forEach(col => {
    const th = document.createElement('th');
    const stats = columnStats[col];
    if (userColWidths[col]) {
      th.style.width = userColWidths[col] + 'px';
      th.style.minWidth = userColWidths[col] + 'px';
    } else if (stats) {
      if (stats.type === 'string-words' && !stats.isLowCardinality) {
        th.style.minWidth = '200px';
        th.style.width = stats.avg > 8 ? '300px' : '200px';
      } else if (stats.type === 'number' || stats.isLowCardinality) {
        th.style.minWidth = '80px';
        th.style.width = '100px';
      } else {
        th.style.minWidth = '100px';
      }
    }
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

    const fv = filterVisible[col];
    const fc = columnFilters[col] || '';
    const filterBtnCls = 'filter-toggle' + (fc ? ' active' : '');
    const filterInputDisplay = fv || fc ? '' : ' style="display:none"';

    th.innerHTML =
      '<div class="col-header">' +
        '<div class="col-header-name" onclick="toggleSort(\'' + escHtml(col) + '\')">' +
          '<span class="sort-icon">' + arrow + '</span>' +
          '<span>' + escHtml(col) + '</span>' +
          '<button class="' + filterBtnCls + '" onclick="event.stopPropagation();toggleFilterInput(\'' + escHtml(col) + '\')">&#128269;</button>' +
        '</div>' +
        statsHtml +
        '<div class="col-filter-wrap" id="cfwrap-' + escHtml(col) + '"' + filterInputDisplay + '>' +
          '<input class="col-filter" type="text" placeholder="Filter…" value="' + escHtml(fc) + '" ' +
            'oninput="setColumnFilter(\'' + escHtml(col) + '\', this.value); updateFilterClear(\'' + escHtml(col) + '\')" ' +
            'onclick="event.stopPropagation()">' +
          '<button class="col-filter-clear' + (fc ? ' active' : '') + '" id="cfclear-' + escHtml(col) + '" ' +
            'onclick="event.stopPropagation();clearColumnFilter(\'' + escHtml(col) + '\')">&#215;</button>' +
        '</div>' +
        '<div class="col-histogram" id="hist-' + escHtml(col) + '"></div>' +
      '</div>' +
      '<div class="col-resize" data-col="' + escHtml(col) + '"></div>';
    tr.appendChild(th);
  });

  thead.appendChild(tr);

  // Render histograms
  columns.forEach(col => {
    const container = document.getElementById('hist-' + col);
    if (container) renderHistogram(columnStats[col], container, false);
  });
}

function toggleFilterInput(col) {
  filterVisible[col] = !filterVisible[col];
  var wrap = document.getElementById('cfwrap-' + col);
  if (wrap) {
    if (filterVisible[col] || columnFilters[col]) {
      wrap.style.display = '';
      var inp = wrap.querySelector('.col-filter');
      if (inp) inp.focus();
    } else {
      wrap.style.display = 'none';
    }
  }
}

function updateFilterClear(col) {
  var btn = document.getElementById('cfclear-' + col);
  if (btn) {
    btn.classList.toggle('active', !!columnFilters[col]);
  }
}

function clearColumnFilter(col) {
  delete columnFilters[col];
  var wrap = document.getElementById('cfwrap-' + col);
  if (wrap) {
    var inp = wrap.querySelector('.col-filter');
    if (inp) inp.value = '';
  }
  updateFilterClear(col);
  renderTableBody();
}

function renderTableBody() {
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  const sorted = getSortedIndices();

  sorted.forEach(idx => {
    const doc = allDocs[idx];
    const tr = document.createElement('tr');
    if (selectedRow === idx) tr.className = 'row-selected';
    tr.onclick = function(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
      openRowDetail(idx);
    };

    // Row number with feedback dot
    const tdIdx = document.createElement('td');
    tdIdx.className = 'col-idx';
    const idxInner = '<div class="idx-cell-inner">' +
      (doc._feedbacks && doc._feedbacks.length ? '<span class="fb-dot"></span>' : '') +
      '<span>' + (doc.doc_index + 1) + '</span>' +
    '</div>';
    tdIdx.innerHTML = idxInner;
    tr.appendChild(tdIdx);

    // Data cells (no operation column, no feedback column)
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

    tbody.appendChild(tr);
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

/* --- Row detail panel --- */
function openRowDetail(idx) {
  selectedRow = idx;
  renderDetailPanel();
  document.getElementById('detail-panel').classList.add('open');
  renderTableBody();
}

function closeRowDetail() {
  selectedRow = null;
  document.getElementById('detail-panel').classList.remove('open');
  renderTableBody();
}

function navigateRow(delta) {
  if (selectedRow === null) return;
  const sorted = getSortedIndices();
  const curPos = sorted.indexOf(selectedRow);
  if (curPos === -1) return;
  const newPos = curPos + delta;
  if (newPos < 0 || newPos >= sorted.length) return;
  selectedRow = sorted[newPos];
  renderDetailPanel();
  renderTableBody();
}

function renderDetailPanel() {
  if (selectedRow === null) return;
  const doc = allDocs[selectedRow];
  if (!doc) return;

  const sorted = getSortedIndices();
  const curPos = sorted.indexOf(selectedRow);
  document.getElementById('detail-title').innerHTML = '<span>' + (curPos + 1) + ' of ' + sorted.length + '</span>' +
    '<span class="detail-hotkey-hint">← → to navigate</span>';

  // Nav button state
  document.getElementById('detail-prev').disabled = (curPos <= 0);
  document.getElementById('detail-next').disabled = (curPos >= sorted.length - 1);

  // Body: operation tag + fields
  let bodyHtml = '<div class="detail-op-tag">' + escHtml(doc.op_type + ':' + doc.op_name.split('/').pop()) + '</div>';
  for (const key of Object.keys(doc.fields)) {
    const val = doc.fields[key];
    let displayVal;
    if (val == null) {
      displayVal = '<span style="color:var(--muted-foreground);font-style:italic">—</span>';
    } else if (typeof val === 'string') {
      displayVal = escHtml(val);
    } else {
      displayVal = escHtml(JSON.stringify(val, null, 2));
    }
    bodyHtml += '<div class="detail-field">' +
      '<div class="detail-field-key">' + escHtml(key) + '</div>' +
      '<div class="detail-field-val">' + displayVal + '</div>' +
    '</div>';
  }
  document.getElementById('detail-body').innerHTML = bodyHtml;

  // Feedback section
  const fbs = doc._feedbacks || [];
  let fbHtml = '<div class="detail-fb-label">Feedback' + (fbs.length ? ' (' + fbs.length + ')' : '') + '</div>';
  fbs.forEach((fb, fi) => {
    fbHtml += '<div class="detail-fb-card">' +
      '<div class="detail-fb-text">' + escHtml(fb) + '</div>' +
      '<button class="detail-fb-delete" onclick="deleteDocFeedback(' + selectedRow + ',' + fi + ')" title="Delete">&#215;</button>' +
    '</div>';
  });
  fbHtml += '<div class="detail-fb-row">' +
    '<textarea class="detail-fb-input" id="detail-fb-input" rows="1" placeholder="Add feedback…" ' +
      'oninput="autoGrowTextarea(this)" ' +
      'onkeydown="if(event.key===\'Enter\'&&!event.shiftKey){event.preventDefault();sendDocFeedback(' + selectedRow + ');}"></textarea>' +
    '<button class="btn btn-primary" onclick="sendDocFeedback(' + selectedRow + ')">Send</button>' +
  '</div>';
  document.getElementById('detail-fb').innerHTML = fbHtml;
}

/* Hotkeys: ESC close, Left/Right arrows navigate rows (like playground) */
document.addEventListener('keydown', function(e) {
  if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return;
  if (e.key === 'Escape' && selectedRow !== null) {
    closeRowDetail();
  } else if (e.key === 'ArrowLeft' && selectedRow !== null) {
    e.preventDefault();
    navigateRow(-1);
  } else if (e.key === 'ArrowRight' && selectedRow !== null) {
    e.preventDefault();
    navigateRow(1);
  }
});

/* --- Column resize --- */
var userColWidths = {};
(function() {
  let resizing = null;
  document.addEventListener('mousedown', function(e) {
    if (!e.target.classList.contains('col-resize')) return;
    e.preventDefault();
    const th = e.target.parentElement;
    const col = e.target.getAttribute('data-col');
    const startX = e.clientX;
    const startW = th.offsetWidth;
    e.target.classList.add('active');
    resizing = { th: th, handle: e.target, col: col, startX: startX, startW: startW };
  });
  document.addEventListener('mousemove', function(e) {
    if (!resizing) return;
    e.preventDefault();
    const w = Math.max(50, resizing.startW + (e.clientX - resizing.startX));
    resizing.th.style.width = w + 'px';
    resizing.th.style.minWidth = w + 'px';
    userColWidths[resizing.col] = w;
  });
  document.addEventListener('mouseup', function() {
    if (resizing) { resizing.handle.classList.remove('active'); resizing = null; }
  });
})();

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
      '<div class="viz-chart" id="vizchart-' + escHtml(col) + '"></div>';
    grid.appendChild(card);

    const chartEl = document.getElementById('vizchart-' + col);
    renderHistogram(stats, chartEl, true);
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

  // Auto-advance selectedOp only if user hasn't manually picked one
  if (!userPickedOp) {
    let latestOpWithDocs = null;
    if (lastOps.length) {
      for (let i = lastOps.length - 1; i >= 0; i--) {
        const opName = lastOps[i].name;
        if (allDocs.some(d => d.op_name === opName)) {
          latestOpWithDocs = opName;
          break;
        }
      }
    }
    if (latestOpWithDocs && selectedOp !== latestOpWithDocs) {
      selectedOp = latestOpWithDocs;
      columnFilters = {};
      filterVisible = {};
      sortCol = null;
      updateOps(lastOps);
    }
  }

  columns = discoverColumns();
  recomputeStats();

  renderTableHead();
  renderTableBody();
  if (currentTab === 'visualize') renderVizPanel();

  const visible = getVisibleDocs();
  document.getElementById('f-rows').textContent = visible.length + ' row' + (visible.length === 1 ? '' : 's');
}

/* --- Feedback --- */
function sendDocFeedback(idx) {
  var input = document.getElementById('detail-fb-input');
  if (!input) return;
  const text = input.value.trim();
  if (!text) return;
  const doc = allDocs[idx];
  fetch('/feedback/doc', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ op_name: doc.op_name, doc_index: doc.doc_index, doc_snapshot: doc.fields, text: text })
  });
  if (!doc._feedbacks) doc._feedbacks = [];
  doc._feedbacks.push(text);
  input.value = '';
  autoGrowTextarea(input);
  renderTableBody();
  renderDetailPanel();
}

function deleteDocFeedback(idx, fbIndex) {
  const doc = allDocs[idx];
  if (!doc || !doc._feedbacks) return;
  fetch('/feedback/doc/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ op_name: doc.op_name, doc_index: doc.doc_index, feedback_index: fbIndex })
  });
  doc._feedbacks.splice(fbIndex, 1);
  renderTableBody();
  renderDetailPanel();
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

/* --- Toasts (agent messages) --- */
let seenMsgIds = new Set();

function showToast(msg) {
  if (seenMsgIds.has(msg.id)) {
    // Update existing toast if response came in
    if (msg.response) {
      const existing = document.getElementById('toast-' + msg.id);
      if (existing) {
        const actionsEl = existing.querySelector('.toast-actions');
        if (actionsEl) {
          actionsEl.innerHTML = '<div class="toast-responded">✓ ' + escHtml(msg.response) + '</div>';
        }
      }
    }
    return;
  }
  seenMsgIds.add(msg.id);
  const container = document.getElementById('toasts');
  const el = document.createElement('div');
  el.className = 'toast ' + (msg.type || 'info');
  el.id = 'toast-' + msg.id;

  let actionsHtml = '';
  if (msg.actions && msg.actions.length > 0) {
    actionsHtml = '<div class="toast-actions">';
    msg.actions.forEach(a => {
      const cls = a.toLowerCase().includes('confirm') || a.toLowerCase().includes('run') || a.toLowerCase().includes('yes')
        ? 'confirm' : 'dismiss';
      actionsHtml += '<button class="toast-action ' + cls + '" onclick="respondToast(' + msg.id + ',\'' + escHtml(a) + '\')">' + escHtml(a) + '</button>';
    });
    actionsHtml += '</div>';
  }

  const isLong = msg.text.length > 100;
  const textCls = isLong ? 'toast-text truncated' : 'toast-text';
  const expandBtn = isLong ? '<button class="toast-expand" onclick="toggleToastExpand(this)">Show more</button>' : '';

  el.innerHTML =
    '<div class="toast-body">' +
      '<div class="toast-label">Agent</div>' +
      '<div class="' + textCls + '">' + escHtml(msg.text) + '</div>' +
      expandBtn +
      actionsHtml +
    '</div>' +
    '<button class="toast-dismiss-x" onclick="this.parentElement.remove()">×</button>';
  container.appendChild(el);
  if (!msg.actions || msg.actions.length === 0) {
    setTimeout(() => { if (el.parentElement) el.remove(); }, 15000);
  }
}

function toggleToastExpand(btn) {
  const textEl = btn.previousElementSibling;
  if (textEl.classList.contains('truncated')) {
    textEl.classList.remove('truncated');
    btn.textContent = 'Show less';
  } else {
    textEl.classList.add('truncated');
    btn.textContent = 'Show more';
  }
}

function respondToast(msgId, action) {
  fetch('/message/respond', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: msgId, action: action}),
  });
  const el = document.getElementById('toast-' + msgId);
  if (el) {
    const actionsEl = el.querySelector('.toast-actions');
    if (actionsEl) {
      actionsEl.innerHTML = '<div class="toast-responded">✓ ' + escHtml(action) + '</div>';
    }
    setTimeout(() => { if (el.parentElement) el.remove(); }, 5000);
  }
}

/* --- SSE --- */
const evtSource = new EventSource('/events');
evtSource.onmessage = function(e) {
  const data = JSON.parse(e.data);

  // Detect table reset (e.g. re-run after confirm)
  if (data.reset_token !== undefined && data.reset_token > lastResetToken) {
    lastResetToken = data.reset_token;
    allDocs = [];
    columns = [];
    columnStats = {};
    seenDocKeys = new Set();
    selectedRow = null;
    selectedOp = null;
    userPickedOp = false;
    finished = false;
    document.getElementById('detail-panel').classList.remove('open');
    document.getElementById('complete-banner').classList.add('hidden');
    document.getElementById('kill-btn').classList.remove('hidden');
    const dot = document.getElementById('status-dot');
    dot.classList.remove('done'); dot.classList.add('live');
    renderTableHead();
    renderTableBody();
  }

  updateOps(data.ops);
  syncDocs(data.all_docs);
  document.getElementById('h-cost').textContent = fmtCost(data.total_cost);
  document.getElementById('h-time').textContent = fmtDur(data.elapsed);
  document.getElementById('f-feedback').textContent = 'Feedback: ' + data.feedback_count;
  document.getElementById('f-status').textContent = data.finished ? 'Complete' : 'Running';

  if (data.agent_messages) {
    data.agent_messages.forEach(msg => showToast(msg));
  }

  // Sync external doc feedback (submitted via API, not through UI)
  if (data.doc_feedback) {
    let fbChanged = false;
    data.doc_feedback.forEach(fb => {
      const doc = allDocs.find(d => d.op_name === fb.op_name && d.doc_index === fb.doc_index);
      if (doc) {
        if (!doc._feedbacks) doc._feedbacks = [];
        if (!doc._feedbacks.includes(fb.feedback)) {
          doc._feedbacks.push(fb.feedback);
          fbChanged = true;
        }
      }
    });
    if (fbChanged) {
      renderTableBody();
    }
  }

  if (data.finished && !finished) {
    finished = true;
    const dot = document.getElementById('status-dot');
    dot.classList.remove('live');
    dot.classList.add('done');
    const banner = document.getElementById('complete-banner');
    banner.classList.remove('hidden');
    document.getElementById('complete-summary').textContent = fmtCost(data.total_cost) + ' · ' + fmtDur(data.elapsed) + ' · ' + getVisibleDocs().length + ' outputs';
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
# Persistent server helpers
# ---------------------------------------------------------------------------

def _detect_server() -> int | None:
    """Return the port of a running persistent server, or None."""
    try:
        with open(_PORT_FILE) as f:
            port = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        req = Request(f"http://localhost:{port}/health")
        with urlopen(req, timeout=1) as resp:
            if resp.status == 200:
                return port
    except (URLError, OSError):
        pass
    try:
        os.remove(_PORT_FILE)
    except OSError:
        pass
    return None


def _push_state_to_server(port: int, tracker: ProgressTracker):
    """Background loop that pushes tracker state to a persistent server."""
    _seen_doc_fb = 0
    _seen_pipe_fb = 0
    while not getattr(tracker, "_push_stop", threading.Event()).is_set():
        try:
            state = tracker.snapshot()
            ops = []
            for op in state.ops:
                ops.append({
                    "name": op.name, "op_type": op.op_type, "model": op.model,
                    "status": op.status, "total": op.total, "completed": op.completed,
                    "errors": op.errors, "out_count": op.out_count,
                    "cost": op.cost, "elapsed": op.elapsed,
                })
            all_docs = []
            for op in state.ops:
                for i, doc in enumerate(op.outputs):
                    all_docs.append({
                        "op_name": op.name, "op_type": op.op_type, "doc_index": i,
                        "fields": {k: _trunc(v) for k, v in doc.items() if not k.startswith("_")},
                    })
            event = {
                "ops": ops, "all_docs": all_docs,
                "total_cost": state.total_cost, "elapsed": state.elapsed,
                "finished": state.finished,
            }
            body = json.dumps(event).encode()
            req = Request(
                f"http://localhost:{port}/state/push",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            urlopen(req, timeout=2)
        except (URLError, OSError) as exc:
            import sys
            print(f"[push] error: {exc}", file=sys.stderr, flush=True)
        # Check for feedback and kill requests from the server
        try:
            req = Request(f"http://localhost:{port}/feedback/poll")
            with urlopen(req, timeout=1) as resp:
                data = json.loads(resp.read())
                if data.get("killed") and data.get("kill_reason") is not None:
                    tracker.kill_requested = True
                doc_fb = data.get("doc_feedback", [])
                if len(doc_fb) > _seen_doc_fb:
                    for fb in doc_fb[_seen_doc_fb:]:
                        print(f"[FEEDBACK:doc] op={fb.get('operation','')} "
                              f"doc_index={fb.get('doc_index','')} | "
                              f"{fb.get('feedback','')}", flush=True)
                    _seen_doc_fb = len(doc_fb)
                pipe_fb = data.get("pipeline_feedback", [])
                if len(pipe_fb) > _seen_pipe_fb:
                    for fb in pipe_fb[_seen_pipe_fb:]:
                        print(f"[FEEDBACK:pipeline] {fb.get('feedback','')}",
                              flush=True)
                    _seen_pipe_fb = len(pipe_fb)
        except (URLError, OSError):
            pass
        tracker._push_stop.wait(0.5)
    # Final push
    try:
        state = tracker.snapshot()
        ops = []
        for op in state.ops:
            ops.append({
                "name": op.name, "op_type": op.op_type, "model": op.model,
                "status": op.status, "total": op.total, "completed": op.completed,
                "errors": op.errors, "out_count": op.out_count,
                "cost": op.cost, "elapsed": op.elapsed,
            })
        all_docs = []
        for op in state.ops:
            for i, doc in enumerate(op.outputs):
                all_docs.append({
                    "op_name": op.name, "op_type": op.op_type, "doc_index": i,
                    "fields": {k: _trunc(v) for k, v in doc.items() if not k.startswith("_")},
                })
        event = {
            "ops": ops, "all_docs": all_docs,
            "total_cost": state.total_cost, "elapsed": state.elapsed,
            "finished": True,
        }
        body = json.dumps(event).encode()
        req = Request(
            f"http://localhost:{port}/state/push",
            data=body, headers={"Content-Type": "application/json"},
        )
        urlopen(req, timeout=2)
    except (URLError, OSError):
        pass


def start_server(port: int = 0) -> int:
    """Start a persistent feedback server.

    Returns the port. Writes it to ``.docetl_server_port``.
    The server runs until the process is killed.
    """
    existing = _detect_server()
    if existing is not None:
        print(f"Server already running on port {existing}")
        return existing

    feedback = FeedbackStore()
    broadcaster = _Broadcaster(None, feedback)
    broadcaster.start()

    handler_cls = _make_handler(None, feedback, broadcaster)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    actual_port = server.server_address[1]

    with open(_PORT_FILE, "w") as f:
        f.write(str(actual_port))

    url = f"http://localhost:{actual_port}"
    print(f"DocETL feedback server running at {url}")
    print(f"Port written to {_PORT_FILE}")
    print("Press Ctrl+C to stop.")

    import webbrowser
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.remove(_PORT_FILE)
        except OSError:
            pass
        server.shutdown()
    return actual_port


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _auto_start_server() -> int | None:
    """Spawn a persistent server as a detached subprocess. Returns port."""
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, "-m", "docetl.cli", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Wait for the port file to appear (server writes it on startup).
    for _ in range(50):
        time.sleep(0.1)
        port = _detect_server()
        if port is not None:
            return port
    # Server didn't start in time — fall back.
    try:
        proc.kill()
    except OSError:
        pass
    return None


def run_with_web_ui(runner: "DSLRunner") -> float:
    """Run the pipeline with a web-based progress + feedback UI.

    If a persistent server is running (started via ``docetl serve``), the
    pipeline pushes state to it.  Otherwise it auto-starts one as a detached
    subprocess so it survives across pipeline runs.
    """
    import threading

    from tqdm import tqdm as _tqdm

    _tqdm.set_lock(threading.RLock())

    tracker = ProgressTracker(concurrency=min(runner.max_threads or 1, 64))
    runner.progress_tracker = tracker
    runner._tui_active = True
    set_active_tracker(tracker)
    tracker.pipeline_start(runner.list_pipeline_operations())

    server_port = _detect_server()

    if server_port is None:
        runner.console.log("[dim]No persistent server found — starting one…[/dim]")
        server_port = _auto_start_server()

    if server_port is not None:
        return _run_with_remote_server(runner, tracker, server_port)
    else:
        return _run_with_inline_server(runner, tracker)


def _run_with_remote_server(runner: "DSLRunner", tracker: ProgressTracker, port: int) -> float:
    """Push state to a persistent server. Returns immediately after pipeline."""
    import threading

    runner.console.log(
        f"[bold blue]Using persistent server:[/bold blue] http://localhost:{port}"
    )

    # Reset the server UI for this new run
    try:
        req = Request(
            f"http://localhost:{port}/reset",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(req, timeout=2)
    except (URLError, OSError):
        pass

    tracker._push_stop = threading.Event()
    push_thread = threading.Thread(
        target=_push_state_to_server, args=(port, tracker), daemon=True
    )
    push_thread.start()

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
        tracker._push_stop.set()
        push_thread.join(timeout=5)
        set_active_tracker(None)
        runner.progress_tracker = None
        runner._tui_active = False

    runner.console.log(
        f"[bold]Pipeline finished.[/bold] Feedback server still running at http://localhost:{port}"
    )

    # Print all feedback received so it appears in subagent output
    try:
        req = Request(f"http://localhost:{port}/feedback/poll")
        with urlopen(req, timeout=2) as resp:
            fb = json.loads(resp.read())
            if fb.get("doc_feedback") or fb.get("pipeline_feedback"):
                print("\n=== Feedback received during this run ===", flush=True)
                for d in fb.get("doc_feedback", []):
                    print(f"  [doc] op={d.get('operation','')} "
                          f"doc_index={d.get('doc_index','')} | "
                          f"{d.get('feedback','')}", flush=True)
                for p in fb.get("pipeline_feedback", []):
                    print(f"  [pipeline] {p.get('feedback','')}", flush=True)
                print("=========================================\n", flush=True)
    except (URLError, OSError):
        pass

    if killed:
        cost = runner.total_cost
    return cost


def _run_with_inline_server(runner: "DSLRunner", tracker: ProgressTracker) -> float:
    """Start an inline server, run pipeline, wait for feedback, then shut down."""
    import threading

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
        "[dim]Opening browser… The pipeline is running.[/dim]"
    )
    import webbrowser
    webbrowser.open(url)

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
        set_active_tracker(None)
        runner.progress_tracker = None
        runner._tui_active = False

    broadcaster.stop()

    runner.console.log(
        "[bold]Pipeline finished.[/bold] Review results in the browser "
        "and submit feedback — it will appear in [dim].docetl_feedback.log[/dim]."
    )

    # Brief pause so the final SSE push reaches the browser before shutdown.
    time.sleep(2)
    server.shutdown()

    if feedback.has_any:
        fb_data = feedback.to_dict()
        runner.console.log(f"\n[bold]Human feedback collected:[/bold]")
        for item in feedback.pipeline_feedback:
            runner.console.log(f"  [pipeline] {item['feedback']}")
        for item in feedback.doc_feedback:
            runner.console.log(
                f"  [doc] {item['operation']} #{item['doc_index']}: {item['feedback']}"
            )
        if killed and feedback.kill_reason:
            runner.console.log(f"  [kill reason] {feedback.kill_reason}")

    if killed:
        cost = runner.total_cost

    return cost
