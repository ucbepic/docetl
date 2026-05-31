"use client";

import React, { memo, useEffect, useMemo, useState } from "react";
import { useWebSocket } from "@/contexts/WebSocketContext";

// Mirrors docetl.progress.events.RunState.to_dict()
interface OpState {
  step: string;
  name: string;
  type: string;
  model: string | null;
  status: "queued" | "running" | "done" | "error";
  total: number | null;
  completed: number;
  errors: number;
  out_count: number | null;
  cost: number;
  tokens: number;
  elapsed: number;
  samples?: Record<string, unknown>[];
}

interface RunState {
  run_id: string;
  finished: boolean;
  elapsed: number;
  total_cost: number;
  concurrency: number;
  ops: OpState[];
}

const STATUS_COLOR: Record<OpState["status"], string> = {
  done: "#22c55e",
  running: "#f59e0b",
  error: "#ef4444",
  queued: "#d4d4d8",
};
const OP_GLYPH: Record<OpState["status"], string> = {
  done: "✓",
  running: "◐",
  error: "✗",
  queued: "○",
};

const MAX_CELLS = 2000; // beyond this we aggregate into a heatmap

function fmtDur(secs: number): string {
  const s = Math.floor(secs);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

function fmtK(n: number): string {
  if (n < 1000) return `${n}`;
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

// Synthesize a per-document status from the reliable counts (mirrors OpState.cell_status)
function cellStatus(
  op: OpState,
  index: number,
  runningBand: number
): OpState["status"] {
  if (op.total === null) return "queued";
  if (index < op.errors) return "error";
  if (index < op.completed) return "done";
  if (op.status === "running" && index < op.completed + runningBand)
    return "running";
  return "queued";
}

function heatColor(frac: number, hasError: boolean): string {
  if (hasError) return "#ef4444";
  const r = Math.round(212 - 190 * frac);
  const g = Math.round(212 - 78 * frac);
  const b = Math.round(216 - 122 * frac);
  return `rgb(${r},${g},${b})`;
}

const DocGrid = memo(function DocGrid({
  op,
  concurrency,
  selDoc,
  onSelect,
}: {
  op: OpState;
  concurrency: number;
  selDoc: number | null;
  onSelect: (i: number | null) => void;
}) {
  const total = op.total ?? 0;
  const heatmap = total > MAX_CELLS;
  const cellCount = heatmap ? Math.min(MAX_CELLS, total) : total;
  const bucket = heatmap ? Math.ceil(total / MAX_CELLS) : 1;
  const runningBand = op.status === "running" ? concurrency : 0;

  if (total === 0) {
    return (
      <div className="text-xs text-gray-400 p-2">
        {op.status === "queued" ? "Queued — not started yet." : "No documents."}
      </div>
    );
  }

  const cells = [];
  for (let i = 0; i < cellCount; i++) {
    let color: string;
    let title: string;
    if (heatmap) {
      const lo = i * bucket;
      const hi = Math.min(total, lo + bucket);
      const n = hi - lo;
      const done = Math.max(0, Math.min(n, op.completed - lo));
      const err = Math.max(0, Math.min(n, op.errors - lo));
      color = heatColor(n ? done / n : 0, err > 0);
      title = `docs ${lo}–${hi - 1}: ${done}/${n} done`;
    } else {
      const st = cellStatus(op, i, runningBand);
      color = STATUS_COLOR[st];
      title = `doc #${i}: ${st}`;
    }
    const selected = !heatmap && selDoc === i;
    cells.push(
      <div
        key={i}
        title={title}
        onClick={heatmap ? undefined : () => onSelect(selDoc === i ? null : i)}
        className={`rounded-[2px] ${heatmap ? "" : "cursor-pointer"}`}
        style={{
          width: 9,
          height: 9,
          background: color,
          outline: selected ? "2px solid #2563eb" : "none",
          outlineOffset: 1,
        }}
      />
    );
  }

  return (
    <div className="flex flex-col gap-1">
      {heatmap && (
        <div className="text-[11px] text-purple-600 font-medium">
          heatmap — each cell ≈ {bucket.toLocaleString()} docs
        </div>
      )}
      <div className="flex flex-wrap gap-[3px] content-start">{cells}</div>
    </div>
  );
});

export const PipelineProgress = memo(function PipelineProgress({
  stateOverride,
}: {
  // For previews/tests: render a fixed RunState instead of reading the websocket.
  stateOverride?: RunState;
}) {
  const { lastMessage } = useWebSocket();
  const [liveState, setLiveState] = useState<RunState | null>(null);
  const [selOp, setSelOp] = useState(0);
  const [selDoc, setSelDoc] = useState<number | null>(null);

  useEffect(() => {
    if (lastMessage?.type === "state" && lastMessage.data) {
      setLiveState(lastMessage.data as RunState);
    }
  }, [lastMessage]);

  const state = stateOverride ?? liveState;

  // reset doc selection when switching operations
  useEffect(() => {
    setSelDoc(null);
  }, [selOp]);

  const op = useMemo(() => {
    if (!state || state.ops.length === 0) return null;
    return state.ops[Math.min(selOp, state.ops.length - 1)];
  }, [state, selOp]);

  if (!state || state.ops.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-gray-400">
        Progress will appear here when a pipeline run starts.
      </div>
    );
  }

  const doneOps = state.ops.filter((o) => o.status === "done").length;

  // group ops by step for the left panel
  const groups: { step: string; ops: { op: OpState; idx: number }[] }[] = [];
  state.ops.forEach((o, idx) => {
    const g = groups.find((x) => x.step === o.step);
    if (g) g.ops.push({ op: o, idx });
    else groups.push({ step: o.step, ops: [{ op: o, idx }] });
  });

  const sample =
    op && selDoc !== null && op.samples && op.samples[selDoc]
      ? op.samples[selDoc]
      : null;

  return (
    <div className="flex h-full gap-3 p-3 text-sm">
      {/* Left: operations */}
      <div className="w-64 shrink-0 overflow-auto border rounded-md p-2">
        <div className="font-semibold">Pipeline</div>
        <div className="text-xs text-gray-500 mb-2">
          {doneOps}/{state.ops.length} ops · ${state.total_cost.toFixed(2)} ·{" "}
          {fmtDur(state.elapsed)}
          {state.finished && (
            <span className="ml-1 text-green-600 font-medium">✓ complete</span>
          )}
        </div>
        {groups.map((g) => (
          <div key={g.step} className="mb-2">
            <div className="text-[11px] uppercase tracking-wide text-cyan-700 font-semibold">
              {g.step}
            </div>
            {g.ops.map(({ op: o, idx }) => (
              <button
                key={o.name}
                onClick={() => setSelOp(idx)}
                className={`w-full text-left rounded px-1 py-1 ${
                  idx === selOp ? "bg-blue-50 ring-1 ring-blue-300" : "hover:bg-gray-50"
                }`}
              >
                <div className="flex items-center gap-1">
                  <span style={{ color: STATUS_COLOR[o.status] }}>
                    {OP_GLYPH[o.status]}
                  </span>
                  <span className="truncate font-medium">
                    {o.type}:{o.name.split("/").pop()}
                  </span>
                </div>
                <div className="text-[11px] text-gray-500 pl-4">
                  {o.total != null && (
                    <>
                      {o.completed.toLocaleString()}/{o.total.toLocaleString()}
                      {o.status === "running" &&
                        ` · ${Math.round((100 * o.completed) / Math.max(1, o.total))}%`}
                    </>
                  )}
                  {o.errors > 0 && (
                    <span className="text-red-500"> · !{o.errors}</span>
                  )}
                  {(o.cost > 0 || o.tokens > 0) && (
                    <>
                      {" "}
                      · ${o.cost.toFixed(3)} · {fmtK(o.tokens)} tok
                    </>
                  )}
                </div>
              </button>
            ))}
          </div>
        ))}
      </div>

      {/* Middle: dot grid */}
      <div className="flex-1 min-w-0 overflow-auto border rounded-md p-2">
        {op && (
          <>
            <div className="mb-2 flex items-baseline gap-2">
              <span className="font-semibold">
                {op.type}:{op.name.split("/").pop()}
              </span>
              {op.total != null && (
                <span className="text-xs text-gray-500">
                  {op.completed.toLocaleString()}/{op.total.toLocaleString()}
                </span>
              )}
              <span
                className="text-xs font-medium"
                style={{ color: STATUS_COLOR[op.status] }}
              >
                {op.status}
              </span>
            </div>
            <DocGrid
              op={op}
              concurrency={state.concurrency}
              selDoc={selDoc}
              onSelect={setSelDoc}
            />
          </>
        )}
      </div>

      {/* Right: detail */}
      <div className="w-80 shrink-0 overflow-auto border rounded-md p-2">
        {sample ? (
          <DocDetail doc={sample} opName={op!.name} index={selDoc!} />
        ) : op ? (
          <OpDetail op={op} />
        ) : null}
      </div>
    </div>
  );
});

function OpDetail({ op }: { op: OpState }) {
  const rows: [string, string][] = [
    ["step", op.step],
    ["model", op.model ?? "—"],
    ["status", op.status],
    ["docs", op.total != null ? `${op.completed}/${op.total}` : "—"],
    ["output", op.out_count != null ? `${op.out_count}` : "—"],
    ["errors", `${op.errors}`],
    ["cost", `$${op.cost.toFixed(4)}`],
    ["tokens", op.tokens.toLocaleString()],
    ["elapsed", fmtDur(op.elapsed)],
  ];
  return (
    <div>
      <div className="font-semibold mb-2">
        {op.type}:{op.name.split("/").pop()}
      </div>
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between text-xs py-0.5">
          <span className="text-gray-500">{k}</span>
          <span className="font-mono">{v}</span>
        </div>
      ))}
      <div className="text-[11px] text-gray-400 mt-3">
        Click a dot to inspect that document.
      </div>
    </div>
  );
}

function DocDetail({
  doc,
  opName,
  index,
}: {
  doc: Record<string, unknown>;
  opName: string;
  index: number;
}) {
  const obsKey = `_observability_${opName.split("/").pop()}`;
  const prompt =
    doc[obsKey] && typeof doc[obsKey] === "object"
      ? (doc[obsKey] as { prompt?: string }).prompt
      : undefined;
  const display = Object.fromEntries(
    Object.entries(doc).filter(([k]) => !k.startsWith("_observability_"))
  );
  return (
    <div>
      <div className="font-semibold mb-2">Document #{index}</div>
      <div className="text-[11px] uppercase tracking-wide text-gray-500">
        output
      </div>
      <pre className="text-[11px] whitespace-pre-wrap font-mono bg-gray-50 rounded p-2 overflow-auto">
        {JSON.stringify(display, null, 2)}
      </pre>
      {prompt && (
        <>
          <div className="text-[11px] uppercase tracking-wide text-gray-500 mt-2">
            prompt
          </div>
          <pre className="text-[11px] whitespace-pre-wrap font-mono text-gray-600 bg-gray-50 rounded p-2 overflow-auto">
            {prompt}
          </pre>
        </>
      )}
    </div>
  );
}
