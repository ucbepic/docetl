"use client";

import React, { useMemo, useState, useCallback, useEffect } from "react";
import Editor from "@monaco-editor/react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Info,
  Trash2,
  Plus,
  GripVertical,
  Copy,
  Sparkles,
  Settings,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  DragDropContext,
  Droppable,
  Draggable,
  type DropResult,
} from "react-beautiful-dnd";
import { v4 as uuidv4 } from "uuid";
import { useToast } from "@/hooks/use-toast";

// Types for the visualization grammar
interface BlockOp {
  type: string;
  label?: string;
  width?: number; // relative unit
  height?: number; // relative unit
  id?: string;
}

interface StackOp {
  type: "stack";
  operators: Array<BlockOp>;
  id?: string;
}

type Operator = BlockOp | StackOp;

interface PipelineConfig {
  pipelines: Array<{
    name: string;
    operators: Operator[];
  }>;
}

// Utilities for IDs
const ensureOperatorIds = (ops: Operator[]): Operator[] =>
  ops.map((op) => {
    if ((op as any).type === "stack") {
      const stack = op as StackOp;
      return {
        ...stack,
        id: (stack as any).id ?? uuidv4(),
        operators: stack.operators.map((b) => ({
          ...b,
          id: (b as any).id ?? uuidv4(),
        })),
      } as StackOp;
    }
    const b = op as BlockOp;
    return { ...b, id: (b as any).id ?? uuidv4() } as BlockOp;
  });

const ensureIds = (cfg: PipelineConfig): PipelineConfig => ({
  pipelines: cfg.pipelines.map((p) => ({
    ...p,
    operators: ensureOperatorIds(p.operators),
  })),
});

const DEFAULT_CONFIG: PipelineConfig = {
  pipelines: [
    {
      name: "Pipeline 1: Original (Powerful LLM)",
      operators: [
        {
          type: "llm-powerful",
          label: "Extract 3 Factors",
          width: 1.5,
          height: 1.0,
        },
      ],
    },
    {
      name: "Pipeline 2: Task Decomposition (Powerful LLM)",
      operators: [
        { type: "llm-powerful", label: "Factor 1", width: 0.5, height: 1.0 },
        { type: "llm-powerful", label: "Factor 2", width: 0.5, height: 1.0 },
        { type: "llm-powerful", label: "Factor 3", width: 0.5, height: 1.0 },
      ],
    },
    {
      name: "Pipeline 3: Model Substitution (Cheap LLM)",
      operators: [
        { type: "llm-cheap", label: "Factor 1", width: 0.5, height: 1.0 },
        { type: "llm-cheap", label: "Factor 2", width: 0.5, height: 1.0 },
        { type: "llm-cheap", label: "Factor 3", width: 0.5, height: 1.0 },
      ],
    },
    {
      name: "Pipeline 4: Code Trim + Decomp (Cheap LLM)",
      operators: [
        { type: "code", label: "Trim Doc", width: 0.5, height: 1.0 },
        { type: "llm-cheap", label: "Factor 1", width: 0.5, height: 0.5 },
        { type: "llm-cheap", label: "Factor 2", width: 0.5, height: 0.5 },
        { type: "llm-cheap", label: "Factor 3", width: 0.5, height: 0.5 },
      ],
    },
    {
      name: "Pipeline 5: Swap with a Cascade of Code and LLM",
      operators: [
        { type: "code", label: "Trim Doc", width: 0.5, height: 1.0 },
        {
          type: "stack",
          operators: [
            { type: "code", label: "Regex (80%)", width: 0.75, height: 0.4 },
            {
              type: "llm-powerful",
              label: "Contextual (20%)",
              width: 0.75,
              height: 0.1,
            },
          ],
        },
        { type: "llm-cheap", label: "Factor 2", width: 0.5, height: 0.5 },
        { type: "llm-cheap", label: "Factor 3", width: 0.5, height: 0.5 },
      ],
    },
  ],
};

const DEFAULT_CONFIG_TEXT = JSON.stringify(DEFAULT_CONFIG, null, 2);

// Color mapping for academic paper visualization (professional, print-friendly)
const TYPE_CLASS: Record<string, string> = {
  "llm-powerful":
    "bg-violet-200 text-violet-950 border-violet-500 dark:bg-violet-800 dark:text-violet-100 dark:border-violet-500",
  "llm-cheap":
    "bg-violet-50 text-violet-800 border-violet-300 dark:bg-violet-900 dark:text-violet-200 dark:border-violet-600",
  code: "bg-slate-100 text-slate-900 border-slate-400 dark:bg-slate-800 dark:text-slate-100 dark:border-slate-600",
  default:
    "bg-red-50 text-red-900 border-red-300 dark:bg-red-900 dark:text-red-100 dark:border-red-700",
};

const getLabelType = (type: string): string =>
  type.startsWith("llm-") ? "LLM" : type.toUpperCase();

export default function VisualizationBuilder(): JSX.Element {
  const [configText, setConfigText] = useState<string>(DEFAULT_CONFIG_TEXT);
  const [error, setError] = useState<string>("");
  const [baseWidthRem, setBaseWidthRem] = useState<number>(8);
  const [baseHeightRem, setBaseHeightRem] = useState<number>(5);
  const [parsed, setParsed] = useState<PipelineConfig>(
    ensureIds(DEFAULT_CONFIG)
  );
  // Fix for react-beautiful-dnd SSR issues with Next.js
  const [isMounted, setIsMounted] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Parse config safely
  const parseConfig = useCallback((text: string) => {
    try {
      const obj = JSON.parse(text);
      if (!obj || !Array.isArray(obj.pipelines)) {
        throw new Error("Config must contain a 'pipelines' array");
      }
      setParsed(ensureIds(obj));
      setError("");
    } catch (e: any) {
      setError(e.message || "Invalid JSON");
    }
  }, []);

  useEffect(() => {
    parseConfig(configText);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleEditorChange = (value?: string) => {
    const text = value ?? "";
    setConfigText(text);
    parseConfig(text);
  };

  const computeMaxHeightRem = useCallback(
    (ops: Operator[]): number => {
      let maxRem = 0;
      for (const op of ops) {
        if ((op as StackOp).type === "stack") {
          const stack = op as StackOp;
          const total = stack.operators.reduce(
            (sum, child) => sum + (child.height || 1) * baseHeightRem,
            0
          );
          if (total > maxRem) maxRem = total;
        } else {
          const h = ((op as BlockOp).height || 1) * baseHeightRem;
          if (h > maxRem) maxRem = h;
        }
      }
      return maxRem;
    },
    [baseHeightRem]
  );

  const renderBlock = (
    op: BlockOp,
    key?: string | number,
    isInStack = false
  ) => {
    const widthRem = (op.width || 1) * baseWidthRem;
    const heightRem = (op.height || 1) * baseHeightRem;
    const color = TYPE_CLASS[op.type] || TYPE_CLASS.default;
    const isLLM = op.type.startsWith("llm-");
    const Icon = isLLM ? Sparkles : Settings;

    // Scale font size and icon based on block size
    const isSmall = widthRem < 6 || heightRem < 4;
    const isVerySmall = widthRem < 4 || heightRem < 3;
    const iconSize = isVerySmall ? 10 : isSmall ? 12 : 16;

    // Reduce padding for blocks in stacks to minimize extra height
    const paddingClass = isInStack ? "px-1 py-0.5" : "px-2 py-1";

    return (
      <div
        key={key}
        className={`flex flex-col justify-center items-center text-center rounded-md shadow-sm border ${paddingClass} ${color}`}
        style={{ width: `${widthRem}rem`, height: `${heightRem}rem` }}
      >
        <Icon className={`${!isVerySmall ? "mb-1" : ""}`} size={iconSize} />
        <div
          className={`${
            isVerySmall
              ? "text-[0.5rem]"
              : isSmall
              ? "text-[0.625rem]"
              : "text-sm"
          } leading-tight overflow-hidden`}
          style={{ maxWidth: "100%", wordWrap: "break-word" }}
        >
          {op.label || ""}
        </div>
      </div>
    );
  };

  const renderOperator = (op: Operator, idx: number) => {
    if ((op as StackOp).type === "stack") {
      const stack = op as StackOp;
      // Compute stack width = max child width
      const stackMaxWidthRem = stack.operators.reduce(
        (max, child) => Math.max(max, (child.width || 1) * baseWidthRem),
        0
      );
      // Stack height is the sum of child heights (they should sum to the parent's data size)
      // No gap between children since they represent splits of the same data volume
      return (
        <div
          key={`stack-${idx}`}
          className="flex flex-col"
          style={{ width: `${stackMaxWidthRem}rem` }}
        >
          {stack.operators.map((child, cIdx) =>
            renderBlock(child, `stack-item-${idx}-${cIdx}`, true)
          )}
        </div>
      );
    }
    return renderBlock(op as BlockOp, idx, false);
  };

  const pipelines = useMemo(() => parsed.pipelines ?? [], [parsed]);
  // --- Helpers for visual editor (single source of truth is `parsed`) ---
  const setConfig = useCallback((next: PipelineConfig) => {
    setParsed(next);
    setConfigText(JSON.stringify(next, null, 2));
    setError("");
  }, []);

  const newDefaultBlock = useCallback(
    (): BlockOp => ({
      id: uuidv4(),
      type: "llm-cheap",
      label: "New Block",
      width: 0.5,
      height: 1.0,
    }),
    []
  );

  const addPipeline = useCallback(() => {
    const next: PipelineConfig = {
      pipelines: [
        ...pipelines,
        { name: `New Pipeline ${pipelines.length + 1}`, operators: [] },
      ],
    };
    setConfig(next);
  }, [pipelines, setConfig]);

  const removePipeline = useCallback(
    (pIdx: number) => {
      const next: PipelineConfig = {
        pipelines: pipelines.filter((_, i) => i !== pIdx),
      };
      setConfig(next);
    },
    [pipelines, setConfig]
  );

  const updatePipelineName = useCallback(
    (pIdx: number, name: string) => {
      const next: PipelineConfig = {
        pipelines: pipelines.map((p, i) => (i === pIdx ? { ...p, name } : p)),
      };
      setConfig(next);
    },
    [pipelines, setConfig]
  );

  const addBlock = useCallback(
    (pIdx: number) => {
      const nextPipes = pipelines.map((p, i) =>
        i === pIdx
          ? { ...p, operators: [...p.operators, newDefaultBlock()] }
          : p
      );
      setConfig({ pipelines: nextPipes });
    },
    [pipelines, newDefaultBlock, setConfig]
  );

  const removeBlock = useCallback(
    (pIdx: number, bIdx: number) => {
      const nextPipes = pipelines.map((p, i) =>
        i === pIdx
          ? { ...p, operators: p.operators.filter((_, j) => j !== bIdx) }
          : p
      );
      setConfig({ pipelines: nextPipes });
    },
    [pipelines, setConfig]
  );

  const updateBlockField = useCallback(
    (pIdx: number, bIdx: number, field: keyof BlockOp | "type", value: any) => {
      const next = pipelines.map((p, i) => {
        if (i !== pIdx) return p;
        const op = p.operators[bIdx] as Operator;
        let newOp: Operator = op;
        if (field === "type" && value === "stack") {
          newOp = { id: uuidv4(), type: "stack", operators: [] } as StackOp;
        } else if (field === "type" && value !== "stack") {
          // convert stack -> block
          newOp = { ...newDefaultBlock(), type: String(value) } as BlockOp;
        } else if ((op as StackOp).type === "stack") {
          // stacks do not have label/width/height directly
          newOp = op;
        } else {
          newOp = { ...(op as BlockOp), [field]: value } as BlockOp;
        }
        const ops = p.operators.map((o, j) => (j === bIdx ? newOp : o));
        return { ...p, operators: ops };
      });
      setConfig({ pipelines: next });
    },
    [pipelines, setConfig, newDefaultBlock]
  );

  // Stack children helpers
  const addSubBlock = useCallback(
    (pIdx: number, bIdx: number) => {
      const next = pipelines.map((p, i) => {
        if (i !== pIdx) return p;
        const op = p.operators[bIdx] as StackOp;
        if (op.type !== "stack") return p;
        const children = [...op.operators, newDefaultBlock()];
        const newStack: StackOp = { ...op, operators: children };
        const ops = p.operators.map((o, j) => (j === bIdx ? newStack : o));
        return { ...p, operators: ops };
      });
      setConfig({ pipelines: next });
    },
    [pipelines, setConfig, newDefaultBlock]
  );

  const updateSubBlockField = useCallback(
    (
      pIdx: number,
      bIdx: number,
      sIdx: number,
      field: keyof BlockOp | "type",
      value: any
    ) => {
      const next = pipelines.map((p, i) => {
        if (i !== pIdx) return p;
        const op = p.operators[bIdx] as StackOp;
        if (op.type !== "stack") return p;
        const child = op.operators[sIdx];
        let updated: BlockOp = child;
        if (field === "type" && value === "stack") {
          // Prevent nesting stacks for simplicity
          updated = { ...child, type: "code" };
        } else {
          updated = { ...child, [field]: value } as BlockOp;
        }
        const children = op.operators.map((c, k) => (k === sIdx ? updated : c));
        const newStack: StackOp = { ...op, operators: children };
        const ops = p.operators.map((o, j) => (j === bIdx ? newStack : o));
        return { ...p, operators: ops };
      });
      setConfig({ pipelines: next });
    },
    [pipelines, setConfig]
  );

  const removeSubBlock = useCallback(
    (pIdx: number, bIdx: number, sIdx: number) => {
      const next = pipelines.map((p, i) => {
        if (i !== pIdx) return p;
        const op = p.operators[bIdx] as StackOp;
        if (op.type !== "stack") return p;
        const children = op.operators.filter((_, k) => k !== sIdx);
        const newStack: StackOp = { ...op, operators: children };
        const ops = p.operators.map((o, j) => (j === bIdx ? newStack : o));
        return { ...p, operators: ops };
      });
      setConfig({ pipelines: next });
    },
    [pipelines, setConfig]
  );

  // Copy pipeline as SVG
  const copyPipelineAsSVG = useCallback(
    (pipelineIdx: number) => {
      const pipeline = pipelines[pipelineIdx];
      if (!pipeline) return;

      // Calculate dimensions
      const maxHeight = computeMaxHeightRem(pipeline.operators);
      const totalWidth = pipeline.operators.reduce((sum, op) => {
        if ((op as any).type === "stack") {
          const stack = op as any;
          return (
            sum +
            Math.max(
              ...stack.operators.map((c: any) => (c.width || 1) * baseWidthRem)
            )
          );
        }
        return sum + ((op as any).width || 1) * baseWidthRem;
      }, 0); // No gap between operators - they should be flush

      const remToPx = 16; // 1rem = 16px
      const svgWidth = totalWidth * remToPx;
      const svgHeight = maxHeight * remToPx;
      const padding = 8;

      // Helper to create block SVG
      const createBlockSVG = (op: BlockOp, x: number, y: number): string => {
        const widthPx = (op.width || 1) * baseWidthRem * remToPx;
        const heightPx = (op.height || 1) * baseHeightRem * remToPx;
        const color =
          op.type === "llm-powerful"
            ? "#ddd6fe"
            : op.type === "llm-cheap"
            ? "#f3e8ff"
            : "#e2e8f0";
        const borderColor =
          op.type === "llm-powerful"
            ? "#a78bfa"
            : op.type === "llm-cheap"
            ? "#d8b4fe"
            : "#94a3b8";
        const textColor =
          op.type === "llm-powerful"
            ? "#3b0764"
            : op.type === "llm-cheap"
            ? "#6b21a8"
            : "#334155";
        const isLLM = op.type.startsWith("llm-");

        // Adaptive sizing based on block dimensions
        const isTiny = heightPx < 16;
        const isSmallBlock = heightPx < 40 || widthPx < 80;

        const fontSize = isTiny ? 7 : isSmallBlock ? 9 : 12;
        const iconFontSize = isTiny ? 10 : isSmallBlock ? 14 : 18;

        // Use Unicode symbols for icons
        const iconSymbol = isLLM ? "✨" : "⚙️";
        const centerX = x + widthPx / 2;
        const centerY = y + heightPx / 2;

        // Position icon and label
        const hasLabel = op.label && op.label.length > 0;
        const spacing = isTiny ? 2 : isSmallBlock ? 4 : 8;
        const iconY = hasLabel && !isTiny ? centerY - spacing : centerY + 6;
        const labelY = hasLabel ? centerY + spacing + fontSize / 2 : centerY;

        let content = `\n    <rect x="${x}" y="${y}" width="${widthPx}" height="${heightPx}" fill="${color}" stroke="${borderColor}" stroke-width="2" rx="6"/>`;

        if (!isTiny) {
          content += `\n    <text x="${centerX}" y="${iconY}" text-anchor="middle" font-size="${iconFontSize}" fill="${textColor}" dominant-baseline="middle">${iconSymbol}</text>`;
        }

        if (hasLabel && heightPx >= 12) {
          const maxChars = Math.floor(widthPx / (fontSize * 0.5));
          const label =
            op.label.length > maxChars
              ? op.label.substring(0, maxChars - 3) + "..."
              : op.label;
          content += `\n    <text x="${centerX}" y="${labelY}" text-anchor="middle" font-size="${fontSize}" fill="${textColor}" font-family="system-ui, -apple-system, sans-serif">${label}</text>`;
        }

        return content;
      };

      let svgContent = `<svg xmlns="http://www.w3.org/2000/svg" width="${
        svgWidth + padding * 2
      }" height="${
        svgHeight + padding * 2
      }">\n  <rect width="100%" height="100%" fill="white"/>`;

      let currentX = padding;
      pipeline.operators.forEach((op) => {
        if ((op as any).type === "stack") {
          const stack = op as any;
          const stackWidth =
            Math.max(
              ...stack.operators.map((c: any) => (c.width || 1) * baseWidthRem)
            ) * remToPx;
          const stackHeight = stack.operators.reduce(
            (sum: number, c: any) =>
              sum + (c.height || 1) * baseHeightRem * remToPx,
            0
          );
          // Align stack to bottom like items-end
          let stackY = padding + (svgHeight - stackHeight);

          stack.operators.forEach((child: any) => {
            const childWidthPx = (child.width || 1) * baseWidthRem * remToPx;
            const childX = currentX + (stackWidth - childWidthPx) / 2; // Center within stack width
            svgContent += createBlockSVG(child, childX, stackY);
            stackY += (child.height || 1) * baseHeightRem * remToPx;
          });

          currentX += stackWidth;
        } else {
          const block = op as BlockOp;
          const blockHeight = (block.height || 1) * baseHeightRem * remToPx;
          // Align to bottom like items-end
          const yPos = padding + (svgHeight - blockHeight);
          svgContent += createBlockSVG(block, currentX, yPos);
          currentX += (block.width || 1) * baseWidthRem * remToPx;
        }
      });

      svgContent += `\n</svg>`;

      // Copy to clipboard
      navigator.clipboard
        .writeText(svgContent)
        .then(() => {
          toast({
            title: "SVG Copied",
            description: "Pipeline visualization copied to clipboard as SVG",
          });
        })
        .catch(() => {
          toast({
            title: "Copy Failed",
            description: "Failed to copy SVG to clipboard",
            variant: "destructive",
          });
        });
    },
    [pipelines, baseWidthRem, baseHeightRem, computeMaxHeightRem, toast]
  );

  // Drag-and-drop handlers
  const onDragEnd = useCallback(
    (result: DropResult) => {
      const { source, destination } = result;
      if (!destination) return;

      // Pipeline-level reordering and cross-pipeline moves
      if (
        source.droppableId.startsWith("pipeline-") &&
        destination.droppableId.startsWith("pipeline-")
      ) {
        const srcP = parseInt(source.droppableId.split("-")[1] || "0", 10);
        const dstP = parseInt(destination.droppableId.split("-")[1] || "0", 10);
        const next = pipelines.map((p) => ({ ...p }));
        const [moved] = next[srcP].operators.splice(source.index, 1);
        next[dstP].operators.splice(destination.index, 0, moved as Operator);
        setConfig({ pipelines: next });
        return;
      }

      // Stack sub-block reordering
      if (
        source.droppableId.startsWith("stack-") &&
        destination.droppableId.startsWith("stack-")
      ) {
        const srcParts = source.droppableId.split("-");
        const dstParts = destination.droppableId.split("-");
        const pIdx = parseInt(srcParts[1] || "0", 10);
        const bIdx = parseInt(srcParts[2] || "0", 10);
        const dpIdx = parseInt(dstParts[1] || "0", 10);
        const dbIdx = parseInt(dstParts[2] || "0", 10);
        const next = pipelines.map((p) => ({ ...p }));
        const srcStack = next[pIdx].operators[bIdx] as StackOp;
        const [moved] = srcStack.operators.splice(source.index, 1);
        const dstStack = next[dpIdx].operators[dbIdx] as StackOp;
        dstStack.operators.splice(destination.index, 0, moved);
        setConfig({ pipelines: next });
      }
    },
    [pipelines, setConfig]
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
            Pipeline Visualization Builder
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Visualize DocETL pipelines. Height = data size; Width = task
            complexity; Color/label = computation type.
          </p>
        </div>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" className="gap-2">
              <Info className="h-4 w-4" /> Info & Settings
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Visualization Grammar & Settings</DialogTitle>
            </DialogHeader>
            <div className="space-y-6">
              <div>
                <h4 className="text-sm font-semibold mb-2">
                  Visualization Grammar Tutorial
                </h4>
                <div className="text-sm text-muted-foreground space-y-3">
                  <p>
                    Create visual representations of DocETL pipelines using a
                    simple JSON structure. Each <b>pipeline</b> contains an
                    array of <b>operators</b> arranged left-to-right.
                  </p>

                  <div>
                    <p className="font-medium text-foreground mb-1">
                      Block Properties:
                    </p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>
                        <code>type</code>: The computation type (
                        <code>llm-powerful</code>, <code>llm-cheap</code>, or{" "}
                        <code>code</code>)
                      </li>
                      <li>
                        <code>label</code>: Descriptive text shown in the block
                      </li>
                      <li>
                        <code>height</code>: Represents relative data size
                        (e.g., 1.0 for full document, 0.5 for trimmed)
                      </li>
                      <li>
                        <code>width</code>: Represents task complexity (e.g.,
                        0.5 for simple factor extraction)
                      </li>
                    </ul>
                  </div>

                  <div>
                    <p className="font-medium text-foreground mb-1">
                      Block Types & Colors:
                    </p>
                    <div className="grid grid-cols-3 gap-2 pl-5">
                      <div className="text-xs">Powerful LLM</div>
                      <div
                        className={`h-8 rounded border ${TYPE_CLASS["llm-powerful"]}`}
                      />
                      <div className="text-xs font-mono">llm-powerful</div>
                      <div className="text-xs">Cheap LLM</div>
                      <div
                        className={`h-8 rounded border ${TYPE_CLASS["llm-cheap"]}`}
                      />
                      <div className="text-xs font-mono">llm-cheap</div>
                      <div className="text-xs">Code/Transform</div>
                      <div
                        className={`h-8 rounded border ${TYPE_CLASS["code"]}`}
                      />
                      <div className="text-xs font-mono">code</div>
                    </div>
                  </div>

                  <div>
                    <p className="font-medium text-foreground mb-1">
                      Stack Operators:
                    </p>
                    <p className="pl-5">
                      Use{" "}
                      <code className="bg-muted px-1 py-0.5 rounded">
                        type: &quot;stack&quot;
                      </code>{" "}
                      to visualize dynamic routing where different inputs take
                      different paths in the same step. Each child&apos;s height
                      represents the proportion of data taking that path (e.g.,
                      0.8 means 80% of inputs). The stack&apos;s total height
                      equals the sum of child heights, matching the data volume
                      from the previous step.
                    </p>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="baseWidth">Base Width (rem)</Label>
                  <Input
                    id="baseWidth"
                    type="number"
                    step="0.5"
                    value={baseWidthRem}
                    onChange={(e) =>
                      setBaseWidthRem(parseFloat(e.target.value) || 0)
                    }
                  />
                </div>
                <div>
                  <Label htmlFor="baseHeight">Base Height (rem)</Label>
                  <Input
                    id="baseHeight"
                    type="number"
                    step="0.5"
                    value={baseHeightRem}
                    onChange={(e) =>
                      setBaseHeightRem(parseFloat(e.target.value) || 0)
                    }
                  />
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="h-full">
          <CardHeader className="pb-3">
            <CardTitle className="text-xl">Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="visual" className="w-full">
              <TabsList>
                <TabsTrigger value="visual">Visual Editor</TabsTrigger>
                <TabsTrigger value="json">JSON</TabsTrigger>
              </TabsList>
              <TabsContent value="visual" className="mt-3">
                {!isMounted ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full" />
                  </div>
                ) : (
                  <DragDropContext onDragEnd={onDragEnd}>
                    <div className="space-y-4">
                      {pipelines.map((pipeline, pIdx) => (
                        <Card key={`p-card-${pIdx}`} className="border-border">
                          <CardHeader className="pb-2">
                            <div className="flex items-center justify-between gap-2">
                              <Input
                                value={pipeline.name}
                                onChange={(e) =>
                                  updatePipelineName(pIdx, e.target.value)
                                }
                                className="h-8 text-sm font-semibold"
                              />
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => removePipeline(pIdx)}
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-2">
                            {pipeline.operators.length === 0 && (
                              <p className="text-xs text-muted-foreground">
                                No blocks yet. Add your first block.
                              </p>
                            )}
                            <Droppable droppableId={`pipeline-${pIdx}`}>
                              {(provided) => (
                                <div
                                  ref={provided.innerRef}
                                  {...provided.droppableProps}
                                  className="space-y-2"
                                >
                                  {pipeline.operators.map((op, bIdx) => {
                                    const isStack =
                                      (op as any).type === "stack";
                                    const draggableId = String(
                                      (op as any).id ?? `${pIdx}-${bIdx}`
                                    );
                                    return (
                                      <Draggable
                                        key={draggableId}
                                        draggableId={draggableId}
                                        index={bIdx}
                                      >
                                        {(dragProvided) => (
                                          <div
                                            ref={dragProvided.innerRef}
                                            {...dragProvided.draggableProps}
                                            className="rounded-md bg-muted/40 border border-border/50 p-3 space-y-2"
                                          >
                                            <div className="flex flex-wrap items-end gap-2">
                                              <div
                                                className="text-muted-foreground cursor-grab"
                                                {...dragProvided.dragHandleProps}
                                              >
                                                <GripVertical className="h-4 w-4" />
                                              </div>
                                              <div className="w-36">
                                                <Label className="text-xs">
                                                  Type
                                                </Label>
                                                <Select
                                                  value={(op as any).type}
                                                  onValueChange={(v) =>
                                                    updateBlockField(
                                                      pIdx,
                                                      bIdx,
                                                      "type",
                                                      v
                                                    )
                                                  }
                                                >
                                                  <SelectTrigger className="h-8 text-sm">
                                                    <SelectValue />
                                                  </SelectTrigger>
                                                  <SelectContent>
                                                    <SelectItem value="llm-powerful">
                                                      llm-powerful
                                                    </SelectItem>
                                                    <SelectItem value="llm-cheap">
                                                      llm-cheap
                                                    </SelectItem>
                                                    <SelectItem value="code">
                                                      code
                                                    </SelectItem>
                                                    <SelectItem value="stack">
                                                      stack
                                                    </SelectItem>
                                                  </SelectContent>
                                                </Select>
                                              </div>
                                              {!isStack && (
                                                <div className="flex-1 min-w-[160px]">
                                                  <Label className="text-xs">
                                                    Label
                                                  </Label>
                                                  <Input
                                                    className="h-8 text-sm"
                                                    value={
                                                      (op as any).label || ""
                                                    }
                                                    onChange={(e) =>
                                                      updateBlockField(
                                                        pIdx,
                                                        bIdx,
                                                        "label",
                                                        e.target.value
                                                      )
                                                    }
                                                  />
                                                </div>
                                              )}
                                              {!isStack && (
                                                <div className="w-28">
                                                  <Label className="text-xs">
                                                    Width
                                                  </Label>
                                                  <Input
                                                    type="number"
                                                    step="0.25"
                                                    min="0"
                                                    className={`h-8 text-sm ${
                                                      (op as any).width &&
                                                      (op as any).width! <= 0
                                                        ? "border-destructive"
                                                        : ""
                                                    }`}
                                                    value={String(
                                                      (op as any).width ?? 1
                                                    )}
                                                    onChange={(e) =>
                                                      updateBlockField(
                                                        pIdx,
                                                        bIdx,
                                                        "width",
                                                        parseFloat(
                                                          e.target.value
                                                        ) || 0
                                                      )
                                                    }
                                                  />
                                                </div>
                                              )}
                                              {!isStack && (
                                                <div className="w-28">
                                                  <Label className="text-xs">
                                                    Height
                                                  </Label>
                                                  <Input
                                                    type="number"
                                                    step="0.25"
                                                    min="0"
                                                    className={`h-8 text-sm ${
                                                      (op as any).height &&
                                                      (op as any).height! <= 0
                                                        ? "border-destructive"
                                                        : ""
                                                    }`}
                                                    value={String(
                                                      (op as any).height ?? 1
                                                    )}
                                                    onChange={(e) =>
                                                      updateBlockField(
                                                        pIdx,
                                                        bIdx,
                                                        "height",
                                                        parseFloat(
                                                          e.target.value
                                                        ) || 0
                                                      )
                                                    }
                                                  />
                                                </div>
                                              )}
                                              <div className="ml-auto flex items-center gap-1">
                                                <Button
                                                  variant="ghost"
                                                  size="icon"
                                                  onClick={() =>
                                                    removeBlock(pIdx, bIdx)
                                                  }
                                                >
                                                  <Trash2 className="h-4 w-4" />
                                                </Button>
                                              </div>
                                            </div>

                                            {isStack &&
                                              (() => {
                                                const stack = op as any;
                                                return (
                                                  <div className="mt-1 space-y-2">
                                                    <div className="text-xs text-muted-foreground">
                                                      Stack paths (vertical
                                                      split):
                                                    </div>
                                                    <Droppable
                                                      droppableId={`stack-${pIdx}-${bIdx}`}
                                                    >
                                                      {(subProvided) => (
                                                        <div
                                                          ref={
                                                            subProvided.innerRef
                                                          }
                                                          {...subProvided.droppableProps}
                                                          className="space-y-2"
                                                        >
                                                          {(
                                                            stack.operators ||
                                                            []
                                                          ).map(
                                                            (
                                                              child: any,
                                                              sIdx: number
                                                            ) => {
                                                              const childId =
                                                                String(
                                                                  child.id ??
                                                                    `${pIdx}-${bIdx}-${sIdx}`
                                                                );
                                                              return (
                                                                <Draggable
                                                                  key={childId}
                                                                  draggableId={
                                                                    childId
                                                                  }
                                                                  index={sIdx}
                                                                >
                                                                  {(
                                                                    subDragProvided
                                                                  ) => (
                                                                    <div
                                                                      ref={
                                                                        subDragProvided.innerRef
                                                                      }
                                                                      {...subDragProvided.draggableProps}
                                                                      className="rounded-sm bg-muted/20 border border-border/40 p-2"
                                                                    >
                                                                      <div className="flex flex-wrap items-end gap-2">
                                                                        <div
                                                                          className="text-muted-foreground cursor-grab"
                                                                          {...subDragProvided.dragHandleProps}
                                                                        >
                                                                          <GripVertical className="h-4 w-4" />
                                                                        </div>
                                                                        <div className="w-36">
                                                                          <Label className="text-xs">
                                                                            Type
                                                                          </Label>
                                                                          <Select
                                                                            value={
                                                                              child.type
                                                                            }
                                                                            onValueChange={(
                                                                              v
                                                                            ) =>
                                                                              updateSubBlockField(
                                                                                pIdx,
                                                                                bIdx,
                                                                                sIdx,
                                                                                "type",
                                                                                v
                                                                              )
                                                                            }
                                                                          >
                                                                            <SelectTrigger className="h-8 text-sm">
                                                                              <SelectValue />
                                                                            </SelectTrigger>
                                                                            <SelectContent>
                                                                              <SelectItem value="llm-powerful">
                                                                                llm-powerful
                                                                              </SelectItem>
                                                                              <SelectItem value="llm-cheap">
                                                                                llm-cheap
                                                                              </SelectItem>
                                                                              <SelectItem value="code">
                                                                                code
                                                                              </SelectItem>
                                                                            </SelectContent>
                                                                          </Select>
                                                                        </div>
                                                                        <div className="flex-1 min-w-[160px]">
                                                                          <Label className="text-xs">
                                                                            Label
                                                                          </Label>
                                                                          <Input
                                                                            className="h-8 text-sm"
                                                                            value={
                                                                              child.label ||
                                                                              ""
                                                                            }
                                                                            onChange={(
                                                                              e
                                                                            ) =>
                                                                              updateSubBlockField(
                                                                                pIdx,
                                                                                bIdx,
                                                                                sIdx,
                                                                                "label",
                                                                                e
                                                                                  .target
                                                                                  .value
                                                                              )
                                                                            }
                                                                          />
                                                                        </div>
                                                                        <div className="w-28">
                                                                          <Label className="text-xs">
                                                                            Width
                                                                          </Label>
                                                                          <Input
                                                                            type="number"
                                                                            step="0.25"
                                                                            min="0"
                                                                            className={`h-8 text-sm ${
                                                                              child.width &&
                                                                              child.width <=
                                                                                0
                                                                                ? "border-destructive"
                                                                                : ""
                                                                            }`}
                                                                            value={String(
                                                                              child.width ??
                                                                                1
                                                                            )}
                                                                            onChange={(
                                                                              e
                                                                            ) =>
                                                                              updateSubBlockField(
                                                                                pIdx,
                                                                                bIdx,
                                                                                sIdx,
                                                                                "width",
                                                                                parseFloat(
                                                                                  e
                                                                                    .target
                                                                                    .value
                                                                                ) ||
                                                                                  0
                                                                              )
                                                                            }
                                                                          />
                                                                        </div>
                                                                        <div className="w-28">
                                                                          <Label className="text-xs">
                                                                            Height
                                                                          </Label>
                                                                          <Input
                                                                            type="number"
                                                                            step="0.25"
                                                                            min="0"
                                                                            className={`h-8 text-sm ${
                                                                              child.height &&
                                                                              child.height <=
                                                                                0
                                                                                ? "border-destructive"
                                                                                : ""
                                                                            }`}
                                                                            value={String(
                                                                              child.height ??
                                                                                1
                                                                            )}
                                                                            onChange={(
                                                                              e
                                                                            ) =>
                                                                              updateSubBlockField(
                                                                                pIdx,
                                                                                bIdx,
                                                                                sIdx,
                                                                                "height",
                                                                                parseFloat(
                                                                                  e
                                                                                    .target
                                                                                    .value
                                                                                ) ||
                                                                                  0
                                                                              )
                                                                            }
                                                                          />
                                                                        </div>
                                                                        <div className="ml-auto">
                                                                          <Button
                                                                            variant="ghost"
                                                                            size="icon"
                                                                            onClick={() =>
                                                                              removeSubBlock(
                                                                                pIdx,
                                                                                bIdx,
                                                                                sIdx
                                                                              )
                                                                            }
                                                                          >
                                                                            <Trash2 className="h-4 w-4" />
                                                                          </Button>
                                                                        </div>
                                                                      </div>
                                                                    </div>
                                                                  )}
                                                                </Draggable>
                                                              );
                                                            }
                                                          )}
                                                          {
                                                            subProvided.placeholder
                                                          }
                                                        </div>
                                                      )}
                                                    </Droppable>
                                                    <Button
                                                      variant="outline"
                                                      size="sm"
                                                      onClick={() =>
                                                        addSubBlock(pIdx, bIdx)
                                                      }
                                                    >
                                                      <Plus className="h-4 w-4 mr-1" />{" "}
                                                      Add Stack Path
                                                    </Button>
                                                  </div>
                                                );
                                              })()}
                                          </div>
                                        )}
                                      </Draggable>
                                    );
                                  })}
                                  {provided.placeholder}
                                </div>
                              )}
                            </Droppable>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => addBlock(pIdx)}
                            >
                              <Plus className="h-4 w-4 mr-1" /> Add Block
                            </Button>
                          </CardContent>
                        </Card>
                      ))}
                      <Button onClick={addPipeline}>
                        <Plus className="h-4 w-4 mr-1" /> Add Pipeline
                      </Button>
                    </div>
                  </DragDropContext>
                )}
              </TabsContent>
              <TabsContent value="json" className="mt-3">
                {error && (
                  <div className="mb-2 text-sm font-medium text-destructive">
                    Error: {error}
                  </div>
                )}
                <div className="border rounded-md overflow-hidden">
                  <Editor
                    height="600px"
                    defaultLanguage="json"
                    value={configText}
                    onChange={handleEditorChange}
                    options={{
                      minimap: { enabled: false },
                      lineNumbers: "on",
                      scrollBeyondLastLine: false,
                      wordWrap: "on",
                      wrappingIndent: "indent",
                      automaticLayout: true,
                      tabSize: 2,
                      fontSize: 14,
                    }}
                  />
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        <Card className="h-full">
          <CardHeader className="pb-3">
            <CardTitle className="text-xl">Live Visualization</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-8">
              {pipelines.map((pipeline, pIdx) => {
                const minHeightRem = computeMaxHeightRem(pipeline.operators);
                return (
                  <div key={pIdx}>
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-lg font-semibold">{pipeline.name}</h3>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => copyPipelineAsSVG(pIdx)}
                        className="gap-2"
                      >
                        <Copy className="h-4 w-4" />
                        Copy as SVG
                      </Button>
                    </div>
                    <div
                      className="w-full bg-background border rounded-md p-4 overflow-x-auto"
                      style={{ minHeight: `${minHeightRem}rem` }}
                    >
                      <div className="flex items-end">
                        {pipeline.operators.map((op, idx) =>
                          renderOperator(op, idx)
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
