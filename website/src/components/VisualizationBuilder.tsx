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
import { Info } from "lucide-react";

// Types for the visualization grammar
interface BlockOp {
  type: string;
  label?: string;
  width?: number; // relative unit
  height?: number; // relative unit
}

interface StackOp {
  type: "stack";
  operators: Array<BlockOp>;
}

type Operator = BlockOp | StackOp;

interface PipelineConfig {
  pipelines: Array<{
    name: string;
    operators: Operator[];
  }>;
}

const DEFAULT_CONFIG: PipelineConfig = {
  pipelines: [
    {
      name: "Pipeline 1: Original (Powerful LLM)",
      operators: [
        { type: "llm-powerful", label: "Extract 8 Factors", width: 2.0, height: 1.0 },
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
        { type: "code", label: "Trim Document", width: 0.75, height: 1.0 },
        { type: "llm-cheap", label: "Factor 1", width: 0.5, height: 0.5 },
        { type: "llm-cheap", label: "Factor 2", width: 0.5, height: 0.5 },
        { type: "llm-cheap", label: "Factor 3", width: 0.5, height: 0.5 },
      ],
    },
    {
      name: "Pipeline 5: Dynamic/Split Operator",
      operators: [
        { type: "code", label: "Trim Document", width: 0.75, height: 1.0 },
        {
          type: "stack",
          operators: [
            { type: "code", label: "Regex (80%)", width: 0.75, height: 0.4 },
            { type: "llm-powerful", label: "Contextual (20%)", width: 0.75, height: 0.4 },
          ],
        },
        { type: "llm-cheap", label: "Extract Other", width: 0.5, height: 0.5 },
      ],
    },
  ],
};

const DEFAULT_CONFIG_TEXT = JSON.stringify(DEFAULT_CONFIG, null, 2);

// Color mapping aligned with DocETL theme (light/dark friendly)
const TYPE_CLASS: Record<string, string> = {
  "llm-powerful": "bg-blue-100 text-blue-900 border-blue-300 dark:bg-blue-950 dark:text-blue-100 dark:border-blue-800",
  "llm-cheap": "bg-blue-50 text-blue-800 border-blue-200 dark:bg-blue-900 dark:text-blue-100 dark:border-blue-700",
  code: "bg-muted text-foreground border-border",
  default: "bg-red-50 text-red-900 border-red-300 dark:bg-red-900 dark:text-red-100 dark:border-red-700",
};

const getLabelType = (type: string): string => (type.startsWith("llm-") ? "LLM" : type.toUpperCase());

export default function VisualizationBuilder(): JSX.Element {
  const [configText, setConfigText] = useState<string>(DEFAULT_CONFIG_TEXT);
  const [error, setError] = useState<string>("");
  const [baseWidthRem, setBaseWidthRem] = useState<number>(8);
  const [baseHeightRem, setBaseHeightRem] = useState<number>(10);
  const [parsed, setParsed] = useState<PipelineConfig>(DEFAULT_CONFIG);

  // Parse config safely
  const parseConfig = useCallback((text: string) => {
    try {
      const obj = JSON.parse(text);
      if (!obj || !Array.isArray(obj.pipelines)) {
        throw new Error("Config must contain a 'pipelines' array");
      }
      setParsed(obj);
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
          const total = stack.operators.reduce((sum, child) => sum + (child.height || 1) * baseHeightRem, 0);
          if (total > maxRem) maxRem = total;
        } else {
          const h = ((op as BlockOp).height || 1) * baseHeightRem;
          if (h > maxRem) maxRem = h;
        }
      }
      return maxRem;
    },
    [baseHeightRem],
  );

  const renderBlock = (op: BlockOp, key?: string | number) => {
    const widthRem = (op.width || 1) * baseWidthRem;
    const heightRem = (op.height || 1) * baseHeightRem;
    const color = TYPE_CLASS[op.type] || TYPE_CLASS.default;
    const labelType = getLabelType(op.type);

    return (
      <div
        key={key}
        className={`flex flex-col justify-center items-center text-center rounded-md shadow-sm border px-3 py-2 ${color}`}
        style={{ width: `${widthRem}rem`, height: `${heightRem}rem` }}
      >
        <div className="text-[0.75rem] font-bold uppercase tracking-wide mb-1">{labelType}</div>
        <div className="text-sm leading-tight">{op.label || ""}</div>
      </div>
    );
  };

  const renderOperator = (op: Operator, idx: number) => {
    if ((op as StackOp).type === "stack") {
      const stack = op as StackOp;
      // Compute stack width = max child width
      const stackMaxWidthRem = stack.operators.reduce(
        (max, child) => Math.max(max, (child.width || 1) * baseWidthRem),
        0,
      );
      return (
        <div key={`stack-${idx}`} className="flex flex-col gap-1" style={{ width: `${stackMaxWidthRem}rem` }}>
          {stack.operators.map((child, cIdx) => renderBlock(child, `stack-item-${idx}-${cIdx}`))}
        </div>
      );
    }
    return renderBlock(op as BlockOp, idx);
  };

  const pipelines = useMemo(() => parsed.pipelines ?? [], [parsed]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Pipeline Visualization Builder</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Visualize DocETL pipelines. Height = data size; Width = task complexity; Color/label = computation type.
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
                <h4 className="text-sm font-semibold mb-2">Grammar</h4>
                <ul className="text-sm list-disc pl-5 space-y-1 text-muted-foreground">
                  <li>
                    <b>Block:</b> One pipeline operator (e.g., <code>map</code>, <code>code</code>).
                  </li>
                  <li>
                    <b>Height:</b> Data size (e.g., full vs. trimmed document).
                  </li>
                  <li>
                    <b>Width:</b> Task complexity (e.g., 1-factor vs. 8-factor).
                  </li>
                  <li>
                    <b>Left-to-Right:</b> Operator sequence.
                  </li>
                  <li>
                    <b>Color/Label:</b> Computation type: <code>llm-powerful</code>, <code>llm-cheap</code>, <code>code</code>.
                  </li>
                  <li>
                    <b>Vertical Stack (type: "stack"):</b> Dynamic routing in a single step.
                  </li>
                </ul>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="baseWidth">Base Width (rem)</Label>
                  <Input
                    id="baseWidth"
                    type="number"
                    step="0.5"
                    value={baseWidthRem}
                    onChange={(e) => setBaseWidthRem(parseFloat(e.target.value) || 0)}
                  />
                </div>
                <div>
                  <Label htmlFor="baseHeight">Base Height (rem)</Label>
                  <Input
                    id="baseHeight"
                    type="number"
                    step="0.5"
                    value={baseHeightRem}
                    onChange={(e) => setBaseHeightRem(parseFloat(e.target.value) || 0)}
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
            <CardTitle className="text-xl">Configuration (JSON)</CardTitle>
          </CardHeader>
          <CardContent>
            {error && <div className="mb-2 text-sm font-medium text-destructive">Error: {error}</div>}
            <div className="border rounded-md" style={{ minHeight: 360 }}>
              <Editor
                height="400px"
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
                    <h3 className="text-lg font-semibold mb-2">{pipeline.name}</h3>
                    <div
                      className="w-full bg-background border rounded-md p-4 overflow-x-auto"
                      style={{ minHeight: `${minHeightRem}rem` }}
                    >
                      <div className="flex items-end gap-2">
                        {pipeline.operators.map((op, idx) => renderOperator(op, idx))}
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
