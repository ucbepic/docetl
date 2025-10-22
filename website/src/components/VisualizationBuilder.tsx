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
import { Info, Trash2, Plus, ChevronLeft, ChevronRight } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

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
  // --- Helpers for visual editor (single source of truth is `parsed`) ---
  const setConfig = useCallback((next: PipelineConfig) => {
    setParsed(next);
    setConfigText(JSON.stringify(next, null, 2));
    setError("");
  }, []);

  const newDefaultBlock = useCallback(
    (): BlockOp => ({ type: "llm-cheap", label: "New Block", width: 0.5, height: 1.0 }),
    [],
  );

  const addPipeline = useCallback(() => {
    const next: PipelineConfig = {
      pipelines: [...pipelines, { name: `New Pipeline ${pipelines.length + 1}`, operators: [] }],
    };
    setConfig(next);
  }, [pipelines, setConfig]);

  const removePipeline = useCallback(
    (pIdx: number) => {
      const next: PipelineConfig = { pipelines: pipelines.filter((_, i) => i !== pIdx) };
      setConfig(next);
    },
    [pipelines, setConfig],
  );

  const updatePipelineName = useCallback(
    (pIdx: number, name: string) => {
      const next: PipelineConfig = {
        pipelines: pipelines.map((p, i) => (i === pIdx ? { ...p, name } : p)),
      };
      setConfig(next);
    },
    [pipelines, setConfig],
  );

  const addBlock = useCallback(
    (pIdx: number) => {
      const nextPipes = pipelines.map((p, i) => (i === pIdx ? { ...p, operators: [...p.operators, newDefaultBlock()] } : p));
      setConfig({ pipelines: nextPipes });
    },
    [pipelines, newDefaultBlock, setConfig],
  );

  const removeBlock = useCallback(
    (pIdx: number, bIdx: number) => {
      const nextPipes = pipelines.map((p, i) => (i === pIdx ? { ...p, operators: p.operators.filter((_, j) => j !== bIdx) } : p));
      setConfig({ pipelines: nextPipes });
    },
    [pipelines, setConfig],
  );

  const reorderBlock = useCallback(
    (pIdx: number, bIdx: number, dir: -1 | 1) => {
      const p = pipelines[pIdx];
      if (!p) return;
      const j = bIdx + dir;
      if (j < 0 || j >= p.operators.length) return;
      const ops = [...p.operators];
      [ops[bIdx], ops[j]] = [ops[j], ops[bIdx]];
      const next = pipelines.map((pp, i) => (i === pIdx ? { ...pp, operators: ops } : pp));
      setConfig({ pipelines: next });
    },
    [pipelines, setConfig],
  );

  const updateBlockField = useCallback(
    (pIdx: number, bIdx: number, field: keyof BlockOp | "type", value: any) => {
      const next = pipelines.map((p, i) => {
        if (i !== pIdx) return p;
        const op = p.operators[bIdx] as Operator;
        let newOp: Operator = op;
        if (field === "type" && value === "stack") {
          newOp = { type: "stack", operators: [] } as StackOp;
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
    [pipelines, setConfig, newDefaultBlock],
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
    [pipelines, setConfig, newDefaultBlock],
  );

  const updateSubBlockField = useCallback(
    (pIdx: number, bIdx: number, sIdx: number, field: keyof BlockOp | "type", value: any) => {
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
    [pipelines, setConfig],
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
    [pipelines, setConfig],
  );

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
                <div className="text-sm text-muted-foreground space-y-2">
                  <p>
                    A <b>pipeline</b> is a left-to-right sequence of <b>blocks</b> (operators). Each block’s <b>height</b> encodes
                    the data size at that step, and its <b>width</b> encodes task complexity. Color and label encode computation type.
                  </p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li><b>Block:</b> A single operator like <code>map</code>, <code>reduce</code>, or <code>code</code>.</li>
                    <li><b>Height:</b> Relative data size (e.g., 1.0 = full doc; 0.5 = trimmed).</li>
                    <li><b>Width:</b> Relative complexity (e.g., 0.5 = simple; 2.0 = complex or multi-factor).</li>
                    <li><b>Sequence:</b> Left → right shows execution order.</li>
                    <li><b>Type → Color/Label:</b> <code>llm-powerful</code>, <code>llm-cheap</code>, <code>code</code>.</li>
                    <li>
                      <b>Stack:</b> <code>type: "stack"</code> models a dynamic fork where inputs are routed to different sub-paths in the same step.
                      The stack’s width is the <i>max</i> child width; its height is the <i>sum</i> of child heights.
                    </li>
                  </ul>
                  <div className="grid grid-cols-3 gap-2 pt-2">
                    <div className="text-xs">Powerful LLM</div>
                    <div className={`h-8 rounded border ${TYPE_CLASS["llm-powerful"]}`} />
                    <div className="text-xs">type: llm-powerful</div>
                    <div className="text-xs">Cheap LLM</div>
                    <div className={`h-8 rounded border ${TYPE_CLASS["llm-cheap"]}`} />
                    <div className="text-xs">type: llm-cheap</div>
                    <div className="text-xs">Code</div>
                    <div className={`h-8 rounded border ${TYPE_CLASS["code"]}`} />
                    <div className="text-xs">type: code</div>
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
            <CardTitle className="text-xl">Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="visual" className="w-full">
              <TabsList>
                <TabsTrigger value="visual">Visual Editor</TabsTrigger>
                <TabsTrigger value="json">JSON</TabsTrigger>
              </TabsList>
              <TabsContent value="visual" className="mt-3">
                <div className="space-y-4">
                  {pipelines.map((pipeline, pIdx) => (
                    <Card key={`p-card-${pIdx}`} className="border-border">
                      <CardHeader className="pb-2">
                        <div className="flex items-center justify-between gap-2">
                          <Input
                            value={pipeline.name}
                            onChange={(e) => updatePipelineName(pIdx, e.target.value)}
                            className="h-8 text-sm font-semibold"
                          />
                          <Button variant="ghost" size="icon" onClick={() => removePipeline(pIdx)}>
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        {pipeline.operators.length === 0 && (
                          <p className="text-xs text-muted-foreground">No blocks yet. Add your first block.</p>
                        )}
                        {pipeline.operators.map((op, bIdx) => {
                          const isStack = (op as any).type === "stack";
                          return (
                            <div key={`op-${pIdx}-${bIdx}`} className="rounded-md border p-3 space-y-2">
                              <div className="flex flex-wrap items-end gap-2">
                                <div className="w-36">
                                  <Label className="text-xs">Type</Label>
                                  <Select
                                    value={(op as any).type}
                                    onValueChange={(v) => updateBlockField(pIdx, bIdx, "type", v)}
                                  >
                                    <SelectTrigger className="h-8 text-sm" />
                                    <SelectContent>
                                      <SelectItem value="llm-powerful">llm-powerful</SelectItem>
                                      <SelectItem value="llm-cheap">llm-cheap</SelectItem>
                                      <SelectItem value="code">code</SelectItem>
                                      <SelectItem value="stack">stack</SelectItem>
                                    </SelectContent>
                                  </Select>
                                </div>
                                {!isStack && (
                                  <div className="flex-1 min-w-[160px]">
                                    <Label className="text-xs">Label</Label>
                                    <Input
                                      className="h-8 text-sm"
                                      value={(op as any).label || ""}
                                      onChange={(e) => updateBlockField(pIdx, bIdx, "label", e.target.value)}
                                    />
                                  </div>
                                )}
                                {!isStack && (
                                  <div className="w-28">
                                    <Label className="text-xs">Width</Label>
                                    <Input
                                      type="number"
                                      step="0.25"
                                      min="0"
                                      className={`h-8 text-sm ${(op as any).width && (op as any).width! <= 0 ? "border-destructive" : ""}`}
                                      value={String((op as any).width ?? 1)}
                                      onChange={(e) => updateBlockField(pIdx, bIdx, "width", parseFloat(e.target.value) || 0)}
                                    />
                                  </div>
                                )}
                                {!isStack && (
                                  <div className="w-28">
                                    <Label className="text-xs">Height</Label>
                                    <Input
                                      type="number"
                                      step="0.25"
                                      min="0"
                                      className={`h-8 text-sm ${(op as any).height && (op as any).height! <= 0 ? "border-destructive" : ""}`}
                                      value={String((op as any).height ?? 1)}
                                      onChange={(e) => updateBlockField(pIdx, bIdx, "height", parseFloat(e.target.value) || 0)}
                                    />
                                  </div>
                                )}
                                <div className="ml-auto flex items-center gap-1">
                                  <Button variant="ghost" size="icon" onClick={() => reorderBlock(pIdx, bIdx, -1)}>
                                    <ChevronLeft className="h-4 w-4" />
                                  </Button>
                                  <Button variant="ghost" size="icon" onClick={() => reorderBlock(pIdx, bIdx, 1)}>
                                    <ChevronRight className="h-4 w-4" />
                                  </Button>
                                  <Button variant="ghost" size="icon" onClick={() => removeBlock(pIdx, bIdx)}>
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                </div>
                              </div>

                              {isStack && (
                                <div className="mt-1 space-y-2">
                                  <div className="text-xs text-muted-foreground">Stack paths (vertical):</div>
                                  {((op as any).operators || []).map((child: any, sIdx: number) => (
                                    <div key={`s-${pIdx}-${bIdx}-${sIdx}`} className="rounded-sm border p-2">
                                      <div className="flex flex-wrap items-end gap-2">
                                        <div className="w-36">
                                          <Label className="text-xs">Type</Label>
                                          <Select
                                            value={child.type}
                                            onValueChange={(v) => updateSubBlockField(pIdx, bIdx, sIdx, "type", v)}
                                          >
                                            <SelectTrigger className="h-8 text-sm" />
                                            <SelectContent>
                                              <SelectItem value="llm-powerful">llm-powerful</SelectItem>
                                              <SelectItem value="llm-cheap">llm-cheap</SelectItem>
                                              <SelectItem value="code">code</SelectItem>
                                            </SelectContent>
                                          </Select>
                                        </div>
                                        <div className="flex-1 min-w-[160px]">
                                          <Label className="text-xs">Label</Label>
                                          <Input
                                            className="h-8 text-sm"
                                            value={child.label || ""}
                                            onChange={(e) => updateSubBlockField(pIdx, bIdx, sIdx, "label", e.target.value)}
                                          />
                                        </div>
                                        <div className="w-28">
                                          <Label className="text-xs">Width</Label>
                                          <Input
                                            type="number"
                                            step="0.25"
                                            min="0"
                                            className={`h-8 text-sm ${child.width && child.width <= 0 ? "border-destructive" : ""}`}
                                            value={String(child.width ?? 1)}
                                            onChange={(e) => updateSubBlockField(pIdx, bIdx, sIdx, "width", parseFloat(e.target.value) || 0)}
                                          />
                                        </div>
                                        <div className="w-28">
                                          <Label className="text-xs">Height</Label>
                                          <Input
                                            type="number"
                                            step="0.25"
                                            min="0"
                                            className={`h-8 text-sm ${child.height && child.height <= 0 ? "border-destructive" : ""}`}
                                            value={String(child.height ?? 1)}
                                            onChange={(e) => updateSubBlockField(pIdx, bIdx, sIdx, "height", parseFloat(e.target.value) || 0)}
                                          />
                                        </div>
                                        <div className="ml-auto">
                                          <Button variant="ghost" size="icon" onClick={() => removeSubBlock(pIdx, bIdx, sIdx)}>
                                            <Trash2 className="h-4 w-4" />
                                          </Button>
                                        </div>
                                      </div>
                                    </div>
                                  ))}
                                  <Button variant="outline" size="sm" onClick={() => addSubBlock(pIdx, bIdx)}>
                                    <Plus className="h-4 w-4 mr-1" /> Add Stack Path
                                  </Button>
                                </div>
                              )}
                            </div>
                          );
                        })}
                        <Button variant="outline" size="sm" onClick={() => addBlock(pIdx)}>
                          <Plus className="h-4 w-4 mr-1" /> Add Block
                        </Button>
                      </CardContent>
                    </Card>
                  ))}
                  <Button onClick={addPipeline}>
                    <Plus className="h-4 w-4 mr-1" /> Add Pipeline
                  </Button>
                </div>
              </TabsContent>
              <TabsContent value="json" className="mt-3">
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
