import React, { useState, useEffect, useMemo, useCallback } from "react";
import { ColumnType } from "@/components/ResizableDataTable";
import ResizableDataTable from "@/components/ResizableDataTable";
import { usePipelineContext } from "@/contexts/PipelineContext";
import { Loader2, Download, ChevronDown, Terminal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Operation, OutputRow } from "@/app/types";
import { Parser } from "json2csv";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useWebSocket } from "@/contexts/WebSocketContext";
import AnsiRenderer from "./AnsiRenderer";
import clsx from "clsx";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from "recharts";
import { memo } from "react";

const TinyPieChart: React.FC<{ percentage: number }> = ({ percentage }) => {
  const size = 16;
  const radius = 6;
  const strokeWidth = 2;
  const center = size / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDasharray = `${
    (percentage * circumference) / 100
  } ${circumference}`;

  return (
    <div className="relative w-4 h-4 flex items-center justify-center">
      <svg
        className="w-4 h-4 -rotate-90"
        viewBox={`0 0 ${size} ${size}`}
        width={size}
        height={size}
      >
        <circle
          cx={center}
          cy={center}
          r={radius}
          className="fill-none stroke-gray-200"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={center}
          cy={center}
          r={radius}
          className={clsx(
            "fill-none transition-all duration-500 ease-out",
            percentage > 100 ? "stroke-emerald-500" : "stroke-rose-500"
          )}
          strokeWidth={strokeWidth}
          strokeDasharray={strokeDasharray}
        />
      </svg>
    </div>
  );
};

// Create a custom hook to find the operation only when needed
const useOperation = (operationId: string | undefined) => {
  const { operations } = usePipelineContext();
  return useMemo(
    () => operations.find((op) => op.id === operationId),
    [operationId] // Only depend on the ID, not the operations array
  );
};

// Update the useOutputContext to not include operations
const useOutputContext = () => {
  const {
    output,
    isLoadingOutputs,
    terminalOutput,
    setTerminalOutput,
    optimizerProgress,
    sampleSize,
    operations,
  } = usePipelineContext();

  return {
    output,
    isLoadingOutputs,
    terminalOutput,
    setTerminalOutput,
    optimizerProgress,
    sampleSize,
    operations,
  };
};

// First, move TableContent outside and give it a display name
const TableContent = memo(
  ({
    opName,
    isLoadingOutputs,
    outputs,
    operation,
    columns,
  }: {
    opName: string | undefined;
    isLoadingOutputs: boolean;
    outputs: OutputRow[];
    operation: Operation | undefined;
    columns: ColumnType<OutputRow>[];
  }) => {
    return (
      <div className="flex-1 min-h-0">
        {!opName ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-muted-foreground">No operation selected.</p>
          </div>
        ) : isLoadingOutputs ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
            <span className="ml-2 text-muted-foreground">
              Loading outputs...
            </span>
          </div>
        ) : outputs.length > 0 ? (
          <div className="h-full">
            <ResizableDataTable
              data={outputs}
              columns={columns}
              boldedColumns={
                operation?.output?.schema
                  ? operation.output.schema.map((field) => field.key)
                  : []
              }
              startingRowHeight={180}
              currentOperation={opName}
            />
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-muted-foreground">No outputs available.</p>
          </div>
        )}
      </div>
    );
  }
);
TableContent.displayName = "TableContent";

// Move VisualizeContent outside
const VisualizeContent = memo(
  ({
    opName,
    outputs,
    operation,
  }: {
    opName: string | undefined;
    outputs: OutputRow[];
    operation: Operation | undefined;
  }) => {
    const visualizationColumn = useMemo(() => {
      if (!opName) return null;
      const reduceColumnName = `_counts_prereduce_${opName}`;
      const resolveColumnName = `_kv_pairs_preresolve_${opName}`;
      return outputs.length > 0 && reduceColumnName in outputs[0]
        ? { name: reduceColumnName, type: "reduce" }
        : outputs.length > 0 && resolveColumnName in outputs[0]
        ? { name: resolveColumnName, type: "resolve" }
        : null;
    }, [outputs, opName, operation]);

    if (!visualizationColumn || !operation) {
      return (
        <div className="flex items-center justify-center h-full">
          <p className="text-muted-foreground">
            No visualization data available.
          </p>
        </div>
      );
    }

    if (operation.type === "reduce") {
      const reduceKeys = operation.otherKwargs?.reduce_key || [];
      const chartData = outputs
        .sort(
          (a, b) =>
            Number(b[visualizationColumn.name]) -
            Number(a[visualizationColumn.name])
        )
        .map((row) => ({
          name: reduceKeys.map((key: string) => `${row[key]}`).join(", "),
          value: Number(row[visualizationColumn.name]),
        }));

      return (
        <div className="h-full p-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ left: 100, right: 20, top: 20, bottom: 20 }}
            >
              <XAxis
                type="number"
                tickFormatter={(value) => value.toLocaleString()}
                className="text-foreground text-xs"
                stroke="currentColor"
              />
              <YAxis
                type="category"
                dataKey="name"
                width={100}
                className="text-foreground text-xs"
                stroke="currentColor"
              />
              <RechartsTooltip
                formatter={(value: number) => [value.toLocaleString(), "Count"]}
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "var(--radius)",
                  color: "hsl(var(--popover-foreground))",
                  padding: "8px 12px",
                  boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
                }}
              />
              <Bar
                dataKey="value"
                fill="hsl(var(--chart-2))"
                name="Count"
                radius={[0, 4, 4, 0]} // Adjusted for vertical layout
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      );
    } else if (operation.type === "resolve") {
      const groupedData = useMemo(() => {
        const intersectionKeys = new Set(
          outputs.flatMap((row) => {
            // @ts-expect-error - Record type needs refinement for kvPairs structure
            const kvPairs = row[visualizationColumn.name] as Record<
              string,
              unknown
            >;
            return Object.keys(kvPairs).filter((key) => key in row);
          })
        );

        const groupedByIntersection: {
          [key: string]: {
            count: number;
            oldValues: Record<string, unknown>[];
          };
        } = {};

        outputs.forEach((row) => {
          // @ts-expect-error - Record type needs refinement for kvPairs structure
          const kvPairs = row[visualizationColumn.name] as Record<
            string,
            unknown
          >;
          const key = Array.from(intersectionKeys)
            .map((k) => row[k])
            .join("|");

          if (!groupedByIntersection[key]) {
            groupedByIntersection[key] = {
              count: 0,
              oldValues: [],
            };
          }
          groupedByIntersection[key].count++;
          groupedByIntersection[key].oldValues.push(kvPairs);
        });

        return Object.entries(groupedByIntersection)
          .map(([key, data]) => ({
            name: key,
            value: data.count,
            oldValues: data.oldValues,
          }))
          .sort((a, b) => b.value - a.value);
      }, [outputs, visualizationColumn.name]);

      return (
        <div className="h-full p-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={groupedData}
              layout="vertical"
              margin={{ left: 100, right: 20, top: 20, bottom: 20 }}
            >
              <XAxis
                type="number"
                tickFormatter={(value) => value.toLocaleString()}
                className="text-foreground text-xs"
                stroke="currentColor"
              />
              <YAxis
                type="category"
                dataKey="name"
                width={100}
                className="text-foreground text-xs"
                stroke="currentColor"
              />
              <RechartsTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;

                    // Compute distinct values by stringifying and using Set
                    const distinctValues = Array.from(
                      new Set(
                        data.oldValues.map((value) =>
                          JSON.stringify(value, null, 2)
                        )
                      )
                      // @ts-expect-error - Record type needs refinement for kvPairs structure
                    ).map((str) => JSON.parse(str));

                    // Calculate percentage of total documents
                    const totalDocs = groupedData.reduce(
                      (sum, item) => sum + item.value,
                      0
                    );
                    const percentage = ((data.value / totalDocs) * 100).toFixed(
                      1
                    );

                    return (
                      <div className="bg-[hsl(var(--popover))] border border-[hsl(var(--border))] rounded-[var(--radius)] p-3 shadow-md max-h-[400px] overflow-y-auto w-[400px]">
                        <p className="font-medium mb-2">
                          {data.value.toLocaleString()} Documents ({percentage}
                          %)
                        </p>
                        <div className="text-sm space-y-2">
                          <p className="font-medium text-[hsl(var(--muted-foreground))]">
                            {distinctValues.length} distinct value
                            {distinctValues.length !== 1 ? "s" : ""} before
                            resolution:
                          </p>
                          {distinctValues.map((value, idx) => (
                            <pre
                              key={idx}
                              className="text-xs bg-[hsl(var(--muted)/.1)] p-2 rounded overflow-x-auto whitespace-pre-wrap break-all"
                            >
                              {JSON.stringify(value, null, 2)}
                            </pre>
                          ))}
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar
                dataKey="value"
                fill="hsl(var(--chart-2))"
                name="Documents"
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      );
    }

    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">
          Visualization not supported for this operation type.
        </p>
      </div>
    );
  }
);
VisualizeContent.displayName = "VisualizeContent";

// Move ConsoleContent outside
export const ConsoleContent = memo(() => {
  const { terminalOutput, setTerminalOutput, optimizerProgress } =
    useOutputContext();
  const { readyState } = useWebSocket();

  return (
    <div className="flex flex-col h-full p-4">
      {optimizerProgress && (
        <div className="flex-none mb-4 p-[6px] rounded-lg relative">
          {/* Animated gradient border */}
          <div
            className="absolute inset-0 rounded-lg opacity-80"
            style={{
              background:
                "linear-gradient(45deg, #60a5fa, #c084fc, #818cf8, #60a5fa, #60a5fa, #c084fc, #818cf8)",
              backgroundSize: "300% 300%",
              animation: "gradient 8s linear infinite",
            }}
          />

          {/* Inner content container */}
          <div className="relative rounded-lg p-4 bg-white">
            <div className="flex items-center justify-between mb-2">
              <div className="text-sm font-medium bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent">
                {optimizerProgress.status}
              </div>
              <div className="text-xs text-blue-600">
                {Math.round(optimizerProgress.progress * 100)}%
              </div>
            </div>
            <div className="relative w-full h-2 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="absolute top-0 left-0 h-full"
                style={{
                  width: `${optimizerProgress.progress * 100}%`,
                  background:
                    "linear-gradient(45deg, #60a5fa, #c084fc, #818cf8, #60a5fa, #60a5fa, #c084fc, #818cf8)",
                  backgroundSize: "300% 300%",
                  animation: "gradient 8s linear infinite",
                }}
              />
            </div>

            {optimizerProgress.shouldOptimize && (
              <div className="mt-4 space-y-4">
                <details className="group">
                  <summary className="cursor-pointer list-none">
                    <div className="flex items-center">
                      <div className="text-xs font-medium uppercase tracking-wider bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent">
                        Optimizing because
                      </div>
                      <ChevronDown className="w-4 h-4 ml-2 text-gray-500 transition-transform group-open:rotate-180" />
                    </div>
                  </summary>
                  <div className="mt-1 text-sm text-gray-600">
                    {optimizerProgress.rationale}
                  </div>
                </details>

                {optimizerProgress.validatorPrompt && (
                  <details className="group">
                    <summary className="cursor-pointer list-none">
                      <div className="flex items-center">
                        <div className="text-xs font-medium uppercase tracking-wider bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent">
                          Using this prompt to evaluate the best plan
                        </div>
                        <ChevronDown className="w-4 h-4 ml-2 text-gray-500 transition-transform group-open:rotate-180" />
                      </div>
                    </summary>
                    <div className="mt-1 text-sm text-gray-600 whitespace-pre-wrap border-l-4 border-purple-300 pl-3 italic">
                      {optimizerProgress.validatorPrompt}
                    </div>
                  </details>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      <div className="flex-1 overflow-auto">
        <AnsiRenderer
          text={terminalOutput || ""}
          readyState={readyState}
          setTerminalOutput={setTerminalOutput}
        />
      </div>
    </div>
  );
});
ConsoleContent.displayName = "ConsoleContent";

// Main Output component
export const Output = memo(() => {
  const { output, isLoadingOutputs, sampleSize, operations } =
    useOutputContext();
  const operation = useOperation(output?.operationId);

  const [outputs, setOutputs] = useState<OutputRow[]>([]);
  const [inputCount, setInputCount] = useState<number>(0);
  const [outputCount, setOutputCount] = useState<number>(0);

  const [opName, setOpName] = useState<string | undefined>(undefined);
  const [isResolveOrReduce, setIsResolveOrReduce] = useState<boolean>(false);

  const [activeTab, setActiveTab] = useState<string>("table");
  const { readyState } = useWebSocket();

  // Effect for operation updates
  useEffect(() => {
    setOpName(operation?.name);
    setIsResolveOrReduce(
      operation?.type === "resolve" || operation?.type === "reduce"
    );
  }, [operation]);

  // Effect for tab changes
  useEffect(() => {
    setActiveTab(isLoadingOutputs ? "console" : "table");
  }, [isLoadingOutputs]);

  // Memoize columns
  const columns = useMemo(() => {
    const importantColumns = operation?.output?.schema
      ? operation.output.schema.map((field) => field.key)
      : [];

    return outputs.length > 0
      ? (Object.keys(outputs[0]).map((key) => ({
          accessorKey: key,
          header: key,
          cell: ({ getValue }: { getValue: () => unknown }) => {
            const value = getValue();
            const stringValue =
              typeof value === "object" && value !== null
                ? JSON.stringify(value, null, 2)
                : String(value);
            return (
              <pre className="whitespace-pre-wrap font-mono text-sm">
                {stringValue}
              </pre>
            );
          },
          initialWidth: importantColumns?.includes(key) ? 300 : 150,
        })) as ColumnType<OutputRow>[])
      : [];
  }, [outputs, operation?.output?.schema]);

  // Memoize handlers
  const downloadCSV = useCallback(() => {
    if (outputs.length === 0) return;

    try {
      const parser = new Parser();
      const csvContent = parser.parse(outputs);

      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", "output.csv");
        link.style.visibility = "hidden";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (err) {
      console.error("Error converting to CSV:", err);
    }
  }, [outputs]);

  const handleTabChange = useCallback((value: string) => {
    setActiveTab(value);
  }, []);

  // Memoize computed values
  const selectivityFactor = useMemo(
    () => (inputCount > 0 ? (outputCount / inputCount).toFixed(2) : "N/A"),
    [inputCount, outputCount]
  );

  // Update the data fetching effect
  useEffect(() => {
    const fetchData = async () => {
      if (output && !isLoadingOutputs) {
        const importantColumns =
          operation?.otherKwargs?.prompts?.[0]?.output_keys;
        try {
          // Fetch output data
          const outputResponse = await fetch(
            `/api/readFile?path=${output.path}`
          );
          if (!outputResponse.ok) {
            throw new Error("Failed to fetch output file");
          }
          const outputContent = await outputResponse.text();
          let parsedOutputs = JSON.parse(outputContent) as OutputRow[];
          setOutputCount(parsedOutputs.length);

          // Sort and reorder columns (existing logic)
          if (parsedOutputs.length > 0) {
            if ("date" in parsedOutputs[0]) {
              parsedOutputs.sort((a, b) => {
                const dateA = (a as OutputRow & { date?: string }).date;
                const dateB = (b as OutputRow & { date?: string }).date;
                if (dateA && dateB) {
                  return new Date(dateB).getTime() - new Date(dateA).getTime();
                }
                return 0;
              });
            }

            if (importantColumns && importantColumns.length > 0) {
              parsedOutputs = parsedOutputs.map((row) => {
                const orderedRow: OutputRow = {};
                importantColumns.forEach((col: string) => {
                  if (col in row) {
                    orderedRow[col] = row[col];
                  }
                });
                Object.keys(row).forEach((key) => {
                  if (!importantColumns.includes(key)) {
                    orderedRow[key] = row[key];
                  }
                });
                return orderedRow;
              });
            }
          }

          setOutputs(parsedOutputs);

          // Check if this is the first operation
          const isFirstOperation = operation?.id === operations[0]?.id;

          // Set input count based on whether it's the first operation
          if (isFirstOperation && sampleSize !== null) {
            setInputCount(sampleSize);
          } else if (output.inputPath) {
            const inputResponse = await fetch(
              `/api/readFile?path=${output.inputPath}`
            );
            if (!inputResponse.ok) {
              throw new Error("Failed to fetch input file");
            }
            const inputContent = await inputResponse.text();
            const parsedInputs = JSON.parse(inputContent);
            setInputCount(
              Array.isArray(parsedInputs) ? parsedInputs.length : 1
            );
          } else {
            setInputCount(0);
          }
        } catch (error) {
          console.error("Error fetching data:", error);
        }
      }
    };

    fetchData();
  }, [
    output,
    operation?.otherKwargs?.prompts,
    isLoadingOutputs,
    operations,
    operation?.id,
    sampleSize,
  ]);

  return (
    <div className="flex flex-col h-full bg-white">
      <Tabs
        value={activeTab}
        onValueChange={handleTabChange}
        className="flex-1 flex flex-col min-h-0"
      >
        <div className="flex-none px-4 pt-4">
          <div className="flex justify-between items-center border-b pb-3">
            <div className="flex items-center gap-4">
              <h2 className="text-base font-bold flex items-center">
                <Terminal className="mr-2" size={14} />
                OUTPUT
                {opName && (
                  <span className="text-gray-500 ml-1">- {opName}</span>
                )}
              </h2>
              <TabsList>
                <TabsTrigger value="console" className="flex items-center">
                  Console
                  {readyState === WebSocket.OPEN && (
                    <span className="ml-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                    </span>
                  )}
                </TabsTrigger>
                <TabsTrigger value="table">Table</TabsTrigger>
                <TabsTrigger value="visualize" disabled={!isResolveOrReduce}>
                  Visualize Input Distribution
                </TabsTrigger>
              </TabsList>
            </div>
            <div className="flex items-center space-x-2">
              <div className="flex items-center gap-2 text-sm">
                <div className="flex items-center px-2 py-1 border border-gray-200 rounded-md">
                  <span className="text-gray-900 font-medium">
                    {isLoadingOutputs ? "0" : inputCount}
                  </span>
                  <span className="text-gray-500 ml-1">in</span>
                </div>
                <span className="text-gray-400">→</span>
                <div className="flex items-center px-2 py-1 border border-gray-200 rounded-md">
                  <span className="text-gray-900 font-medium">
                    {isLoadingOutputs ? "0" : outputCount}
                  </span>
                  <span className="text-gray-500 ml-1">out</span>
                </div>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <div className="flex items-center px-2 py-1 border border-gray-200 rounded-md cursor-help">
                        <span
                          className={clsx(
                            "font-medium",
                            !isLoadingOutputs &&
                              selectivityFactor !== "N/A" &&
                              Number(selectivityFactor) > 1 &&
                              "text-emerald-600",
                            !isLoadingOutputs &&
                              selectivityFactor !== "N/A" &&
                              Number(selectivityFactor) < 1 &&
                              "text-rose-600",
                            (isLoadingOutputs || selectivityFactor === "N/A") &&
                              "text-gray-900"
                          )}
                        >
                          {isLoadingOutputs ? "N/A" : selectivityFactor}×
                        </span>
                        {!isLoadingOutputs &&
                          selectivityFactor !== "N/A" &&
                          Number(selectivityFactor) < 1 && (
                            <div className="ml-1">
                              <TinyPieChart
                                percentage={Number(selectivityFactor) * 100}
                              />
                            </div>
                          )}
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="font-medium">Output to input ratio</p>
                      {!isLoadingOutputs && selectivityFactor !== "N/A" && (
                        <p className="text-xs mt-1">
                          {Number(selectivityFactor) > 1
                            ? "Operation increases data volume"
                            : Number(selectivityFactor) < 1
                            ? "Operation reduces data volume"
                            : "Operation maintains data volume"}
                        </p>
                      )}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      // className="h-8 w-8 text-gray-500 hover:text-gray-700"
                      onClick={downloadCSV}
                      disabled={outputs.length === 0}
                    >
                      <Download size={14} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Download as CSV</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
        </div>

        <div className="flex-1 min-h-0">
          <TabsContent
            value="table"
            className="h-full data-[state=active]:flex flex-col"
          >
            <TableContent
              opName={opName}
              isLoadingOutputs={isLoadingOutputs}
              outputs={outputs}
              operation={operation}
              columns={columns}
            />
          </TabsContent>
          <TabsContent
            value="console"
            className="h-full data-[state=active]:flex flex-col"
          >
            <ConsoleContent />
          </TabsContent>
          <TabsContent
            value="visualize"
            className="h-full data-[state=active]:flex flex-col"
          >
            <VisualizeContent
              opName={opName}
              outputs={outputs}
              operation={operation}
            />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
});
Output.displayName = "Output";
