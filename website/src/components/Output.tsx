import React, { useState, useEffect, useMemo } from "react";
import { ColumnType } from "@/components/ResizableDataTable";
import ResizableDataTable from "@/components/ResizableDataTable";
import { usePipelineContext } from "@/contexts/PipelineContext";
import { Loader2, Download, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import BookmarkableText from "@/components/BookmarkableText";
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

export const ConsoleContent: React.FC = () => {
  const { terminalOutput, setTerminalOutput, optimizerProgress } =
    usePipelineContext();
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
};

export const Output: React.FC = () => {
  const { output, isLoadingOutputs, operations } = usePipelineContext();
  const [outputs, setOutputs] = useState<OutputRow[]>([]);
  const [inputCount, setInputCount] = useState<number>(0);
  const [outputCount, setOutputCount] = useState<number>(0);

  const [operation, setOperation] = useState<Operation | undefined>(undefined);
  const [opName, setOpName] = useState<string | undefined>(undefined);
  const [isResolveOrReduce, setIsResolveOrReduce] = useState<boolean>(false);

  const [defaultTab, setDefaultTab] = useState<string>("table");
  const { readyState } = useWebSocket();

  useEffect(() => {
    if (!isLoadingOutputs) {
      setDefaultTab("table");
    } else {
      setDefaultTab("console");
    }
  }, [isLoadingOutputs]);

  useEffect(() => {
    const foundOperation = operations.find(
      (op: Operation) => op.id === output?.operationId
    );
    setOperation(foundOperation);
    setOpName(foundOperation?.name);
    setIsResolveOrReduce(
      foundOperation?.type === "resolve" || foundOperation?.type === "reduce"
    );
  }, [operations, output]);

  useEffect(() => {
    const fetchData = async () => {
      if (output) {
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
                const dateA = (a as any).date;
                const dateB = (b as any).date;
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

          // Fetch input data if inputPath exists
          if (output.inputPath) {
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
  }, [output, operation, isLoadingOutputs]);

  const columns: ColumnType<any>[] = React.useMemo(() => {
    const importantColumns = operation?.output?.schema
      ? operation.output.schema.map((field) => field.key)
      : [];

    return outputs.length > 0
      ? Object.keys(outputs[0]).map((key) => ({
          accessorKey: key,
          header: key,
          cell: ({ getValue }: { getValue: () => any }) => {
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
        }))
      : [];
  }, [outputs, operation?.output?.schema]);

  const TableContent = () => (
    <div className="flex-1 min-h-0">
      {isLoadingOutputs ? (
        <div className="flex items-center justify-center h-full">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
          <span className="ml-2 text-muted-foreground">Loading outputs...</span>
        </div>
      ) : outputs.length > 0 ? (
        <BookmarkableText source="output" className="h-full">
          <ResizableDataTable
            data={outputs}
            columns={columns}
            boldedColumns={
              operation?.output?.schema
                ? operation.output.schema.map((field) => field.key)
                : []
            }
            startingRowHeight={180}
          />
        </BookmarkableText>
      ) : (
        <div className="flex items-center justify-center h-full">
          <p className="text-muted-foreground">No outputs available.</p>
        </div>
      )}
    </div>
  );

  const VisualizeContent = () => {
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
      return (
        <div className="h-full overflow-auto p-4">
          {outputs
            .sort(
              (a, b) =>
                Number(b[visualizationColumn.name]) -
                Number(a[visualizationColumn.name])
            )
            .map((row, index) => (
              <div key={index} className="mb-2">
                <h3 className="text-sm font-semibold mb-2">
                  {reduceKeys
                    .map((key: string) => `${key}: ${row[key]}`)
                    .join(", ")}{" "}
                  ({row[visualizationColumn.name]})
                </h3>
                <div className="flex">
                  {Array.from({
                    length: Number(row[visualizationColumn.name]),
                  }).map((_, i) => (
                    <div key={i} className="w-4 h-10 bg-primary mr-1" />
                  ))}
                </div>
              </div>
            ))}
        </div>
      );
    } else if (operation.type === "resolve") {
      const groupedData = useMemo(() => {
        const intersectionKeys = new Set(
          outputs.flatMap((row) => {
            const kvPairs = row[visualizationColumn.name];
            return Object.keys(kvPairs).filter((key) => key in row);
          })
        );

        const groupedByIntersection: { [key: string]: any[] } = {};
        outputs.forEach((row) => {
          const kvPairs = row[visualizationColumn.name];
          const key = Array.from(intersectionKeys)
            .map((k) => row[k])
            .join("|");
          if (!groupedByIntersection[key]) {
            groupedByIntersection[key] = [];
          }
          groupedByIntersection[key].push({ row, oldValues: kvPairs });
        });

        return groupedByIntersection;
      }, [outputs, visualizationColumn.name]);

      return (
        <div className="h-full overflow-auto p-4">
          {Object.entries(groupedData)
            .sort(([, groupA], [, groupB]) => groupB.length - groupA.length)
            .map(([key, group]: [string, any[]]) => (
              <div key={key} className="mb-2">
                <h3 className="text-sm font-semibold mb-2">
                  {key} ({group.length})
                </h3>
                <div className="flex">
                  {group.map((item, index) => (
                    <TooltipProvider key={index}>
                      <Tooltip delayDuration={0}>
                        <TooltipTrigger>
                          <Button
                            className="w-4 h-10 p-0 mr-1"
                            variant="default"
                          />
                        </TooltipTrigger>
                        <TooltipContent
                          side="bottom"
                          align="center"
                          className="max-w-[300px] overflow-auto"
                        >
                          <p className="text-xs">
                            Document values before resolution:
                          </p>
                          <pre className="whitespace-pre-wrap break-words">
                            {JSON.stringify(item.oldValues, null, 2)}
                          </pre>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  ))}
                </div>
              </div>
            ))}
        </div>
      );
    } else {
      return (
        <div className="flex items-center justify-center h-full">
          <p className="text-muted-foreground">
            Visualization not supported for this operation type.
          </p>
        </div>
      );
    }
  };

  const downloadCSV = () => {
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
  };

  const selectivityFactor =
    inputCount > 0 ? (outputCount / inputCount).toFixed(2) : "N/A";

  return (
    <div className="flex flex-col h-full bg-white">
      <div className="flex-none p-4 pb-2">
        <div className="flex justify-between items-center">
          <h2 className="text-sm font-bold flex items-center uppercase">
            Output{" "}
            {opName && <span className="text-gray-500 ml-1">- {opName}</span>}
          </h2>
          <div className="flex items-center space-x-2">
            <div className="flex items-center gap-2 text-sm">
              <div className="flex items-center px-2 py-1 border border-gray-200 rounded-md">
                <span className="text-gray-900 font-medium">{inputCount}</span>
                <span className="text-gray-500 ml-1">in</span>
              </div>
              <span className="text-gray-400">→</span>
              <div className="flex items-center px-2 py-1 border border-gray-200 rounded-md">
                <span className="text-gray-900 font-medium">{outputCount}</span>
                <span className="text-gray-500 ml-1">out</span>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <div className="flex items-center px-2 py-1 border border-gray-200 rounded-md">
                      <span
                        className={clsx(
                          "font-medium",
                          selectivityFactor !== "N/A" &&
                            Number(selectivityFactor) > 1 &&
                            "text-emerald-600",
                          selectivityFactor !== "N/A" &&
                            Number(selectivityFactor) < 1 &&
                            "text-rose-600",
                          selectivityFactor === "N/A" && "text-gray-900"
                        )}
                      >
                        {selectivityFactor}×
                      </span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="font-medium">Output to input ratio</p>
                    {selectivityFactor !== "N/A" && (
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
                    variant="ghost"
                    size="sm"
                    className="p-1"
                    onClick={downloadCSV}
                    disabled={outputs.length === 0}
                  >
                    <Download size={16} />
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

      <Tabs defaultValue={defaultTab} className="flex-1 flex flex-col min-h-0">
        <div className="flex-none px-4">
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

        <div className="flex-1 min-h-0">
          <TabsContent
            value="table"
            className="h-full data-[state=active]:flex flex-col"
          >
            <TableContent />
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
            <VisualizeContent />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};
