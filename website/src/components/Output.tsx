import React, { useState, useEffect } from "react";
import { ColumnType } from "@/components/ResizableDataTable";
import ResizableDataTable from "@/components/ResizableDataTable";
import { usePipelineContext } from "@/contexts/PipelineContext";
import { Loader2, Download, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
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
import ForceGraph2D from "react-force-graph-2d";
import {
  BarChart,
  XAxis,
  YAxis,
  Bar,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from "recharts";

interface GraphNode {
  id: number;
  size: number;
  items: unknown[];
  [key: string]: unknown;
}

interface ForceGraphNode extends GraphNode {
  x?: number;
  y?: number;
  __r?: number;
}

interface GraphLink {
  source: number;
  target: number;
  distance: number;
  [key: string]: unknown;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphLink[];
  distance_matrix: number[][];
}

interface DateRow extends Record<string, unknown> {
  date?: string;
}

interface ForceGraphMethods {
  link: () => {
    distance: (fn: (link: GraphLink) => number) => void;
  };
  nodes: () => {
    strength: (strength: number) => void;
  };
}

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

  const [activeTab, setActiveTab] = useState<string>("table");
  const { readyState } = useWebSocket();

  useEffect(() => {
    if (isLoadingOutputs) {
      setActiveTab("console");
    } else {
      setActiveTab("table");
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
                const dateA = (a as DateRow).date;
                const dateB = (b as DateRow).date;
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

  const columns: ColumnType<Record<string, unknown>>[] = React.useMemo(() => {
    const importantColumns = operation?.output?.schema
      ? operation.output.schema.map((field) => field.key)
      : [];

    return outputs.length > 0
      ? Object.keys(outputs[0]).map((key) => ({
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
    const [graphData, setGraphData] = useState<GraphData | null>(null);
    const [isLoadingGraph, setIsLoadingGraph] = useState(false);

    useEffect(() => {
      const fetchGraphData = async () => {
        if (!output?.path || operation?.type !== "resolve") return;

        setIsLoadingGraph(true);
        try {
          const graphPath = output.path.replace(".json", "_debug_graph.json");
          const response = await fetch(`/api/readFile?path=${graphPath}`);
          if (!response.ok) {
            throw new Error("Failed to fetch graph data");
          }
          const data = await response.json();

          // Normalize distances
          const distances = data.edges.map((edge: GraphLink) => edge.distance);
          const minDist = Math.min(...distances);
          const maxDist = Math.max(...distances);
          const normalizedEdges = data.edges.map((edge: GraphLink) => ({
            ...edge,
            distance: (edge.distance - minDist) / (maxDist - minDist),
          }));

          // Filter edges - only keep meaningful connections
          const filteredEdges = normalizedEdges.filter(
            (edge: GraphLink) => edge.distance < 0.2
          );

          setGraphData({
            ...data,
            edges: filteredEdges,
          });
        } catch (error) {
          console.error("Error fetching graph data:", error);
        } finally {
          setIsLoadingGraph(false);
        }
      };

      fetchGraphData();
    }, [output?.path, operation?.type]);

    if (operation?.type === "reduce") {
      const reduceKeys = operation.otherKwargs?.reduce_key || [];
      const visualizationColumn = `_counts_prereduce_${opName}`;

      const chartData = outputs
        .sort(
          (a, b) =>
            Number(b[visualizationColumn]) - Number(a[visualizationColumn])
        )
        .map((row) => ({
          name: reduceKeys.map((key: string) => `${row[key]}`).join(", "),
          value: Number(row[visualizationColumn]),
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
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      );
    }

    if (isLoadingGraph) {
      return (
        <div className="flex items-center justify-center h-full">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
          <span className="ml-2 text-muted-foreground">
            Loading graph data...
          </span>
        </div>
      );
    }

    if (!graphData) {
      return (
        <div className="flex items-center justify-center h-full">
          <p className="text-muted-foreground">No graph data available.</p>
        </div>
      );
    }

    const graphDataFormatted = {
      nodes: graphData.nodes,
      links: graphData.edges.map((edge) => ({
        ...edge,
        source: edge.source,
        target: edge.target,
      })),
    };

    return (
      <div className="h-full w-full">
        <ForceGraph2D
          graphData={graphDataFormatted}
          nodeRelSize={8}
          nodeVal={(node) => {
            const size = (node as GraphNode).size;
            return Math.sqrt(size) * 8;
          }}
          linkLabel={(link) =>
            `Similarity: ${(100 * (1 - (link as GraphLink).distance)).toFixed(
              1
            )}%`
          }
          linkWidth={3}
          linkColor={() => "rgba(100, 116, 139, 0.4)"}
          backgroundColor="#ffffff"
          nodeColor={() => "hsl(222, 67%, 51%)"}
          nodeLabel={(node) => {
            const n = node as GraphNode;
            const relevantKeys =
              operation?.output?.schema?.map((field) => field.key) || [];
            const itemsPreview = n.items
              .map((item) => {
                const filteredItem: Record<string, unknown> = {};
                relevantKeys.forEach((key) => {
                  if (key in (item as Record<string, unknown>)) {
                    filteredItem[key] = (item as Record<string, unknown>)[key];
                  }
                });
                return JSON.stringify(filteredItem, null, 2);
              })
              .join("\n\n");

            return `Cluster ${n.id} (${n.size} items)\n\nItems:\n${itemsPreview}`;
          }}
          linkDirectionalParticles={0}
          d3VelocityDecay={0.3}
          d3Force={(d3Force: ForceGraphMethods) => {
            d3Force.link().distance((link) => {
              const sourceSize = (link.source as unknown as GraphNode).size;
              const targetSize = (link.target as unknown as GraphNode).size;
              return 200 + Math.sqrt(sourceSize + targetSize) * 4;
            });
            d3Force.nodes().strength(-1000);
          }}
          cooldownTicks={100}
          warmupTicks={100}
        />
      </div>
    );
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

      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        className="flex-1 flex flex-col min-h-0"
      >
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
