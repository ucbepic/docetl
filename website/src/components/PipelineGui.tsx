import React, { useState, useCallback, useEffect, useRef } from "react";
import { Operation, File } from "@/app/types";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { OperationCard } from "@/components/OperationCard";
import { Button } from "@/components/ui/button";
import {
  Play,
  Settings,
  PieChart,
  RefreshCw,
  Download,
  FileUp,
  Loader2,
  StopCircle,
  Brain,
  GitBranch,
  Pencil,
  Plus,
  ChevronRight,
  ChevronLeft,
} from "lucide-react";
import { usePipelineContext } from "@/contexts/PipelineContext";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useToast } from "@/hooks/use-toast";
import { useWebSocket } from "@/contexts/WebSocketContext";
import { Input } from "@/components/ui/input";
import { schemaDictToItemSet } from "./utils";
import { v4 as uuidv4 } from "uuid";
import { useOptimizeCheck } from "@/hooks/useOptimizeCheck";
import { canBeOptimized } from "@/lib/utils";
import { Textarea } from "./ui/textarea";
import { OptimizationDialog } from "@/components/OptimizationDialog";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { useRestorePipeline } from "@/hooks/useRestorePipeline";
import PipelineSettings from "@/components/PipelineSettings";
import {
  DOCWRANGLER_HOSTED_COST_LIMIT,
  isDocWranglerHosted,
} from "@/lib/utils";

interface OperationMenuItemProps {
  name: string;
  description: string;
  onClick: () => void;
}

const OperationMenuItem: React.FC<OperationMenuItemProps> = ({
  name,
  description,
  onClick,
}) => {
  return (
    <HoverCard openDelay={0} closeDelay={0}>
      <HoverCardTrigger asChild>
        <div className="relative w-full">
          <DropdownMenuItem
            onClick={onClick}
            className="w-full cursor-help font-medium hover:bg-primary/10"
          >
            {name}
          </DropdownMenuItem>
        </div>
      </HoverCardTrigger>
      <HoverCardContent side="right" align="start" className="w-72 p-2">
        <div className="space-y-1">
          <h4 className="font-medium text-sm">{name} Operation</h4>
          <p className="text-xs text-muted-foreground leading-snug">
            {description}
          </p>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
};

interface AddOperationDropdownProps {
  onAddOperation: (
    llmType: "LLM" | "non-LLM",
    type: string,
    name: string
  ) => void;
  trigger: React.ReactNode;
}

const AddOperationDropdown: React.FC<AddOperationDropdownProps> = ({
  onAddOperation,
  trigger,
}) => {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{trigger}</DropdownMenuTrigger>
      <DropdownMenuContent>
        <DropdownMenuLabel className="font-bold text-sm bg-muted/50 py-2">
          Add LLM Operation
        </DropdownMenuLabel>
        <OperationMenuItem
          name="Map"
          description="Transforms each input item for complex data processing and insight extraction. 1 to 1 operation (each document gets one result, but the output of the operation can be any type, like a list)."
          onClick={() => onAddOperation("LLM", "map", "Untitled Map")}
        />
        <OperationMenuItem
          name="Reduce"
          description="Aggregates data by key for summarization or folding. Many to 1 operation (many documents get combined into one result)."
          onClick={() => onAddOperation("LLM", "reduce", "Untitled Reduce")}
        />
        <OperationMenuItem
          name="Resolve"
          description="Identifies and merges duplicate entities for data consistency. Keeps the same number of documents; just resolves values."
          onClick={() => onAddOperation("LLM", "resolve", "Untitled Resolve")}
        />
        <OperationMenuItem
          name="Filter"
          description="Selectively includes or excludes data based on specific conditions. This is like a map operation, but with a boolean output schema. The size of your dataset may decrease, as documents that evaluate to false based on the prompt will be dropped from the dataset."
          onClick={() => onAddOperation("LLM", "filter", "Untitled Filter")}
        />
        <OperationMenuItem
          name="Parallel Map"
          description="Like a Map operation, but processes multiple documents in parallel for improved performance. Best used when documents can be processed independently."
          onClick={() =>
            onAddOperation("LLM", "parallel_map", "Untitled Parallel Map")
          }
        />
        <OperationMenuItem
          name="Rank"
          description="Sorts documents based on the given criteria. This may drop documents if you are using the `k` parameter; otherwise it returns the same documents, but in a sorted order."
          onClick={() => onAddOperation("LLM", "rank", "Untitled Rank")}
        />
        <OperationMenuItem
          name="Extract"
          description="Extracts specific sections of text from documents based on provided criteria."
          onClick={() => onAddOperation("LLM", "extract", "Untitled Extract")}
        />
        <DropdownMenuSeparator />
        <DropdownMenuLabel className="font-bold text-sm bg-muted/50 py-2">
          Add Non-LLM Operation
        </DropdownMenuLabel>
        <OperationMenuItem
          name="Unnest"
          description="Flattens nested arrays or objects in your documents, creating new documents for each nested item."
          onClick={() => onAddOperation("non-LLM", "unnest", "Untitled Unnest")}
        />
        <OperationMenuItem
          name="Split"
          description="Divides documents into multiple parts based on specified criteria, creating new documents for each part."
          onClick={() => onAddOperation("non-LLM", "split", "Untitled Split")}
        />
        <OperationMenuItem
          name="Gather"
          description="Collects and groups related data from multiple documents into a single document based on a common key."
          onClick={() => onAddOperation("non-LLM", "gather", "Untitled Gather")}
        />
        <OperationMenuItem
          name="Sample"
          description="Randomly selects a subset of documents from your dataset for testing or analysis."
          onClick={() => onAddOperation("non-LLM", "sample", "Untitled Sample")}
        />
        <DropdownMenuSeparator />
        <DropdownMenuLabel className="font-bold text-sm bg-muted/50 py-2">
          Code Operations
        </DropdownMenuLabel>
        <OperationMenuItem
          name="Code Map"
          description="Like the LLM Map operation, but uses a Python function instead of an LLM. Write custom Python code to transform each document."
          onClick={() =>
            onAddOperation("non-LLM", "code_map", "Untitled Code Map")
          }
        />
        <OperationMenuItem
          name="Code Reduce"
          description="Like the LLM Reduce operation, but uses a Python function instead of an LLM. Write custom Python code to aggregate multiple documents into one."
          onClick={() =>
            onAddOperation("non-LLM", "code_reduce", "Untitled Code Reduce")
          }
        />
        <OperationMenuItem
          name="Code Filter"
          description="Like the LLM Filter operation, but uses a Python function instead of an LLM. Write custom Python code to determine which documents to keep."
          onClick={() =>
            onAddOperation("non-LLM", "code_filter", "Untitled Code Filter")
          }
        />
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

const PipelineGUI: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const {
    operations,
    setOperations,
    pipelineName,
    setPipelineName,
    sampleSize,
    setSampleSize,
    setNumOpRun,
    currentFile,
    setCurrentFile,
    setFiles,
    setOutput,
    isLoadingOutputs,
    setIsLoadingOutputs,
    files,
    setCost,
    cost,
    defaultModel,
    setDefaultModel,
    setTerminalOutput,
    optimizerModel,
    setOptimizerModel,
    setOptimizerProgress,
    autoOptimizeCheck,
    setAutoOptimizeCheck,
    systemPrompt,
    setSystemPrompt,
    namespace,
    apiKeys,
    extraPipelineSettings,
    setExtraPipelineSettings,
  } = usePipelineContext();
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const { toast } = useToast();
  const { connect, sendMessage, lastMessage, readyState, disconnect } =
    useWebSocket();
  const [optimizationDialog, setOptimizationDialog] = useState<{
    isOpen: boolean;
    content: string;
    prompt?: string;
    inputData?: Array<Record<string, unknown>>;
    outputData?: Array<Record<string, unknown>>;
    operationName?: string;
    operationId?: string;
  }>({
    isOpen: false,
    content: "",
    prompt: undefined,
    operationName: undefined,
    operationId: undefined,
  });
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedPipelineName, setEditedPipelineName] = useState(pipelineName);
  const [isLeftSideCollapsed, setIsLeftSideCollapsed] = useState(false);

  const { submitTask } = useOptimizeCheck({
    onComplete: (result) => {
      setOperations((prev) => {
        const newOps = [...prev];
        if (newOps.length > 0) {
          const lastOp = newOps[newOps.length - 1];
          lastOp.shouldOptimizeResult = result.should_optimize;
        }
        return newOps;
      });
      setCost((prev) => prev + result.cost);

      if (result.should_optimize) {
        toast({
          title: `Hey! Consider decomposing ${
            operations[operations.length - 1].name
          }`,
          description: (
            <span
              className="cursor-pointer text-blue-500 hover:text-blue-700"
              onClick={() => {
                const lastOp = operations[operations.length - 1];
                setOptimizationDialog({
                  isOpen: true,
                  content: result.should_optimize,
                  prompt: lastOp.prompt || "No prompt specified",
                  operationName: lastOp.name,
                  operationId: lastOp.id,
                  inputData: result.input_data,
                  outputData: result.output_data,
                });
              }}
            >
              Click here to see why.
            </span>
          ),
          duration: Infinity,
        });
      }
    },
    onError: (error) => {
      toast({
        title: "Optimization Check Failed",
        description: error,
        variant: "destructive",
      });
    },
  });

  const { restoreFromYAML } = useRestorePipeline({
    setOperations,
    setPipelineName,
    setSampleSize,
    setDefaultModel,
    setFiles,
    setCurrentFile,
    setSystemPrompt,
    files,
    setOutput,
  });

  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === "output") {
        setTerminalOutput(lastMessage.data);
      } else if (lastMessage.type === "optimizer_progress") {
        setOptimizerProgress({
          status: lastMessage.status,
          progress: lastMessage.progress,
          shouldOptimize: lastMessage.should_optimize,
          rationale: lastMessage.rationale,
          validatorPrompt: lastMessage.validator_prompt,
        });
      } else if (lastMessage.type === "result") {
        const runCost = lastMessage.data.cost || 0;
        setOptimizerProgress(null);

        // See if there was an optimized operation
        const optimizedOps = lastMessage.data.optimized_ops;
        if (optimizedOps) {
          const newOperations = optimizedOps.map((optimizedOp) => {
            const {
              id,
              type,
              name,
              prompt,
              output,
              validate,
              gleaning,
              sample,
              ...otherKwargs
            } = optimizedOp;

            // Find matching operation in previous operations list
            const existingOp = operations.find((op) => op.name === name);

            return {
              id: id || uuidv4(),
              llmType:
                type === "map" ||
                type === "reduce" ||
                type === "resolve" ||
                type === "filter" ||
                type === "parallel_map" ||
                type === "rank" ||
                type === "extract"
                  ? "LLM"
                  : "non-LLM",
              type: type,
              name: name || "Untitled Operation",
              prompt: prompt,
              output: output
                ? {
                    schema: schemaDictToItemSet(output.schema),
                  }
                : undefined,
              validate: validate,
              gleaning: gleaning,
              sample: sample,
              otherKwargs: otherKwargs || {},
              ...(existingOp?.runIndex && { runIndex: existingOp.runIndex }),
              visibility: true,
            } as Operation;
          });

          setOperations(newOperations);
        } else {
          // No optimized operations, so we need to check if we should optimize the last operation
          // Trigger should optimize for the last operation
          if (autoOptimizeCheck) {
            const lastOp = operations[operations.length - 1];
            if (lastOp && canBeOptimized(lastOp.type)) {
              submitTask({
                yaml_config: lastMessage.data.yaml_config,
                step_name: "data_processing", // TODO: Make this a constant
                op_name: lastOp.name,
              });
            }
          }
        }

        setCost((prevCost) => prevCost + runCost);
        toast({
          title: "Operation Complete",
          description: `The operation cost $${runCost.toFixed(4)}`,
          duration: 3000,
        });

        // Close the WebSocket connection
        disconnect();

        setIsLoadingOutputs(false);
      } else if (lastMessage.type === "error") {
        let description = lastMessage.data;
        if (description.includes("Connection error")) {
          description =
            description +
            " Consider checking your API keys (Edit > Edit API Keys) and ensuring you have a stable internet connection.";
        }
        toast({
          title: "Execution Error",
          description: description,
          variant: "destructive",
          duration: Infinity,
        });

        // Close the WebSocket connection
        disconnect();

        setIsLoadingOutputs(false);
      }
    }
  }, [lastMessage, setCost, setIsLoadingOutputs, setTerminalOutput]);

  useEffect(() => {
    if (pipelineName) {
      setEditedPipelineName(pipelineName);
    }
  }, [pipelineName]);

  useEffect(() => {
    if (autoOptimizeCheck) {
      setAutoOptimizeCheck(autoOptimizeCheck);
    }
  }, [autoOptimizeCheck]);

  useEffect(() => {
    if (defaultModel) {
      setDefaultModel(defaultModel);
    }
  }, [defaultModel]);

  useEffect(() => {
    if (currentFile) {
      setCurrentFile(currentFile);
    }
  }, [currentFile]);

  useEffect(() => {
    if (optimizerModel) {
      setOptimizerModel(optimizerModel);
    }
  }, [optimizerModel]);

  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width < 1100) {
          setIsLeftSideCollapsed(true);
        } else {
          setIsLeftSideCollapsed(false);
        }
      }
    });

    if (headerRef.current) {
      resizeObserver.observe(headerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      try {
        const fileToUpload: File = {
          name: file.name,
          path: file.name,
          type: "pipeline-yaml",
          blob: file,
        };
        await restoreFromYAML(fileToUpload);
      } catch (error) {
        console.error("Error handling file upload:", error);
      }
    }
  };

  const handleExport = async () => {
    try {
      const response = await fetch("/api/getPipelineConfig", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          default_model: defaultModel,
          data: { path: currentFile?.path || "" },
          operations,
          operation_id: operations[operations.length - 1].id,
          name: pipelineName,
          sample_size: sampleSize,
          namespace: namespace,
          system_prompt: systemPrompt,
          optimizerModel: optimizerModel,
          extraPipelineSettings: extraPipelineSettings,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to export pipeline configuration");
      }

      const { pipelineConfig } = await response.json();

      const blob = new Blob([pipelineConfig], { type: "text/yaml" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = `${pipelineName}.yaml`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast({
        title: "Pipeline Exported",
        description: `Your pipeline configuration has been exported successfully to ${pipelineName}.yaml.`,
        duration: 3000,
      });
    } catch (error) {
      console.error("Error exporting pipeline configuration:", error);
      toast({
        title: "Error",
        description: "Failed to export pipeline configuration",
        variant: "destructive",
      });
    }
  };

  const onRunAll = useCallback(
    async (clear_intermediate: boolean) => {
      // Check if cost exceeds limit and no API keys are set
      if (
        isDocWranglerHosted() &&
        cost > DOCWRANGLER_HOSTED_COST_LIMIT &&
        (!apiKeys || Object.keys(apiKeys).length === 0)
      ) {
        toast({
          title: "Cost Limit Exceeded",
          description: `You're trying to run operations that may exceed the cost limit of $${DOCWRANGLER_HOSTED_COST_LIMIT.toFixed(
            2
          )}. Please add your API keys in settings to continue.`,
          variant: "destructive",
          duration: 5000,
        });
        return;
      }

      // Find the last visible operation
      const lastVisibleOpIndex = operations.findLastIndex(
        (op) => op.visibility !== false
      );
      if (lastVisibleOpIndex < 0) return;

      const lastOperation = operations[lastVisibleOpIndex];
      setOptimizerProgress(null);
      setIsLoadingOutputs(true);
      setNumOpRun((prevNum) => {
        const newNum = prevNum + operations.length;
        const updatedOperations = operations.map((op, index) => ({
          ...op,
          runIndex: prevNum + index + 1,
          shouldOptimizeResult: undefined,
        }));
        setOperations(updatedOperations);
        return newNum;
      });

      setTerminalOutput("");

      try {
        // Get the latest API keys from context
        const currentApiKeys = apiKeys;

        const response = await fetch("/api/writePipelineConfig", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            default_model: defaultModel,
            data: { path: currentFile?.path || "" },
            operations,
            operation_id: lastOperation.id,
            name: pipelineName,
            sample_size: sampleSize,
            clear_intermediate: clear_intermediate,
            system_prompt: systemPrompt,
            namespace: namespace,
            apiKeys: currentApiKeys, // Use the latest API keys
            optimizerModel: optimizerModel,
            extraPipelineSettings: extraPipelineSettings,
          }),
        });

        if (!response.ok) {
          throw new Error(await response.text());
        }

        const { filePath, inputPath, outputPath } = await response.json();

        setOutput({
          operationId: lastOperation.id,
          path: outputPath,
          inputPath: inputPath,
        });

        // Ensure the WebSocket is connected before sending the message
        await connect();

        sendMessage({
          yaml_config: filePath,
          clear_intermediate: clear_intermediate,
        });
      } catch (error) {
        console.error("Error writing pipeline config:", error);
        toast({
          title: "Error",
          description: error.message,
          variant: "destructive",
        });
        // Close the WebSocket connection
        disconnect();
        setIsLoadingOutputs(false);
      }
    },
    [
      operations,
      currentFile,
      setIsLoadingOutputs,
      setNumOpRun,
      sendMessage,
      readyState,
      defaultModel,
      pipelineName,
      sampleSize,
      apiKeys,
      systemPrompt,
      namespace,
      extraPipelineSettings,
      cost,
    ]
  );

  const handleAddOperation = (
    llmType: "LLM" | "non-LLM",
    type: string,
    name: string
  ) => {
    const newOperation: Operation = {
      id: String(Date.now()),
      llmType,
      type: type as Operation["type"],
      name: `${name} ${operations.length}`,
      visibility: true,
    };
    setOperations([...operations, newOperation]);
  };

  const handleStop = () => {
    sendMessage("kill");

    if (readyState === WebSocket.CLOSED && isLoadingOutputs) {
      setIsLoadingOutputs(false);
    }
  };

  const handleOptimizeFromDialog = async () => {
    if (!optimizationDialog.operationId) return;

    try {
      setTerminalOutput("");
      setIsLoadingOutputs(true);

      const response = await fetch("/api/writePipelineConfig", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          default_model: defaultModel,
          data: { path: currentFile?.path || "" },
          operations,
          operation_id: optimizationDialog.operationId,
          name: pipelineName,
          sample_size: sampleSize,
          optimize: true,
          namespace: namespace,
          apiKeys: apiKeys,
          optimizerModel: optimizerModel,
          extraPipelineSettings: extraPipelineSettings,
        }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const { filePath } = await response.json();

      await connect();

      sendMessage({
        yaml_config: filePath,
        optimize: true,
      });
    } catch (error) {
      console.error("Error optimizing operation:", error);
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
      disconnect();
      setIsLoadingOutputs(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div
        ref={headerRef}
        className="flex-none relative bg-background border-b sticky top-0 z-10 shadow-sm"
      >
        <div className="p-2">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <div className="flex items-center">
                {isEditingName ? (
                  <Input
                    value={editedPipelineName}
                    onChange={(e) => setEditedPipelineName(e.target.value)}
                    onBlur={() => {
                      setIsEditingName(false);
                      setPipelineName(editedPipelineName);
                    }}
                    onKeyPress={(e) => {
                      if (e.key === "Enter") {
                        setIsEditingName(false);
                        setPipelineName(editedPipelineName);
                      }
                    }}
                    className="max-w-[200px] h-7 text-sm font-bold"
                    autoFocus
                  />
                ) : (
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <h2
                          className="text-base font-bold cursor-pointer hover:text-primary/80 flex items-center gap-1.5 group"
                          onClick={() => setIsEditingName(true)}
                        >
                          {pipelineName}
                          <Pencil
                            size={13}
                            className="opacity-0 group-hover:opacity-70 transition-opacity"
                          />
                        </h2>
                      </TooltipTrigger>
                      <TooltipContent side="bottom">
                        <p>Click to rename pipeline</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}

                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0 flex-shrink-0"
                  onClick={() => setIsLeftSideCollapsed(!isLeftSideCollapsed)}
                >
                  {isLeftSideCollapsed ? (
                    <ChevronRight size={16} />
                  ) : (
                    <ChevronLeft size={16} />
                  )}
                </Button>
              </div>

              <div
                className={`flex items-center gap-2 transition-transform duration-200 origin-left ${
                  isLeftSideCollapsed ? "scale-x-0 w-0" : "scale-x-100"
                }`}
              >
                <div className="flex items-center gap-2 flex-shrink-0">
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-8 whitespace-nowrap"
                      >
                        <GitBranch size={14} className="mr-2" />
                        Overview
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent
                      side="bottom"
                      align="start"
                      className="w-96 p-4"
                    >
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <h4 className="font-medium">Pipeline Flow</h4>
                          <span className="text-xs text-muted-foreground">
                            {operations.filter((op) => op.visibility).length}{" "}
                            operations
                          </span>
                        </div>
                        <div className="bg-muted p-3 rounded-md space-y-2">
                          {operations.length > 0 ? (
                            operations
                              .filter((op) => op.visibility)
                              .map((op, index, arr) => (
                                <div key={op.id} className="flex items-center">
                                  <div className="flex-1 bg-background p-2 rounded-md text-sm">
                                    <div className="font-medium">{op.name}</div>
                                    <div className="text-xs text-muted-foreground">
                                      {op.type}
                                    </div>
                                  </div>
                                  {index < arr.length - 1 && (
                                    <div className="mx-2 text-muted-foreground">
                                      ↓
                                    </div>
                                  )}
                                </div>
                              ))
                          ) : (
                            <div className="text-sm text-muted-foreground">
                              No operations in the pipeline
                            </div>
                          )}
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>

                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-8 whitespace-nowrap"
                      >
                        <Brain size={14} className="mr-2" />
                        System Prompts
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-88">
                      <div className="grid gap-3">
                        <div className="space-y-1">
                          <h4 className="text-lg font-semibold">
                            System Configuration
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            This will be in the system prompt for <b>every</b>{" "}
                            operation!
                          </p>
                        </div>
                        <div className="grid gap-3">
                          <div className="space-y-1">
                            <Label
                              htmlFor="datasetDescription"
                              className="text-sm font-medium"
                            >
                              Dataset Description
                            </Label>
                            <Textarea
                              id="datasetDescription"
                              placeholder="a collection of documents"
                              defaultValue={systemPrompt.datasetDescription}
                              onBlur={(e) => {
                                const value = e.target.value;
                                setTimeout(() => {
                                  setSystemPrompt((prev) => ({
                                    ...prev,
                                    datasetDescription: value,
                                  }));
                                }, 0);
                              }}
                              className="h-[3.5rem]"
                            />
                          </div>
                          <div className="space-y-1">
                            <Label
                              htmlFor="persona"
                              className="text-sm font-medium"
                            >
                              Persona
                            </Label>
                            <Textarea
                              id="persona"
                              placeholder="a helpful assistant"
                              defaultValue={systemPrompt.persona}
                              onBlur={(e) => {
                                const value = e.target.value;
                                setTimeout(() => {
                                  setSystemPrompt((prev) => ({
                                    ...prev,
                                    persona: value,
                                  }));
                                }, 0);
                              }}
                              className="h-[3.5rem]"
                            />
                          </div>
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>

                  <TooltipProvider>
                    <Tooltip delayDuration={0}>
                      <TooltipTrigger asChild>
                        <div className="flex items-center flex-shrink-0">
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-8 px-2 flex items-center gap-2 whitespace-nowrap"
                          >
                            <PieChart size={14} />
                            <Input
                              type="number"
                              value={sampleSize || ""}
                              onChange={(e) => {
                                const value = e.target.value;
                                setSampleSize(
                                  value === "" ? null : parseInt(value, 10)
                                );
                              }}
                              className="w-12 h-6 text-xs border-0 p-0 focus-visible:ring-0"
                              placeholder="All"
                            />
                          </Button>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent side="bottom">
                        <p>Run pipeline on a sample of documents</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>

                <div className="flex items-center border-l pl-2 flex-shrink-0">
                  <div className="flex items-center space-x-1">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => fileInputRef.current?.click()}
                            className="h-8 w-8"
                          >
                            <FileUp size={16} />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent side="bottom">
                          <p>Load from YAML</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                    <Input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileUpload}
                      accept=".yaml,.yml"
                      className="hidden"
                    />
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            size="icon"
                            variant="ghost"
                            onClick={() => handleExport()}
                            className="h-8 w-8"
                          >
                            <Download size={16} />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent side="bottom">
                          <p>Save to YAML</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => setIsSettingsOpen(true)}
                      className="h-8 w-8"
                    >
                      <Settings size={16} />
                    </Button>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex space-x-3 flex-shrink-0">
              <AddOperationDropdown
                onAddOperation={handleAddOperation}
                trigger={
                  <Button
                    size="sm"
                    variant="outline"
                    className="rounded-sm whitespace-nowrap"
                  >
                    Add Operation <Plus size={16} className="ml-2" />
                  </Button>
                }
              />

              <div className="flex space-x-2 border-l pl-3">
                <Button
                  size="sm"
                  variant="destructive"
                  className="rounded-sm whitespace-nowrap"
                  onClick={handleStop}
                  disabled={!isLoadingOutputs}
                >
                  <StopCircle size={16} className="mr-2" />
                  Stop
                </Button>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        size="sm"
                        variant="secondary"
                        className="rounded-sm bg-secondary hover:bg-secondary/90 text-secondary-foreground font-medium whitespace-nowrap"
                        onClick={() => onRunAll(true)}
                        disabled={isLoadingOutputs}
                      >
                        {isLoadingOutputs ? (
                          <Loader2 size={16} className="mr-2 animate-spin" />
                        ) : (
                          <RefreshCw size={16} className="mr-2" />
                        )}
                        Run Fresh
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="w-72">
                      <p>Run pipeline after clearing all cached results</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <Button
                  size="sm"
                  variant="default"
                  className="rounded-sm whitespace-nowrap"
                  disabled={isLoadingOutputs}
                  onClick={() => onRunAll(false)}
                >
                  {isLoadingOutputs ? (
                    <Loader2 size={16} className="mr-2 animate-spin" />
                  ) : (
                    <Play size={16} className="mr-2" />
                  )}
                  Run
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto min-h-0 p-2">
        <div className="space-y-2">
          {operations.map((op, index) => (
            <OperationCard key={op.id} index={index} id={op.id} />
          ))}
          <AddOperationDropdown
            onAddOperation={handleAddOperation}
            trigger={
              <Button
                variant="outline"
                className="w-full border-dashed h-16 hover:border-primary hover:bg-accent/50 transition-colors"
              >
                <Plus className="mr-2 h-4 w-4" />
                Add Operation
              </Button>
            }
          />
        </div>
      </div>
      <PipelineSettings
        isOpen={isSettingsOpen}
        onOpenChange={setIsSettingsOpen}
        pipelineName={pipelineName}
        setPipelineName={setPipelineName}
        currentFile={currentFile}
        setCurrentFile={setCurrentFile}
        defaultModel={defaultModel}
        setDefaultModel={setDefaultModel}
        optimizerModel={optimizerModel}
        setOptimizerModel={setOptimizerModel}
        autoOptimizeCheck={autoOptimizeCheck}
        setAutoOptimizeCheck={setAutoOptimizeCheck}
        files={files}
        apiKeys={apiKeys}
        extraPipelineSettings={extraPipelineSettings}
        setExtraPipelineSettings={setExtraPipelineSettings}
      />
      <OptimizationDialog
        isOpen={optimizationDialog.isOpen}
        content={optimizationDialog.content}
        prompt={optimizationDialog.prompt}
        operationName={optimizationDialog.operationName}
        inputData={optimizationDialog.inputData}
        outputData={optimizationDialog.outputData}
        onOpenChange={(open) =>
          setOptimizationDialog((prev) => ({ ...prev, isOpen: open }))
        }
        onDecompose={handleOptimizeFromDialog}
      />
    </div>
  );
};

export default PipelineGUI;
