import React, { useState, useCallback, useEffect, useRef } from "react";
import { DropResult } from "react-beautiful-dnd";
import yaml from "js-yaml";
import { Operation, File } from "@/app/types";
import { Droppable, DragDropContext } from "react-beautiful-dnd";
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
  Plus,
  ChevronDown,
  Play,
  Settings,
  PieChart,
  RefreshCw,
  Download,
  FileUp,
  Save,
  Loader2,
  StopCircle,
  Brain,
} from "lucide-react";
import { usePipelineContext } from "@/contexts/PipelineContext";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useToast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useWebSocket } from "@/contexts/WebSocketContext";
import { Input } from "@/components/ui/input";
import path from "path";
import { schemaDictToItemSet } from "./utils";
import { v4 as uuidv4 } from "uuid";
import { useOptimizeCheck } from "@/hooks/useOptimizeCheck";
import { canBeOptimized } from "@/lib/utils";
import { Switch } from "./ui/switch";
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
    <HoverCard openDelay={200}>
      <HoverCardTrigger asChild>
        <div className="relative w-full">
          <DropdownMenuItem onClick={onClick} className="w-full cursor-help">
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

const PipelineGUI: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const {
    operations,
    setOperations,
    pipelineName,
    setPipelineName,
    sampleSize,
    setSampleSize,
    numOpRun,
    setNumOpRun,
    currentFile,
    setCurrentFile,
    setFiles,
    setOutput,
    isLoadingOutputs,
    setIsLoadingOutputs,
    files,
    setCost,
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
  } = usePipelineContext();
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [tempPipelineName, setTempPipelineName] = useState(pipelineName);
  const [tempAutoOptimizeCheck, setTempAutoOptimizeCheck] =
    useState(autoOptimizeCheck);
  const [tempOptimizerModel, setTempOptimizerModel] = useState(optimizerModel);
  const [tempSampleSize, setTempSampleSize] = useState(
    sampleSize?.toString() || ""
  );
  const [tempCurrentFile, setTempCurrentFile] = useState<File | null>(
    currentFile
  );
  const [tempDefaultModel, setTempDefaultModel] = useState(defaultModel);
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
  }>({
    isOpen: false,
    content: "",
    prompt: undefined,
    operationName: undefined,
  });

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
                type === "parallel_map"
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
        toast({
          title: "Error",
          description: lastMessage.data,
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
      setTempPipelineName(pipelineName);
    }
  }, [pipelineName]);

  useEffect(() => {
    if (autoOptimizeCheck) {
      setTempAutoOptimizeCheck(autoOptimizeCheck);
    }
  }, [autoOptimizeCheck]);

  useEffect(() => {
    if (defaultModel) {
      setTempDefaultModel(defaultModel);
    }
  }, [defaultModel]);

  useEffect(() => {
    if (currentFile) {
      setTempCurrentFile(currentFile);
    }
  }, [currentFile]);

  useEffect(() => {
    if (optimizerModel) {
      setTempOptimizerModel(optimizerModel);
    }
  }, [optimizerModel]);

  useEffect(() => {
    if (sampleSize) {
      setTempSampleSize(sampleSize.toString());
    }
  }, [sampleSize]);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const content = e.target?.result;
        if (typeof content === "string") {
          try {
            const yamlFileName = file.name.split("/").pop()?.split(".")[0];
            const yamlContent = yaml.load(content) as any;
            setOperations([]);

            // Update PipelineContext with the loaded YAML data
            setOperations(
              (yamlContent.operations || []).map((op: any) => {
                const {
                  id,
                  llmType,
                  type,
                  name,
                  prompt,
                  output,
                  validate,
                  sample,
                  ...otherKwargs
                } = op;

                // If the operation type is 'reduce', ensure reduce_key is a list
                if (type === "reduce" && otherKwargs.reduce_key) {
                  otherKwargs.reduce_key = Array.isArray(otherKwargs.reduce_key)
                    ? otherKwargs.reduce_key
                    : [otherKwargs.reduce_key];
                }

                return {
                  id: id || uuidv4(),
                  llmType:
                    type === "map" ||
                    type === "reduce" ||
                    type === "resolve" ||
                    type === "filter" ||
                    type === "parallel_map"
                      ? "LLM"
                      : "non-LLM",
                  type: type as Operation["type"],
                  name: name || "Untitled Operation",
                  prompt,
                  output: output
                    ? {
                        schema: schemaDictToItemSet(output.schema),
                      }
                    : undefined,
                  validate,
                  sample,
                  otherKwargs,
                  visibility: true,
                } as Operation;
              })
            );
            setPipelineName(yamlFileName || "Untitled Pipeline");
            setSampleSize(yamlContent.operations?.[0]?.sample || null);
            setDefaultModel(yamlContent.default_model || "gpt-4o-mini");

            // Set current file if it exists in the YAML
            // Look for paths in all datasets
            const datasetPaths = Object.values(yamlContent.datasets || {})
              .filter((dataset: any) => dataset.type === "file" && dataset.path)
              .map((dataset: any) => dataset.path);

            if (datasetPaths.length > 0) {
              const newFiles = datasetPaths.map((filePath) => ({
                name: path.basename(filePath),
                path: filePath,
                type: "json",
              }));

              setFiles((prevFiles: File[]) => {
                const uniqueNewFiles = newFiles
                  .filter(
                    (newFile) =>
                      !prevFiles.some(
                        (prevFile) => prevFile.path === newFile.path
                      )
                  )
                  .map((file) => ({
                    ...file,
                    type: "json" as const, // Explicitly type as literal "json"
                  }));
                return [...prevFiles, ...uniqueNewFiles];
              });

              // Set the first file as current if no current file exists
              if (!currentFile) {
                setCurrentFile({ ...newFiles[0], type: "json" });
              }
            }

            toast({
              title: "Pipeline Loaded",
              description:
                "Your pipeline configuration has been loaded successfully.",
              duration: 3000,
            });
          } catch (error) {
            console.error("Error parsing YAML:", error);
            toast({
              title: "Error",
              description: "Failed to parse the uploaded YAML file.",
              variant: "destructive",
            });
          }
        }
      };
      reader.readAsText(file);
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
          namespace,
          system_prompt: systemPrompt,
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
      const lastOpIndex = operations.length - 1;
      if (lastOpIndex < 0) return;

      const lastOperation = operations[lastOpIndex];
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
            namespace,
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
      name: `${name} ${numOpRun}`,
      visibility: true,
    };
    setOperations([...operations, newOperation]);
  };

  const handleSettingsSave = () => {
    setPipelineName(tempPipelineName);
    setSampleSize(
      tempSampleSize === ""
        ? null
        : tempSampleSize === null
        ? null
        : parseInt(tempSampleSize, 10)
    );
    setCurrentFile(tempCurrentFile);
    setDefaultModel(tempDefaultModel);
    setIsSettingsOpen(false);
    setOptimizerModel(tempOptimizerModel);
    setAutoOptimizeCheck(tempAutoOptimizeCheck);
  };

  const handleDragEnd = (result: DropResult) => {
    if (!result.destination) return;

    const { source, destination } = result;

    if (
      source.droppableId === "operations" &&
      destination.droppableId === "operations"
    ) {
      setOperations((prevOperations) => {
        const newOperations = Array.from(prevOperations);
        const [removed] = newOperations.splice(source.index, 1);
        newOperations.splice(destination.index, 0, removed);
        return newOperations;
      });
    }
  };

  const handleStop = () => {
    sendMessage("kill");

    if (readyState === WebSocket.CLOSED && isLoadingOutputs) {
      setIsLoadingOutputs(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-none p-2 bg-white border-b sticky top-0 z-10">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <h2 className="text-sm font-bold uppercase">
              {pipelineName.toUpperCase()}
            </h2>
            {sampleSize && (
              <TooltipProvider delayDuration={0}>
                <Tooltip>
                  <TooltipTrigger>
                    <div className="flex items-center cursor-help">
                      <PieChart size={16} className="text-primary mr-2" />
                      <span className="text-xs text-primary">
                        {sampleSize} samples
                      </span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-[200px]">
                    <p>
                      Pipeline will run on a sample of {sampleSize} random
                      documents.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
            <div className="flex items-center space-x-0">
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
                  <TooltipContent>
                    <p>Initialize from config file</p>
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
                  <TooltipContent>
                    <p>Download pipeline config file</p>
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

              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 flex items-center gap-1"
                  >
                    <Brain size={14} className="text-primary" />
                    <span className="text-xs text-primary">
                      Set system prompts
                    </span>
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
            </div>
          </div>
          <div className="flex space-x-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button size="sm" className="rounded-sm">
                  <Plus size={16} className="mr-2" /> Add Operation{" "}
                  <ChevronDown size={16} className="ml-2" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuLabel>LLM Operations</DropdownMenuLabel>
                <OperationMenuItem
                  name="Map"
                  description="Transforms each input item for complex data processing and insight extraction. 1 to 1 operation (each document gets one result, but the output of the operation can be any type, like a list)."
                  onClick={() =>
                    handleAddOperation("LLM", "map", "Untitled Map")
                  }
                />
                <OperationMenuItem
                  name="Reduce"
                  description="Aggregates data by key for summarization or folding. Many to 1 operation (many documents get combined into one result)."
                  onClick={() =>
                    handleAddOperation("LLM", "reduce", "Untitled Reduce")
                  }
                />
                <OperationMenuItem
                  name="Resolve"
                  description="Identifies and merges duplicate entities for data consistency. Keeps the same number of documents; just resolves values."
                  onClick={() =>
                    handleAddOperation("LLM", "resolve", "Untitled Resolve")
                  }
                />
                <OperationMenuItem
                  name="Filter"
                  description="Selectively includes or excludes data based on specific conditions. This is like a map operation, but with a boolean output schema. The size of your dataset may decrease, as documents that evaluate to false based on the prompt will be dropped from the dataset."
                  onClick={() =>
                    handleAddOperation("LLM", "filter", "Untitled Filter")
                  }
                />
                <OperationMenuItem
                  name="Parallel Map"
                  description="Like a Map operation, but processes multiple documents in parallel for improved performance. Best used when documents can be processed independently."
                  onClick={() =>
                    handleAddOperation(
                      "LLM",
                      "parallel_map",
                      "Untitled Parallel Map"
                    )
                  }
                />
                <DropdownMenuSeparator />
                <DropdownMenuLabel>Non-LLM Operations</DropdownMenuLabel>
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation("non-LLM", "unnest", "Untitled Unnest")
                  }
                >
                  Unnest
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation("non-LLM", "split", "Untitled Split")
                  }
                >
                  Split
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation("non-LLM", "gather", "Untitled Gather")
                  }
                >
                  Gather
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation("non-LLM", "sample", "Untitled Sample")
                  }
                >
                  Sample
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuLabel>Code Operations</DropdownMenuLabel>
                <OperationMenuItem
                  name="Code Map"
                  description="Like the LLM Map operation, but uses a Python function instead of an LLM. Write custom Python code to transform each document."
                  onClick={() =>
                    handleAddOperation(
                      "non-LLM",
                      "code_map",
                      "Untitled Code Map"
                    )
                  }
                />
                <OperationMenuItem
                  name="Code Reduce"
                  description="Like the LLM Reduce operation, but uses a Python function instead of an LLM. Write custom Python code to aggregate multiple documents into one."
                  onClick={() =>
                    handleAddOperation(
                      "non-LLM",
                      "code_reduce",
                      "Untitled Code Reduce"
                    )
                  }
                />
                <OperationMenuItem
                  name="Code Filter"
                  description="Like the LLM Filter operation, but uses a Python function instead of an LLM. Write custom Python code to determine which documents to keep."
                  onClick={() =>
                    handleAddOperation(
                      "non-LLM",
                      "code_filter",
                      "Untitled Code Filter"
                    )
                  }
                />
              </DropdownMenuContent>
            </DropdownMenu>
            <div className="flex space-x-2">
              <Button
                size="sm"
                variant="destructive"
                className="rounded-sm"
                onClick={handleStop}
                disabled={!isLoadingOutputs}
              >
                <StopCircle size={16} className="mr-2" />
                Stop Pipeline
              </Button>
              <Button
                size="sm"
                className="rounded-sm"
                onClick={() => onRunAll(true)}
                disabled={isLoadingOutputs}
              >
                {isLoadingOutputs ? (
                  <Loader2 size={16} className="mr-2 animate-spin" />
                ) : (
                  <RefreshCw size={16} className="mr-2" />
                )}
                Clear Cache and Run
              </Button>
              <Button
                size="sm"
                className="rounded-sm"
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
      <div className="flex-1 overflow-y-auto min-h-0 p-2">
        <DragDropContext onDragEnd={handleDragEnd}>
          <Droppable droppableId="operations" type="operation">
            {(provided, snapshot) => (
              <div
                {...provided.droppableProps}
                ref={provided.innerRef}
                className={`space-y-2 ${
                  snapshot.isDraggingOver ? "bg-gray-50" : ""
                }`}
              >
                {operations.map((op, index) => (
                  <OperationCard key={op.id} index={index} />
                ))}
                {provided.placeholder}
              </div>
            )}
          </Droppable>
        </DragDropContext>
      </div>
      <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Pipeline Settings</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="name" className="text-right">
                Name
              </Label>
              <Input
                id="name"
                value={tempPipelineName}
                onChange={(e) => setTempPipelineName(e.target.value)}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="sampling" className="text-right">
                Sample Size
              </Label>
              <Input
                id="sampling"
                type="number"
                value={tempSampleSize}
                onChange={(e) => setTempSampleSize(e.target.value)}
                placeholder="None"
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="currentFile" className="text-right">
                Dataset JSON
              </Label>
              <Select
                value={tempCurrentFile?.path || ""}
                onValueChange={(value) =>
                  setTempCurrentFile(
                    files.find((file) => file.path === value) || null
                  )
                }
              >
                <SelectTrigger className="col-span-3">
                  <SelectValue placeholder="Select a file" />
                </SelectTrigger>
                <SelectContent>
                  {files
                    .filter((file) => file.type === "json")
                    .map((file) => (
                      <SelectItem key={file.path} value={file.path}>
                        {file.name}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="defaultModel" className="text-right">
                Default Model
              </Label>
              <Input
                id="defaultModel"
                value={tempDefaultModel}
                onChange={(e) => setTempDefaultModel(e.target.value)}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="optimize" className="text-right">
                Optimizer Model
              </Label>
              <Select
                value={tempOptimizerModel}
                onValueChange={(value) => setTempOptimizerModel(value)}
              >
                <SelectTrigger className="col-span-3">
                  <SelectValue placeholder="Select optimizer model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gpt-4o">gpt-4o</SelectItem>
                  <SelectItem value="gpt-4o-mini">gpt-4o-mini</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="autoOptimize" className="text-right">
                Automatically Check Whether to Optimize
              </Label>
              <Switch
                id="autoOptimize"
                checked={tempAutoOptimizeCheck}
                onCheckedChange={(checked) => setTempAutoOptimizeCheck(checked)}
                className="col-span-3"
              />
            </div>
          </div>
          <DialogFooter>
            <Button onClick={handleSettingsSave}>Save changes</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
      />
    </div>
  );
};

export default PipelineGUI;
