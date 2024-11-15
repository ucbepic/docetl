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
    highLevelGoal,
    setHighLevelGoal,
  } = usePipelineContext();
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [tempPipelineName, setTempPipelineName] = useState(pipelineName);
  const [tempAutoOptimizeCheck, setTempAutoOptimizeCheck] =
    useState(autoOptimizeCheck);
  const [tempOptimizerModel, setTempOptimizerModel] = useState(optimizerModel);
  const [tempHighLevelGoal, setTempHighLevelGoal] = useState(highLevelGoal);
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
      // Send toast with should optimize result
      // If should_optimize string is not empty, send toast with should optimize result
      if (result.should_optimize) {
        toast({
          title: `Hey! Consider optimizing ${
            operations[operations.length - 1].name
          }`,
          description: result.should_optimize,
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
    if (highLevelGoal) {
      setTempHighLevelGoal(highLevelGoal);
    }
  }, [highLevelGoal]);

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
              }));

              setFiles((prevFiles: File[]) => {
                const uniqueNewFiles = newFiles.filter(
                  (newFile) =>
                    !prevFiles.some(
                      (prevFile) => prevFile.path === newFile.path
                    )
                );
                return [...prevFiles, ...uniqueNewFiles];
              });

              // Set the first file as current if no current file exists
              if (!currentFile) {
                setCurrentFile(newFiles[0]);
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
    setHighLevelGoal(tempHighLevelGoal);
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
                    <div className="flex items-center">
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
            <div className="flex p-0 space-x-0">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => fileInputRef.current?.click()}
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
              >
                <Settings size={16} />
              </Button>
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
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation("LLM", "map", "Untitled Map")
                  }
                >
                  Map
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation("LLM", "reduce", "Untitled Reduce")
                  }
                >
                  Reduce
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation("LLM", "resolve", "Untitled Resolve")
                  }
                >
                  Resolve
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation("LLM", "filter", "Untitled Filter")
                  }
                >
                  Filter
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation(
                      "LLM",
                      "parallel_map",
                      "Untitled Parallel Map"
                    )
                  }
                >
                  Parallel Map
                </DropdownMenuItem>
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
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation(
                      "non-LLM",
                      "code_map",
                      "Untitled Code Map"
                    )
                  }
                >
                  Code Map
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation(
                      "non-LLM",
                      "code_reduce",
                      "Untitled Code Reduce"
                    )
                  }
                >
                  Code Reduce
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() =>
                    handleAddOperation(
                      "non-LLM",
                      "code_filter",
                      "Untitled Code Filter"
                    )
                  }
                >
                  Code Filter
                </DropdownMenuItem>
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
                Clear and Run
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
              <Label htmlFor="goal" className="text-right">
                High-Level Goal
              </Label>
              <Textarea
                id="goal"
                value={tempHighLevelGoal}
                onChange={(e) => setTempHighLevelGoal(e.target.value)}
                className="col-span-3"
                rows={4}
                placeholder="Describe the high-level goal of your pipeline (e.g., 'I want to find common themes across all the documents' or 'I want to summarize the most important things related to X')..."
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
                  {files.map((file) => (
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
    </div>
  );
};

export default PipelineGUI;
