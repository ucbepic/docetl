import React, {
  useReducer,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { Draggable } from "react-beautiful-dnd";
import {
  GripVertical,
  Trash2,
  Zap,
  Settings,
  ListCollapse,
  Wand2,
  ChevronDown,
  Eye,
  EyeOff,
} from "lucide-react";
import { Operation, SchemaItem } from "@/app/types";
import { usePipelineContext } from "@/contexts/PipelineContext";
import { useToast } from "@/hooks/use-toast";
import { Skeleton } from "@/components/ui/skeleton";
import { debounce } from "lodash";
import { Guardrails, GleaningConfig } from "./operations/args";
import createOperationComponent from "./operations/components";
import { useWebSocket } from "@/contexts/WebSocketContext";
import { Badge } from "./ui/badge";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { AIEditPopover } from "@/components/AIEditPopover";
import { canBeOptimized } from "@/lib/utils";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { PromptImprovementDialog } from "@/components/PromptImprovementDialog";

// Separate components
const OperationHeader: React.FC<{
  name: string;
  type: string;
  llmType: string;
  disabled: boolean;
  currOp: boolean;
  expanded: boolean;
  visibility: boolean;
  optimizeResult?: string;
  onEdit: (name: string) => void;
  onDelete: () => void;
  onRunOperation: () => void;
  onToggleSettings: () => void;
  onShowOutput: () => void;
  onOptimize: () => void;
  onAIEdit: (instruction: string) => void;
  onToggleExpand: () => void;
  onToggleVisibility: () => void;
  onImprovePrompt: () => void;
}> = React.memo(
  ({
    name,
    type,
    llmType,
    disabled,
    currOp,
    expanded,
    visibility,
    optimizeResult,
    onEdit,
    onDelete,
    onRunOperation,
    onToggleSettings,
    onShowOutput,
    onOptimize,
    onAIEdit,
    onToggleExpand,
    onToggleVisibility,
    onImprovePrompt,
  }) => {
    const [isEditing, setIsEditing] = useState(false);
    const [editedName, setEditedName] = useState(name);

    const handleEditClick = () => {
      setIsEditing(true);
      setEditedName(name);
    };

    const handleEditComplete = () => {
      setIsEditing(false);
      onEdit(editedName);
    };

    return (
      <div className="relative flex items-center justify-between py-3 px-4">
        {/* Left side buttons */}
        <div
          className={`flex space-x-1 absolute left-1 ${
            !visibility ? "opacity-50" : ""
          }`}
        >
          <Button
            variant="ghost"
            size="sm"
            className="p-0.25 h-6 w-6"
            onClick={onToggleExpand}
          >
            <ChevronDown
              size={14}
              className={`text-gray-500 transform transition-transform ${
                expanded ? "rotate-180" : ""
              }`}
            />
          </Button>
          <div className="relative">
            <div className="flex items-center space-x-1">
              <Badge variant={currOp ? "default" : "secondary"}>{type}</Badge>
              {canBeOptimized(type) && (
                <div className="relative">
                  {optimizeResult !== undefined && (
                    <div
                      className={`absolute -top-1 -right-1 h-2 w-2 rounded-full ${
                        optimizeResult === null || optimizeResult === ""
                          ? "bg-gray-300"
                          : "bg-red-500"
                      }`}
                    />
                  )}
                  <HoverCard>
                    <HoverCardTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="p-0.25 h-6 w-6"
                        onClick={onOptimize}
                        disabled={disabled}
                      >
                        <Zap
                          size={14}
                          className={
                            optimizeResult === undefined ||
                            optimizeResult === null ||
                            optimizeResult === ""
                              ? "text-gray-400"
                              : "text-red-500"
                          }
                        />
                      </Button>
                    </HoverCardTrigger>
                    <HoverCardContent className="w-80 p-2">
                      <p className="text-sm">
                        {optimizeResult === undefined || optimizeResult === null
                          ? "Determining whether to recommend a decomposition..."
                          : optimizeResult === ""
                          ? "No decomposition recommended"
                          : "Decomposition recommended because: " +
                            optimizeResult}
                      </p>
                    </HoverCardContent>
                  </HoverCard>
                </div>
              )}
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="p-0.25 h-6 w-6"
            onClick={onToggleSettings}
          >
            <Settings size={14} className="text-gray-500" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="flex items-center gap-1 px-2 h-6"
            disabled={disabled}
            onClick={onShowOutput}
          >
            <ListCollapse size={14} className="text-primary" />
            <span className="text-xs text-primary">Show outputs</span>
          </Button>
          {llmType === "LLM" && (
            <Button
              variant="ghost"
              size="sm"
              className="flex items-center gap-1 px-2 h-6"
              disabled={disabled}
              onClick={onImprovePrompt}
            >
              <Wand2 size={14} className="text-primary" />
              <span className="text-xs text-primary">Improve prompt</span>
            </Button>
          )}
        </div>

        {/* Centered title */}
        <div
          className={`flex-grow flex justify-center mx-20 ${
            !visibility ? "opacity-50" : ""
          }`}
        >
          {isEditing ? (
            <Input
              value={editedName}
              onChange={(e) => setEditedName(e.target.value)}
              onBlur={handleEditComplete}
              onKeyPress={(e) => e.key === "Enter" && handleEditComplete()}
              className="text-sm font-medium max-w-[200px] font-mono text-center"
              autoFocus
            />
          ) : (
            <span
              className={`text-sm font-medium cursor-pointer truncate max-w-[200px] ${
                llmType === "LLM"
                  ? "bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text"
                  : ""
              }`}
              onClick={handleEditClick}
            >
              {name}
            </span>
          )}
        </div>

        {/* Right side buttons */}
        <div className="absolute right-1 flex items-center space-x-0">
          <Button
            variant={visibility ? "ghost" : "default"}
            size="sm"
            className={`flex items-center gap-1 px-2 h-6 ${
              !visibility
                ? "bg-green-100 hover:bg-green-200"
                : "hover:bg-green-100"
            }`}
            onClick={onToggleVisibility}
          >
            {visibility ? (
              <>
                <EyeOff size={14} className="text-gray-500" />
                <span className="text-xs text-gray-500">Skip operation</span>
              </>
            ) : (
              <>
                <Eye size={14} className="text-green-700" />
                <span className="text-xs font-medium text-green-700">
                  Include operation
                </span>
              </>
            )}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onDelete}
            className={`hover:bg-red-100 p-1 h-7 w-7 ${
              !visibility ? "opacity-50" : ""
            }`}
          >
            <Trash2 size={15} className="text-red-500" />
          </Button>
        </div>
      </div>
    );
  }
);
OperationHeader.displayName = "OperationHeader";

const SettingsModal: React.FC<{
  opName: string;
  opType: string;
  isOpen: boolean;
  onClose: () => void;
  otherKwargs: Record<string, string>;
  onSettingsSave: (newSettings: Record<string, string>) => void;
}> = React.memo(
  ({ opName, opType, isOpen, onClose, otherKwargs, onSettingsSave }) => {
    const [localSettings, setLocalSettings] = React.useState<
      Array<{ id: number; key: string; value: string }>
    >(
      Object.entries(otherKwargs).map(([key, value], index) => ({
        id: index,
        key,
        value,
      }))
    );

    useEffect(() => {
      setLocalSettings(
        Object.entries(otherKwargs).map(([key, value], index) => ({
          id: index,
          key,
          value,
        }))
      );
    }, [otherKwargs]);

    const handleSettingsChange = (
      id: number,
      newKey: string,
      newValue: string
    ) => {
      setLocalSettings((prev) =>
        prev.map((setting) =>
          setting.id === id
            ? { ...setting, key: newKey, value: newValue }
            : setting
        )
      );
    };

    const addSetting = () => {
      setLocalSettings((prev) => [
        ...prev,
        { id: prev.length, key: "", value: "" },
      ]);
    };

    const removeSetting = (id: number) => {
      setLocalSettings((prev) => prev.filter((setting) => setting.id !== id));
    };

    const handleSave = () => {
      const newSettings = localSettings.reduce((acc, { key, value }) => {
        if (key !== "" && value !== "") {
          acc[key] = value;
        }
        return acc;
      }, {} as Record<string, string>);
      onSettingsSave(newSettings);
      onClose();
    };

    const isValidSettings = () => {
      const keys = localSettings.map(({ key }) => key);
      return (
        localSettings.every(({ key, value }) => key !== "" && value !== "") &&
        new Set(keys).size === keys.length
      );
    };

    if (!isOpen) return null;

    return (
      <Dialog open={isOpen} onOpenChange={onClose}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle>{opName}</DialogTitle>
            <DialogDescription>
              Add or modify additional arguments for this {opType} operation.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            {localSettings.map(({ id, key, value }) => (
              <div key={id} className="flex items-center gap-4">
                <Input
                  className="flex-grow font-mono"
                  value={key}
                  onChange={(e) =>
                    handleSettingsChange(id, e.target.value, value)
                  }
                  placeholder="Key"
                />
                <Input
                  className="flex-grow font-mono"
                  value={value}
                  onChange={(e) =>
                    handleSettingsChange(id, key, e.target.value)
                  }
                  placeholder="Value"
                />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeSetting(id)}
                >
                  <Trash2 size={15} />
                </Button>
              </div>
            ))}
            <Button onClick={addSetting}>Add Setting</Button>
          </div>
          <DialogFooter>
            <Button onClick={handleSave} disabled={!isValidSettings()}>
              Save
            </Button>
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }
);

// Action types
type Action =
  | { type: "SET_OPERATION"; payload: Operation }
  | { type: "UPDATE_NAME"; payload: string }
  | { type: "UPDATE_PROMPT"; payload: string }
  | { type: "UPDATE_SCHEMA"; payload: SchemaItem[] }
  | { type: "UPDATE_GUARDRAILS"; payload: string[] }
  | { type: "TOGGLE_EDITING" }
  | { type: "TOGGLE_SCHEMA" }
  | { type: "TOGGLE_GUARDRAILS" }
  | { type: "TOGGLE_SETTINGS" }
  | { type: "SET_RUN_INDEX"; payload: number }
  | { type: "UPDATE_SETTINGS"; payload: Record<string, string> }
  | { type: "TOGGLE_EXPAND" }
  | {
      type: "UPDATE_GLEANINGS";
      payload: { num_rounds: number; validation_prompt: string };
    }
  | { type: "TOGGLE_GLEANINGS" };

// State type
type State = {
  operation: Operation | undefined;
  isEditing: boolean;
  isSchemaExpanded: boolean;
  isGuardrailsExpanded: boolean;
  isSettingsOpen: boolean;
  isExpanded: boolean;
  isGleaningsExpanded: boolean;
};

// Reducer function
function operationReducer(state: State, action: Action): State {
  switch (action.type) {
    case "SET_OPERATION":
      return { ...state, operation: action.payload };
    case "UPDATE_NAME":
      return state.operation
        ? { ...state, operation: { ...state.operation, name: action.payload } }
        : state;
    case "UPDATE_PROMPT":
      return state.operation
        ? {
            ...state,
            operation: { ...state.operation, prompt: action.payload },
          }
        : state;
    case "UPDATE_SCHEMA":
      return state.operation
        ? {
            ...state,
            operation: {
              ...state.operation,
              output: {
                ...state.operation.output,
                schema: action.payload,
              },
            },
          }
        : state;

    case "UPDATE_GUARDRAILS":
      return state.operation
        ? {
            ...state,
            operation: { ...state.operation, validate: action.payload },
          }
        : state;
    case "TOGGLE_EDITING":
      return { ...state, isEditing: !state.isEditing };
    case "TOGGLE_SCHEMA":
      return { ...state, isSchemaExpanded: !state.isSchemaExpanded };
    case "TOGGLE_GUARDRAILS":
      return { ...state, isGuardrailsExpanded: !state.isGuardrailsExpanded };
    case "TOGGLE_SETTINGS":
      return { ...state, isSettingsOpen: !state.isSettingsOpen };
    case "UPDATE_SETTINGS":
      return state.operation
        ? {
            ...state,
            operation: { ...state.operation, otherKwargs: action.payload },
          }
        : state;
    case "SET_RUN_INDEX":
      return state.operation
        ? {
            ...state,
            operation: { ...state.operation, runIndex: action.payload },
          }
        : state;
    case "TOGGLE_EXPAND":
      return { ...state, isExpanded: !state.isExpanded };
    case "UPDATE_GLEANINGS":
      return state.operation
        ? {
            ...state,
            operation: { ...state.operation, gleaning: action.payload },
          }
        : state;
    case "TOGGLE_GLEANINGS":
      return { ...state, isGleaningsExpanded: !state.isGleaningsExpanded };
    default:
      return state;
  }
}

// Initial state
const initialState: State = {
  operation: undefined,
  isEditing: false,
  isSchemaExpanded: true,
  isGuardrailsExpanded: false,
  isSettingsOpen: false,
  isExpanded: true,
  isGleaningsExpanded: false,
};

// Main component
export const OperationCard: React.FC<{ index: number }> = ({ index }) => {
  const [state, dispatch] = useReducer(operationReducer, initialState);
  const {
    operation,
    isEditing,
    isSchemaExpanded,
    isGuardrailsExpanded,
    isSettingsOpen,
    isExpanded,
    isGleaningsExpanded,
  } = state;

  const {
    output: pipelineOutput,
    setOutput,
    isLoadingOutputs,
    setIsLoadingOutputs,
    numOpRun,
    setNumOpRun,
    currentFile,
    operations,
    setOperations,
    pipelineName,
    sampleSize,
    setCost,
    defaultModel,
    optimizerModel,
    setTerminalOutput,
  } = usePipelineContext();
  const { toast } = useToast();

  const operationRef = useRef(operation);
  const { connect, sendMessage, lastMessage, readyState, disconnect } =
    useWebSocket();

  useEffect(() => {
    operationRef.current = operation;
  }, [operation]);

  useEffect(() => {
    dispatch({ type: "SET_OPERATION", payload: operations[index] });

    // Also dispatch the runIndex update
    if (operations[index].runIndex !== undefined) {
      dispatch({ type: "SET_RUN_INDEX", payload: operations[index].runIndex });
    }
  }, [operations, index]);

  const debouncedUpdate = useCallback(
    debounce(() => {
      if (operationRef.current) {
        const updatedOperation = { ...operationRef.current };
        setOperations((prev) =>
          prev.map((op) =>
            op.id === updatedOperation.id ? updatedOperation : op
          )
        );
      }
    }, 500),
    [setOperations]
  );

  const handleOperationUpdate = useCallback(
    (updatedOperation: Operation) => {
      dispatch({ type: "SET_OPERATION", payload: updatedOperation });
      debouncedUpdate();
    },
    [debouncedUpdate]
  );

  const handleRunOperation = useCallback(async () => {
    if (!operation) return;
    setIsLoadingOutputs(true);
    setNumOpRun((prevNum) => {
      const newNum = prevNum + 1;
      dispatch({ type: "SET_RUN_INDEX", payload: newNum });
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
          operation_id: operation.id,
          name: pipelineName,
          sample_size: sampleSize,
        }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const { filePath, inputPath, outputPath } = await response.json();

      setOutput({
        operationId: operation.id,
        path: outputPath,
        inputPath: inputPath,
      });

      // Ensure the WebSocket is connected before sending the message
      await connect();

      sendMessage({
        yaml_config: filePath,
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
  }, [
    operation,
    currentFile,
    operations,
    setIsLoadingOutputs,
    setNumOpRun,
    sendMessage,
    readyState,
    defaultModel,
    pipelineName,
    sampleSize,
  ]);

  const handleSettingsSave = useCallback(
    (newSettings: Record<string, string>) => {
      dispatch({ type: "UPDATE_SETTINGS", payload: newSettings });
      if (operation) {
        const updatedOperation = { ...operation, otherKwargs: newSettings };
        setOperations((prev) =>
          prev.map((op) =>
            op.id === updatedOperation.id ? updatedOperation : op
          )
        );
      }
    },
    [operation, setOperations]
  );

  const handleSchemaUpdate = (newSchema: SchemaItem[]) => {
    dispatch({ type: "UPDATE_SCHEMA", payload: newSchema });
    debouncedUpdate();
  };

  const onOptimize = useCallback(async () => {
    if (!operation) return;

    try {
      // Clear the output
      setTerminalOutput("");
      setIsLoadingOutputs(true);

      // Write pipeline config
      const response = await fetch("/api/writePipelineConfig", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          default_model: defaultModel,
          data: { path: currentFile?.path || "" },
          operations,
          operation_id: operation.id,
          name: pipelineName,
          sample_size: sampleSize,
          optimize: true,
        }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const { filePath } = await response.json();

      // Ensure WebSocket is connected
      await connect();

      // Send message to run the pipeline
      sendMessage({
        yaml_config: filePath,
        optimize: true,
        optimizer_model: optimizerModel,
      });
    } catch (error) {
      console.error("Error optimizing operation:", error);
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
      // Close the WebSocket connection
      disconnect();
    }
  }, [operation]);

  const onShowOutput = useCallback(async () => {
    if (!operation) return;

    try {
      const response = await fetch("/api/getInputOutput", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          default_model: defaultModel,
          data: { path: currentFile?.path || "" },
          operations,
          operation_id: operation.id,
          name: pipelineName,
          sample_size: sampleSize,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get input and output paths");
      }

      const { inputPath, outputPath } = await response.json();

      setOutput({
        operationId: operation.id,
        path: outputPath,
        inputPath: inputPath,
      });
    } catch (error) {
      console.error("Error fetching input and output paths:", error);
      toast({
        title: "Error",
        description: "Failed to get input and output paths",
        variant: "destructive",
      });
    }
  }, [
    operation,
    defaultModel,
    currentFile,
    operations,
    pipelineName,
    sampleSize,
    setOutput,
    toast,
  ]);

  const handleAIEdit = useCallback(
    async (instruction: string) => {
      if (!operation) return;

      try {
        const response = await fetch("/api/edit", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            operation,
            instruction,
          }),
        });

        if (!response.ok) {
          throw new Error("Failed to apply AI edit");
        }

        const updatedOperation = await response.json();
        handleOperationUpdate(updatedOperation);

        toast({
          title: "Success",
          description: "Operation updated successfully",
        });
      } catch (error) {
        console.error("Error applying AI edit:", error);
        toast({
          title: "Error",
          description: "Failed to apply AI edit",
          variant: "destructive",
        });
      }
    },
    [operation, handleOperationUpdate, toast]
  );

  const handleGuardrailsUpdate = useCallback(
    (newGuardrails: string[]) => {
      dispatch({ type: "UPDATE_GUARDRAILS", payload: newGuardrails });
      debouncedUpdate();
    },
    [debouncedUpdate]
  );

  const handleGleaningsUpdate = useCallback(
    (newGleanings: { num_rounds: number; validation_prompt: string }) => {
      dispatch({ type: "UPDATE_GLEANINGS", payload: newGleanings });
      debouncedUpdate();
    },
    [debouncedUpdate]
  );

  const handleVisibilityToggle = useCallback(() => {
    if (!operation) return;

    const updatedOperation = {
      ...operation,
      visibility:
        operation.visibility === undefined ? false : !operation.visibility,
    };

    handleOperationUpdate(updatedOperation);
  }, [operation, handleOperationUpdate]);

  const [showPromptImprovement, setShowPromptImprovement] = useState(false);

  const handlePromptSave = (
    newPrompt:
      | string
      | { comparison_prompt: string; resolution_prompt: string },
    schemaChanges?: Array<[string, string]>
  ) => {
    if (!operation) return;

    let updatedOperation = { ...operation };

    if (operation.type === "resolve") {
      if (typeof newPrompt === "object") {
        updatedOperation = {
          ...updatedOperation,
          otherKwargs: {
            ...operation.otherKwargs,
            comparison_prompt: newPrompt.comparison_prompt,
            resolution_prompt: newPrompt.resolution_prompt,
          },
        };
      }
    } else {
      if (typeof newPrompt === "string") {
        updatedOperation.prompt = newPrompt;
      }
    }

    // Handle schema changes
    if (schemaChanges?.length && operation.output?.schema) {
      const updatedSchema = operation.output.schema.map((item) => {
        const change = schemaChanges.find(([oldKey]) => oldKey === item.key);
        if (change) {
          return { ...item, key: change[1] };
        }
        return item;
      });

      updatedOperation.output = {
        ...operation.output,
        schema: updatedSchema,
      };
    }

    handleOperationUpdate(updatedOperation);
    toast({
      title: "Success",
      description: `Prompt${
        operation.type === "resolve" ? "s" : ""
      } and schema updated successfully`,
    });
  };

  if (!operation) {
    return <SkeletonCard />;
  }

  return (
    <div className="flex items-start w-full">
      <div className="mr-1 w-8 h-8 flex-shrink-0 flex items-center justify-center bg-gray-100 text-gray-600 font-mono text-xs rounded-sm shadow-sm">
        {isLoadingOutputs ? (
          <div className="animate-spin rounded-full h-3 w-3 border-t-2 border-b-2 border-gray-900"></div>
        ) : operation.runIndex ? (
          <>[{operation.runIndex}]</>
        ) : (
          <>[ ]</>
        )}
      </div>
      <Draggable draggableId={operation.id} index={index} key={operation.id}>
        {(provided) => (
          <Card
            ref={provided.innerRef}
            {...provided.draggableProps}
            className={`mb-2 relative rounded-sm shadow-sm w-full ${
              pipelineOutput?.operationId === operation.id
                ? "bg-white border-blue-500 border-2"
                : "bg-white"
            } ${!operation.visibility ? "opacity-50" : ""}`}
          >
            {/* Move the drag handle div outside of the ml-5 container */}
            <div
              {...provided.dragHandleProps}
              className="absolute left-0 top-0 bottom-0 w-6 flex items-center justify-center cursor-move hover:bg-gray-100 border-r border-gray-100"
            >
              <GripVertical size={14} className="text-gray-400" />
            </div>

            {/* Adjust the left margin to accommodate the drag handle */}
            <div className="ml-6">
              <OperationHeader
                name={operation.name}
                type={operation.type}
                llmType={operation.llmType}
                disabled={isLoadingOutputs || pipelineOutput === undefined}
                currOp={operation.id === pipelineOutput?.operationId}
                expanded={isExpanded}
                visibility={operation.visibility}
                optimizeResult={operation.shouldOptimizeResult}
                onEdit={(name) => {
                  dispatch({ type: "UPDATE_NAME", payload: name });
                  debouncedUpdate();
                }}
                onDelete={() =>
                  setOperations((prev) =>
                    prev.filter((op) => op.id !== operation.id)
                  )
                }
                onRunOperation={handleRunOperation}
                onToggleSettings={() => dispatch({ type: "TOGGLE_SETTINGS" })}
                onShowOutput={onShowOutput}
                onOptimize={onOptimize}
                onAIEdit={handleAIEdit}
                onToggleExpand={() => dispatch({ type: "TOGGLE_EXPAND" })}
                onToggleVisibility={handleVisibilityToggle}
                onImprovePrompt={() => setShowPromptImprovement(true)}
              />
              {isExpanded && operation.visibility !== false && (
                <>
                  <CardContent className="py-2 px-3">
                    {createOperationComponent(
                      operation,
                      handleOperationUpdate,
                      isSchemaExpanded,
                      () => dispatch({ type: "TOGGLE_SCHEMA" })
                    )}
                  </CardContent>
                  {operation.llmType === "LLM" && (
                    <>
                      <Guardrails
                        guardrails={operation.validate || []}
                        onUpdate={handleGuardrailsUpdate}
                        isExpanded={isGuardrailsExpanded}
                        onToggle={() => dispatch({ type: "TOGGLE_GUARDRAILS" })}
                      />
                    </>
                  )}
                  {(operation.type === "map" ||
                    operation.type === "reduce" ||
                    operation.type === "filter") && (
                    <GleaningConfig
                      gleaning={operation.gleaning || null}
                      onUpdate={handleGleaningsUpdate}
                      isExpanded={isGleaningsExpanded}
                      onToggle={() => dispatch({ type: "TOGGLE_GLEANINGS" })}
                    />
                  )}
                </>
              )}
              <SettingsModal
                opName={operation.name}
                opType={operation.type}
                isOpen={isSettingsOpen}
                onClose={() => dispatch({ type: "TOGGLE_SETTINGS" })}
                otherKwargs={operation.otherKwargs || {}}
                onSettingsSave={handleSettingsSave}
              />
              {operation.llmType === "LLM" && (
                <PromptImprovementDialog
                  open={showPromptImprovement}
                  onOpenChange={setShowPromptImprovement}
                  currentOperation={operation}
                  onSave={handlePromptSave}
                />
              )}
            </div>
          </Card>
        )}
      </Draggable>
    </div>
  );
};

const SkeletonCard: React.FC = () => (
  <div className="flex items-start w-full">
    <div className="mr-1 w-8 h-8 flex-shrink-0 flex items-center justify-center bg-gray-200 rounded-sm">
      <Skeleton className="h-3 w-3" />
    </div>
    <Card className="mb-2 relative rounded-sm bg-white shadow-sm w-full">
      <CardHeader className="flex justify-between items-center py-2 px-3">
        <Skeleton className="h-3 w-1/3" />
        <Skeleton className="h-3 w-1/4" />
      </CardHeader>
      <CardContent>
        <Skeleton className="h-16 w-full mb-1" />
        <Skeleton className="h-3 w-2/3" />
      </CardContent>
    </Card>
  </div>
);

// Update the getSystemContent function to handle resolve operations
const getSystemContent = (
  pipelineState: string,
  selectedOperation: Operation
) => `You are a prompt engineering expert. Analyze the current operation's prompt${
  selectedOperation.type === "resolve" ? "s" : ""
} and suggest improvements based on the pipeline state.

Current pipeline state:
${pipelineState}

Focus on the operation named "${selectedOperation.name}" ${
  selectedOperation.type === "resolve"
    ? `with comparison prompt:
${selectedOperation.otherKwargs?.comparison_prompt || ""}

and resolution prompt:
${selectedOperation.otherKwargs?.resolution_prompt || ""}`
    : `with prompt:
${selectedOperation.prompt}`
}

${
  selectedOperation.output?.schema
    ? `
Current output schema keys:
${selectedOperation.output.schema.map((item) => `- ${item.key}`).join("\n")}
`
    : ""
}

IMPORTANT: 
1. ${
  selectedOperation.type === "resolve"
    ? "You must ALWAYS include complete revised prompts wrapped in <comparison_prompt></comparison_prompt> AND <resolve_prompt></resolve_prompt> tags in your response"
    : "You must ALWAYS include a complete revised prompt wrapped in <prompt></prompt> tags in your response"
}, even if you're just responding to feedback.

2. Only suggest schema key changes if absolutely necessary - when the current keys are misleading, incorrect, or ambiguous. If the schema keys are fine, don't suggest changes. Include changes in <schema> tags as a list of "oldkey,newkey" pairs, one per line. Example:
<schema>
misleading_key,accurate_key
ambiguous_name,specific_name
</schema>

When responding:
1. Briefly acknowledge/analyze any feedback (1-2 sentences)
2. ALWAYS provide ${
  selectedOperation.type === "resolve"
    ? "complete revised prompts wrapped in <comparison_prompt></comparison_prompt> AND <resolve_prompt></resolve_prompt> tags"
    : "a complete revised prompt wrapped in <prompt></prompt> tags"
}
3. The prompt${
  selectedOperation.type === "resolve" ? "s" : ""
} should include all previous improvements plus any new changes
4. Make prompts specific and concise:
   - For subjective terms like "detailed" or "comprehensive", provide examples or metrics (e.g. "include 3-5 key points per section")
   - For qualitative instructions like "long output", specify length (e.g. "200-300 words") based on my feedback or provide examples
   - When using adjectives, include a reference point (e.g. "technical like API documentation" vs "simple like a blog post")`;

// Add helper function to extract both prompts for resolve operations
function extractPrompts(text: string): {
  comparisonPrompt?: string;
  resolvePrompt?: string;
  prompt?: string;
} {
  const comparisonPrompt = extractTagContent(text, "comparison_prompt");
  const resolvePrompt = extractTagContent(text, "resolve_prompt");
  const prompt = extractTagContent(text, "prompt");

  return {
    comparisonPrompt,
    resolvePrompt,
    prompt,
  };
}
