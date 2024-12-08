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
  Menu,
  Shield,
  Sparkles,
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
interface OperationHeaderProps {
  name: string;
  type: string;
  llmType: string;
  disabled: boolean;
  currOp: boolean;
  expanded: boolean;
  visibility: boolean;
  optimizeResult?: string;
  isGuardrailsExpanded: boolean;
  isGleaningsExpanded: boolean;
  onEdit: (name: string) => void;
  onDelete: () => void;
  onRunOperation: () => void;
  onToggleSettings: () => void;
  onShowOutput: () => void;
  onOptimize: () => void;
  onToggleExpand: () => void;
  onToggleVisibility: () => void;
  onImprovePrompt: () => void;
  onToggleGuardrails: () => void;
  onToggleGleanings: () => void;
}

const OperationHeader: React.FC<OperationHeaderProps> = React.memo(
  ({
    name,
    type,
    llmType,
    disabled,
    currOp,
    expanded,
    visibility,
    optimizeResult,
    isGuardrailsExpanded,
    isGleaningsExpanded,
    onEdit,
    onDelete,
    onToggleSettings,
    onShowOutput,
    onOptimize,
    onToggleExpand,
    onToggleVisibility,
    onImprovePrompt,
    onToggleGuardrails,
    onToggleGleanings,
  }) => {
    const [menuOpen, setMenuOpen] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [editedName, setEditedName] = useState(name);

    return (
      <div className="relative flex items-center py-2 px-4 border-b border-gray-100">
        {/* Operation Type Badge and Optimization Status (The "Noun") */}
        <div className="flex-1 flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Badge variant={currOp ? "default" : "secondary"}>{type}</Badge>

            {canBeOptimized(type) && optimizeResult !== undefined && (
              <HoverCard openDelay={200}>
                <HoverCardTrigger asChild>
                  <div
                    className={`w-2 h-2 rounded-full cursor-help transition-colors
                      ${
                        optimizeResult === null || optimizeResult === ""
                          ? "bg-gray-300"
                          : "bg-amber-500 animate-pulse"
                      }`}
                  />
                </HoverCardTrigger>
                <HoverCardContent className="w-72" side="bottom" align="start">
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium">
                      {optimizeResult === undefined || optimizeResult === null
                        ? "Analyzing Operation"
                        : optimizeResult === ""
                        ? "Decomposition Status"
                        : "Decomposition Recommended"}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {optimizeResult === undefined || optimizeResult === null
                        ? "Analyzing operation complexity..."
                        : optimizeResult === ""
                        ? "No decomposition needed for this operation"
                        : "Recommended decomposition: " + optimizeResult}
                    </p>
                  </div>
                </HoverCardContent>
              </HoverCard>
            )}
          </div>

          {isEditing ? (
            <Input
              value={editedName}
              onChange={(e) => setEditedName(e.target.value)}
              onBlur={() => {
                setIsEditing(false);
                onEdit(editedName);
              }}
              onKeyPress={(e) => {
                if (e.key === "Enter") {
                  setIsEditing(false);
                  onEdit(editedName);
                }
              }}
              className="max-w-[200px] h-7 text-sm font-medium"
              autoFocus
            />
          ) : (
            <span
              className={`text-sm font-medium cursor-default select-none ${
                llmType === "LLM"
                  ? "bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text"
                  : ""
              }`}
              onClick={() => setIsEditing(true)}
            >
              {name}
            </span>
          )}
        </div>

        {/* Action Menu (The "Verb") */}
        <Popover open={menuOpen} onOpenChange={setMenuOpen}>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
              <Menu className="h-4 w-4 text-gray-600" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-56 p-1" align="end">
            <div className="space-y-0.5">
              {/* Core Operation Actions */}
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start text-sm font-normal hover:bg-accent hover:text-accent-foreground"
                onClick={onShowOutput}
                disabled={disabled}
              >
                <ListCollapse className="mr-2 h-4 w-4" />
                Show Outputs
              </Button>

              {/* LLM-specific Actions */}
              {llmType === "LLM" && (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-start text-sm font-normal hover:bg-accent hover:text-accent-foreground"
                    onClick={onToggleGuardrails}
                  >
                    <Shield className="mr-2 h-4 w-4" />
                    {isGuardrailsExpanded
                      ? "Hide Guardrails"
                      : "Show Guardrails"}
                  </Button>

                  {(type === "map" ||
                    type === "reduce" ||
                    type === "filter") && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="w-full justify-start text-sm font-normal hover:bg-accent hover:text-accent-foreground"
                      onClick={onToggleGleanings}
                    >
                      <Shield className="mr-2 h-4 w-4" />
                      {isGleaningsExpanded ? "Hide Gleaning" : "Show Gleaning"}
                    </Button>
                  )}

                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-start text-sm font-normal hover:bg-accent hover:text-accent-foreground"
                    onClick={onImprovePrompt}
                  >
                    <Wand2 className="mr-2 h-4 w-4" />
                    Improve Prompt
                  </Button>
                </>
              )}

              {/* Operation-specific Actions */}

              {canBeOptimized(type) && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start text-sm font-normal hover:bg-accent hover:text-accent-foreground"
                  onClick={onOptimize}
                  disabled={disabled}
                >
                  <Zap className="mr-2 h-4 w-4" />
                  Decompose Operation
                </Button>
              )}

              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start text-sm font-normal hover:bg-accent hover:text-accent-foreground"
                onClick={onToggleSettings}
              >
                <Settings className="mr-2 h-4 w-4" />
                Edit Other Args
              </Button>

              <div className="h-px bg-gray-100 my-1" />

              {/* Visibility Toggle */}
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start text-sm font-normal hover:bg-accent hover:text-accent-foreground"
                onClick={onToggleVisibility}
              >
                {visibility ? (
                  <>
                    <EyeOff className="mr-2 h-4 w-4" />
                    Skip Operation
                  </>
                ) : (
                  <>
                    <Eye className="mr-2 h-4 w-4" />
                    Include Operation
                  </>
                )}
              </Button>

              {/* Delete Operation */}
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start text-sm font-normal text-destructive hover:bg-destructive hover:text-destructive-foreground"
                onClick={onDelete}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Delete
              </Button>
            </div>
          </PopoverContent>
        </Popover>

        {/* Expand/Collapse Button */}
        <Button
          variant="ghost"
          size="sm"
          className="ml-2 h-8 w-8 p-0 hover:bg-gray-100 rounded-full"
          onClick={onToggleExpand}
        >
          <ChevronDown
            className={`h-4 w-4 text-gray-600 transform transition-transform ${
              expanded ? "rotate-180" : ""
            }`}
          />
        </Button>
      </div>
    );
  }
);
OperationHeader.displayName = "OperationHeader";

interface SettingsModalProps {
  opName: string;
  opType: string;
  isOpen: boolean;
  onClose: () => void;
  otherKwargs: Record<string, string>;
  onSettingsSave: (newSettings: Record<string, string>) => void;
}

const SettingsModal: React.FC<SettingsModalProps> = React.memo(
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
SettingsModal.displayName = "SettingsModal";

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

// Add id to the props interface
interface Props {
  index: number;
  id?: string;
}

// Main component
export const OperationCard: React.FC<Props> = ({ index, id }) => {
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
    namespace,
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
          namespace,
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
          namespace,
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
          namespace,
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
    <Draggable draggableId={operation.id} index={index} key={operation.id}>
      {(provided, snapshot) => (
        <div
          ref={provided.innerRef}
          {...provided.draggableProps}
          id={id}
          className={`mb-2 relative rounded-sm shadow-sm w-full ${
            pipelineOutput?.operationId === operation.id
              ? "bg-white border-primary border-2"
              : "bg-white"
          } ${!operation.visibility ? "opacity-50" : ""}`}
        >
          <div
            {...provided.dragHandleProps}
            className="absolute left-0 top-0 bottom-0 w-6 flex items-center justify-center cursor-move hover:bg-gray-100 border-r border-gray-100"
          >
            <GripVertical size={14} className="text-gray-400" />
          </div>

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
              isGuardrailsExpanded={isGuardrailsExpanded}
              isGleaningsExpanded={isGleaningsExpanded}
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
              onToggleExpand={() => dispatch({ type: "TOGGLE_EXPAND" })}
              onToggleVisibility={handleVisibilityToggle}
              onImprovePrompt={() => setShowPromptImprovement(true)}
              onToggleGuardrails={() => dispatch({ type: "TOGGLE_GUARDRAILS" })}
              onToggleGleanings={() => dispatch({ type: "TOGGLE_GLEANINGS" })}
            />
            {isExpanded && operation.visibility !== false && (
              <>
                <CardContent className="p-4">
                  {createOperationComponent(
                    operation,
                    handleOperationUpdate,
                    isSchemaExpanded,
                    () => dispatch({ type: "TOGGLE_SCHEMA" })
                  )}
                </CardContent>

                {operation.llmType === "LLM" && isGuardrailsExpanded && (
                  <div className="px-4 pb-4">
                    <Guardrails
                      guardrails={operation.validate || []}
                      onUpdate={handleGuardrailsUpdate}
                      isExpanded={true}
                      onToggle={() => dispatch({ type: "TOGGLE_GUARDRAILS" })}
                    />
                  </div>
                )}

                {(operation.type === "map" ||
                  operation.type === "reduce" ||
                  operation.type === "filter") &&
                  isGleaningsExpanded && (
                    <div className="px-4 pb-4">
                      <GleaningConfig
                        gleaning={operation.gleaning || null}
                        onUpdate={handleGleaningsUpdate}
                        isExpanded={true}
                        onToggle={() => dispatch({ type: "TOGGLE_GLEANINGS" })}
                      />
                    </div>
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
        </div>
      )}
    </Draggable>
  );
};

const SkeletonCard: React.FC = () => (
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
);
