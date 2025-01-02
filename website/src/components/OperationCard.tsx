import React, {
  useReducer,
  useCallback,
  useEffect,
  useRef,
  useState,
  useMemo,
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
import {
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
  Pencil,
  MoveUp,
  MoveDown,
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
import { canBeOptimized } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { PromptImprovementDialog } from "@/components/PromptImprovementDialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { OperationHelpButton } from "./OperationHelpButton";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";

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
  onMoveUp: () => void;
  onMoveDown: () => void;
  isFirst: boolean;
  isLast: boolean;
  model?: string;
  onModelChange?: (newModel: string) => void;
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
    onMoveUp,
    onMoveDown,
    isFirst,
    isLast,
    model,
    onModelChange,
  }) => {
    const [menuOpen, setMenuOpen] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [editedName, setEditedName] = useState(name);
    const [isEditingModel, setIsEditingModel] = useState(false);
    const [editedModel, setEditedModel] = useState(model);

    return (
      <div className="relative flex items-center py-3 px-4 border-b border-border/30 bg-muted/5">
        {/* Left side - Operation info */}
        <div className="flex-1 flex items-center gap-2">
          <div className="flex items-center gap-2">
            <Badge variant={currOp ? "default" : "secondary"}>{type}</Badge>

            {/* Add help button for LLM operations */}
            {llmType === "LLM" &&
              (type === "map" || type === "reduce" || type === "filter") && (
                <OperationHelpButton type={type} />
              )}

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

            {llmType === "LLM" && (
              <div className="flex items-center">
                {isEditingModel ? (
                  <Input
                    value={editedModel}
                    onChange={(e) => setEditedModel(e.target.value)}
                    onBlur={() => {
                      setIsEditingModel(false);
                      onModelChange?.(editedModel || "");
                    }}
                    onKeyPress={(e) => {
                      if (e.key === "Enter") {
                        setIsEditingModel(false);
                        onModelChange?.(editedModel || "");
                      }
                    }}
                    className="max-w-[150px] h-6 text-xs font-mono"
                    autoFocus
                  />
                ) : (
                  <div
                    className="flex items-center gap-1 group cursor-pointer"
                    onClick={() => setIsEditingModel(true)}
                  >
                    <span className="text-xs font-mono text-muted-foreground">
                      {model}
                    </span>
                    <Pencil
                      size={11}
                      className="opacity-0 group-hover:opacity-70 transition-opacity text-muted-foreground"
                    />
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="flex items-center">
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
                className="max-w-[200px] h-6 text-sm font-medium"
                autoFocus
              />
            ) : (
              <div
                className="flex items-center gap-1 group cursor-pointer"
                onClick={() => setIsEditing(true)}
              >
                <span
                  className={`text-sm font-medium select-none ${
                    llmType === "LLM"
                      ? "bg-gradient-to-r from-blue-600 to-purple-600 text-transparent bg-clip-text font-semibold"
                      : ""
                  }`}
                >
                  {name}
                </span>
                <Pencil
                  size={13}
                  className="opacity-0 group-hover:opacity-70 transition-opacity text-muted-foreground"
                />
              </div>
            )}
          </div>
        </div>

        {/* Action Bar - Keep only the most essential actions */}
        <div className="flex items-center gap-2 mr-2">
          {/* Show Outputs Button */}
          <Button
            variant="outline"
            size="sm"
            className="flex items-center gap-1"
            onClick={onShowOutput}
            disabled={disabled}
          >
            <ListCollapse className="h-4 w-4" />
            <span className="hidden sm:inline">Show Outputs</span>
          </Button>

          {/* LLM-specific Actions */}
          {llmType === "LLM" && (
            <Button
              variant="outline"
              size="sm"
              className="flex items-center gap-1"
              onClick={onImprovePrompt}
            >
              <Wand2 className="h-4 w-4" />
              <span className="hidden sm:inline">Improve Prompt</span>
            </Button>
          )}

          {/* More Options Menu */}
          <Popover open={menuOpen} onOpenChange={setMenuOpen}>
            <PopoverTrigger asChild>
              <Button variant="outline" size="sm" className="h-8 w-8 p-0">
                <Menu className="h-4 w-4" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-56 p-1" align="end">
              <div className="space-y-0.5">
                {/* Move operation actions */}
                {!isFirst && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-start"
                    onClick={onMoveUp}
                  >
                    <MoveUp className="mr-2 h-4 w-4" />
                    Move Up
                  </Button>
                )}
                {!isLast && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-start"
                    onClick={onMoveDown}
                  >
                    <MoveDown className="mr-2 h-4 w-4" />
                    Move Down
                  </Button>
                )}
                {(!isFirst || !isLast) && (
                  <div className="h-px bg-gray-100 my-1" />
                )}

                {/* LLM-specific menu items */}
                {llmType === "LLM" && (
                  <>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="w-full justify-start"
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
                        className="w-full justify-start"
                        onClick={onToggleGleanings}
                      >
                        <Shield className="mr-2 h-4 w-4" />
                        {isGleaningsExpanded
                          ? "Hide Gleaning"
                          : "Show Gleaning"}
                      </Button>
                    )}
                    <div className="h-px bg-gray-100 my-1" />
                  </>
                )}

                {/* Optimization in menu for supported types */}
                {canBeOptimized(type) && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-start"
                    onClick={onOptimize}
                    disabled={disabled}
                  >
                    <Zap className="mr-2 h-4 w-4" />
                    Optimize Operation
                  </Button>
                )}

                {/* Settings */}
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start"
                  onClick={onToggleSettings}
                >
                  <Settings className="mr-2 h-4 w-4" />
                  Other Arguments
                </Button>

                {/* Visibility Toggle */}
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start"
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

                <div className="h-px bg-gray-100 my-1" />

                {/* Delete Operation */}
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start text-destructive hover:bg-destructive hover:text-destructive-foreground"
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
            className="h-8 w-8 p-0 hover:bg-gray-100 rounded-full"
            onClick={onToggleExpand}
          >
            <ChevronDown
              className={`h-4 w-4 text-gray-600 transform transition-transform ${
                expanded ? "rotate-180" : ""
              }`}
            />
          </Button>
        </div>
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
    apiKeys,
    systemPrompt,
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

  const hasOpenAIKey = useMemo(() => {
    return apiKeys.some((key) => key.name === "OPENAI_API_KEY");
  }, [apiKeys]);

  const [showOptimizeDialog, setShowOptimizeDialog] = useState(false);
  const [isLocalMode, setIsLocalMode] = useState(false);

  const onOptimize = useCallback(async () => {
    if (!operation) return;
    setShowOptimizeDialog(true);
  }, [operation]);

  const handleOptimizeConfirm = useCallback(async () => {
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
          clear_intermediate: false,
          system_prompt: systemPrompt,
          namespace: namespace,
          apiKeys: apiKeys,
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
    } finally {
      setShowOptimizeDialog(false);
    }
  }, [
    operation,
    defaultModel,
    currentFile,
    operations,
    pipelineName,
    sampleSize,
    optimizerModel,
    connect,
    sendMessage,
    systemPrompt,
    namespace,
    apiKeys,
  ]);

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

  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const handleMoveUp = useCallback(() => {
    if (index > 0) {
      setOperations((prevOperations) => {
        const newOperations = [...prevOperations];
        [newOperations[index - 1], newOperations[index]] = [
          newOperations[index],
          newOperations[index - 1],
        ];
        return newOperations;
      });
    }
  }, [index, setOperations]);

  const handleMoveDown = useCallback(() => {
    if (index < operations.length - 1) {
      setOperations((prevOperations) => {
        const newOperations = [...prevOperations];
        [newOperations[index], newOperations[index + 1]] = [
          newOperations[index + 1],
          newOperations[index],
        ];
        return newOperations;
      });
    }
  }, [index, operations.length, setOperations]);

  const handleModelChange = useCallback(
    (newModel: string) => {
      if (!operation) return;
      const updatedOperation = {
        ...operation,
        otherKwargs: {
          ...operation.otherKwargs,
          model: newModel,
        },
      };
      handleOperationUpdate(updatedOperation);
    },
    [operation, handleOperationUpdate]
  );

  if (!operation) {
    return <SkeletonCard />;
  }

  return (
    <div
      id={id}
      className={`mb-2 relative rounded-md border shadow-[0_1px_3px_0_rgb(0,0,0,0.05)] w-full pl-6 hover:shadow-md transition-shadow ${
        pipelineOutput?.operationId === operation.id
          ? "bg-white border-primary border-2"
          : "bg-white border-border/40"
      } ${!operation.visibility ? "opacity-50" : ""}`}
    >
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
        onDelete={() => setShowDeleteDialog(true)}
        onRunOperation={handleRunOperation}
        onToggleSettings={() => dispatch({ type: "TOGGLE_SETTINGS" })}
        onShowOutput={onShowOutput}
        onOptimize={onOptimize}
        onToggleExpand={() => dispatch({ type: "TOGGLE_EXPAND" })}
        onToggleVisibility={handleVisibilityToggle}
        onImprovePrompt={() => setShowPromptImprovement(true)}
        onToggleGuardrails={() => dispatch({ type: "TOGGLE_GUARDRAILS" })}
        onToggleGleanings={() => dispatch({ type: "TOGGLE_GLEANINGS" })}
        onMoveUp={handleMoveUp}
        onMoveDown={handleMoveDown}
        isFirst={index === 0}
        isLast={index === operations.length - 1}
        model={operation.otherKwargs?.model || defaultModel}
        onModelChange={handleModelChange}
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
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This action cannot be undone. This will permanently delete the
              operation &quot;{operation.name}&quot; and remove it from the
              pipeline. If you only want to hide the operation from the next
              run, you can toggle the visibility of the operation in the
              operation menu.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => {
                setOperations((prev) =>
                  prev.filter((op) => op.id !== operation.id)
                );
                setShowDeleteDialog(false);
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <AlertDialog
        open={showOptimizeDialog}
        onOpenChange={setShowOptimizeDialog}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Optimize Operation</AlertDialogTitle>
            <AlertDialogDescription>
              {!hasOpenAIKey && !isLocalMode ? (
                <div className="space-y-2">
                  <p className="text-destructive font-medium">
                    OpenAI API Key Required
                  </p>
                  <p>
                    To use the optimizer, please add your OpenAI API key in Edit{" "}
                    {">"}
                    Edit API Keys.
                  </p>
                  <button
                    className="text-destructive underline hover:opacity-80 font-medium"
                    onClick={() => setIsLocalMode(true)}
                  >
                    Ignore if running locally with environment variables
                  </button>
                </div>
              ) : (
                <p>
                  This will analyze the operation and replace it with another
                  pipeline that has higher accuracy (as determined by an
                  LLM-as-a-judge), if it can be found. Do you want to proceed?
                  The process may take between 2 and 10 minutes, depending on
                  how complex your data is.
                </p>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleOptimizeConfirm}
              disabled={!hasOpenAIKey && !isLocalMode}
            >
              Proceed
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

const SkeletonCard: React.FC = () => (
  <Card className="mb-2 relative rounded-md border border-border/40 shadow-[0_1px_3px_0_rgb(0,0,0,0.05)] w-full hover:shadow-md transition-shadow">
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
