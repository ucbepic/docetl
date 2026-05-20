import React, {
  useState,
  useCallback,
  createContext,
  useContext,
  useEffect,
  useRef,
} from "react";
import { Operation, File, OutputType, Bookmark, APIKey } from "@/app/types";
import {
  mockFiles,
  initialOperations,
  mockSampleSize,
  mockPipelineName,
} from "@/mocks/mockData";
import { toast } from "@/hooks/use-toast";
import yaml from "js-yaml";

interface PipelineState {
  operations: Operation[];
  currentFile: File | null;
  output: OutputType | null;
  terminalOutput: string;
  optimizerProgress: {
    status: string;
    progress: number;
    shouldOptimize: boolean;
    rationale: string;
    validatorPrompt: string;
  } | null;
  isLoadingOutputs: boolean;
  isDecomposing: boolean;
  numOpRun: number;
  pipelineName: string;
  sampleSize: number | null;
  files: File[];
  cost: number;
  defaultModel: string;
  optimizerModel: string;
  autoOptimizeCheck: boolean;
  highLevelGoal: string;
  systemPrompt: { datasetDescription: string | null; persona: string | null };
  namespace: string | null;
  apiKeys: APIKey[];
  extraPipelineSettings: Record<string, unknown> | null;
}

interface PipelineContextType extends PipelineState {
  setOperations: React.Dispatch<React.SetStateAction<Operation[]>>;
  setCurrentFile: React.Dispatch<React.SetStateAction<File | null>>;
  setOutput: React.Dispatch<React.SetStateAction<OutputType | null>>;
  setTerminalOutput: React.Dispatch<React.SetStateAction<string>>;
  setOptimizerProgress: React.Dispatch<
    React.SetStateAction<{
      status: string;
      progress: number;
      shouldOptimize: boolean;
      rationale: string;
      validatorPrompt: string;
    } | null>
  >;
  setIsLoadingOutputs: React.Dispatch<React.SetStateAction<boolean>>;
  setIsDecomposing: React.Dispatch<React.SetStateAction<boolean>>;
  setNumOpRun: React.Dispatch<React.SetStateAction<number>>;
  setPipelineName: React.Dispatch<React.SetStateAction<string>>;
  setSampleSize: React.Dispatch<React.SetStateAction<number | null>>;
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
  setCost: React.Dispatch<React.SetStateAction<number>>;
  setDefaultModel: React.Dispatch<React.SetStateAction<string>>;
  setOptimizerModel: React.Dispatch<React.SetStateAction<string>>;
  saveProgress: () => Promise<void>;
  unsavedChanges: boolean;
  clearPipelineState: () => void;
  serializeState: () => Promise<string>;
  setAutoOptimizeCheck: React.Dispatch<React.SetStateAction<boolean>>;
  setHighLevelGoal: React.Dispatch<React.SetStateAction<string>>;
  setSystemPrompt: React.Dispatch<
    React.SetStateAction<{
      datasetDescription: string | null;
      persona: string | null;
    }>
  >;
  setNamespace: React.Dispatch<React.SetStateAction<string | null>>;
  setApiKeys: React.Dispatch<React.SetStateAction<APIKey[]>>;
  setExtraPipelineSettings: React.Dispatch<
    React.SetStateAction<Record<string, unknown> | null>
  >;
  // Ref for triggering decomposition from OperationCard (using ref to avoid infinite loops)
  onRequestDecompositionRef: React.MutableRefObject<
    ((operationId: string, operationName: string) => void) | null
  >;
}

const PipelineContext = createContext<PipelineContextType | undefined>(
  undefined
);

const defaultState = (namespace: string | null): PipelineState => ({
  operations: initialOperations,
  currentFile: null,
  output: null,
  terminalOutput: "",
  optimizerProgress: null,
  isLoadingOutputs: false,
  isDecomposing: false,
  numOpRun: 0,
  pipelineName: mockPipelineName,
  sampleSize: mockSampleSize,
  files: mockFiles,
  cost: 0,
  defaultModel: "vertex_ai/gemini-2.0-flash",
  optimizerModel: "vertex_ai/gemini-2.0-flash",
  autoOptimizeCheck: false,
  highLevelGoal: "",
  systemPrompt: { datasetDescription: null, persona: null },
  namespace,
  apiKeys: [],
  extraPipelineSettings: null,
});

const PERSISTED_KEYS: (keyof PipelineState)[] = [
  "operations",
  "currentFile",
  "output",
  "terminalOutput",
  "isLoadingOutputs",
  "numOpRun",
  "pipelineName",
  "sampleSize",
  "files",
  "cost",
  "defaultModel",
  "optimizerModel",
  "autoOptimizeCheck",
  "highLevelGoal",
  "systemPrompt",
  "extraPipelineSettings",
];

function stateToYaml(state: PipelineState): string {
  const obj: Record<string, unknown> = {};
  for (const key of PERSISTED_KEYS) {
    obj[key] = state[key];
  }
  return yaml.dump(obj, { skipInvalid: true });
}

function yamlToPartialState(content: string): Partial<PipelineState> {
  const obj = yaml.load(content) as Record<string, unknown>;
  const partial: Partial<PipelineState> = {};
  for (const key of PERSISTED_KEYS) {
    if (key in obj) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (partial as any)[key] = obj[key];
    }
  }
  return partial;
}

const serializeState = async (state: PipelineState): Promise<string> => {
  // Get important output samples
  let outputSample = "";
  let currentOperationName = "";
  let schemaInfo = "";

  if (state.output?.path) {
    try {
      const outputResponse = await fetch(
        `/api/readFile?path=${state.output.path}`
      );
      if (!outputResponse.ok) {
        throw new Error("Failed to fetch output file");
      }

      const outputContent = await outputResponse.text();
      const outputs = JSON.parse(outputContent) || [];

      if (outputs.length > 0) {
        const operation = state.operations.find(
          (op) => op.id === state.output?.operationId
        );
        currentOperationName = operation?.name || "";
        const importantColumns =
          operation?.output?.schema?.map((item) => item.key) || [];

        if (outputs.length > 0) {
          const firstRow = outputs[0];
          schemaInfo = Object.entries(firstRow)
            .map(([key, value]) => {
              const type = typeof value;
              return `- ${key}: ${type}${
                importantColumns.includes(key)
                  ? " (output of current operation)"
                  : ""
              }`;
            })
            .join("\n");
        }

        const samples = outputs
          .slice(0, 10)
          .map((row: Record<string, unknown>) => {
            const sampleRow: Record<string, unknown> = {};

            const safeStringify = (value: unknown): string => {
              if (value === null) return "null";
              if (value === undefined) return "undefined";
              if (typeof value === "object") {
                try {
                  return JSON.stringify(value);
                } catch {
                  return "[Complex Object]";
                }
              }
              return String(value);
            };

            importantColumns.forEach((col) => {
              if (col in row) {
                const value = safeStringify(row[col]);
                if (value.length > 10000) {
                  sampleRow[`**${col}**`] =
                    `**${value.slice(0, 10000)}` +
                    `** ... (${value.length - 10000} more characters)`;
                } else {
                  sampleRow[`**${col}**`] = `**${value}**`;
                }
              }
            });

            Object.keys(row).forEach((key) => {
              if (!(key in sampleRow)) {
                const value = safeStringify(row[key]);
                if (value.length > 10000) {
                  sampleRow[key] =
                    value.slice(0, 10000) +
                    ` ... (${value.length - 10000} more characters)`;
                } else {
                  sampleRow[key] = value;
                }
              }
            });

            return sampleRow;
          });

        outputSample =
          samples.length > 0 ? JSON.stringify(samples, null, 2) : "";
      }
    } catch {
      outputSample = "\nError parsing output samples";
    }
  }

  const operationsDetails = state.operations
    .map((op) => {
      return `
- Operation: ${op.name} (${op.type})
  Type: ${op.type}
  Is LLM: ${op.llmType ? "Yes" : "No"}
  Prompt (relevant for llm operations): ${op.prompt || "No prompt"}
  Output Schema (relevant for llm operations): ${JSON.stringify(
    op.output?.schema || []
  )}
  Other arguments: ${JSON.stringify(op.otherKwargs || {}, null, 2)}`;
    })
    .join("\n");

  return `Current Pipeline State:
Pipeline Name: "${state.pipelineName}"
High-Level Goal: "${state.highLevelGoal || "unspecified"}"
Input Dataset File: ${
    state.currentFile ? `"${state.currentFile.name}"` : "None"
  }

Pipeline operations:${operationsDetails}
${
  currentOperationName && outputSample
    ? `
Operation just executed: ${currentOperationName}

Schema Information:
${schemaInfo}

Sample output for current operation (the LLM-generated outputs for this operation are bolded; other keys from other operations or the original input file are included but not bolded):
${outputSample}`
    : ""
}`;
};

export const PipelineProvider: React.FC<{
  children: React.ReactNode;
  workspaceId: string;
}> = ({ children, workspaceId }) => {
  const [state, setState] = useState<PipelineState>(() =>
    defaultState(workspaceId)
  );
  const [unsavedChanges, setUnsavedChanges] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const stateRef = useRef(state);
  const [isMounted, setIsMounted] = useState(false);

  const onRequestDecompositionRef = useRef<
    ((operationId: string, operationName: string) => void) | null
  >(null);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Load workspace from server on mount
  useEffect(() => {
    if (!workspaceId) return;
    fetch(`/api/workspace?id=${workspaceId}`)
      .then(async (res) => {
        if (res.status === 404) {
          setIsLoaded(true);
          return;
        }
        if (!res.ok) throw new Error("Failed to load workspace");
        const data = await res.json();
        if (data.exists && data.content) {
          const partial = yamlToPartialState(data.content);
          setState((prev) => ({ ...prev, ...partial, namespace: workspaceId }));
        }
        setIsLoaded(true);
      })
      .catch((err) => {
        console.error("Error loading workspace:", err);
        setIsLoaded(true);
      });
  }, [workspaceId]);

  const saveProgress = useCallback(async () => {
    const content = stateToYaml({ ...stateRef.current, namespace: workspaceId });
    try {
      const res = await fetch(`/api/workspace?id=${workspaceId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      if (!res.ok) throw new Error("Failed to save workspace");
      setUnsavedChanges(false);
    } catch (err) {
      console.error("Error saving workspace:", err);
      toast({
        title: "Error Saving",
        description: "Could not save workspace to server.",
        variant: "destructive",
      });
    }
  }, [workspaceId]);

  const clearPipelineState = useCallback(() => {
    setState(defaultState(workspaceId));
    setUnsavedChanges(false);
  }, [workspaceId]);

  const setStateAndUpdate = useCallback(
    <K extends keyof PipelineState>(
      key: K,
      value:
        | PipelineState[K]
        | ((prevState: PipelineState[K]) => PipelineState[K])
    ) => {
      setState((prevState) => {
        const newValue =
          typeof value === "function"
            ? (value as (prev: PipelineState[K]) => PipelineState[K])(
                prevState[key]
              )
            : value;
        if (newValue !== prevState[key]) {
          if (key !== "apiKeys") {
            setUnsavedChanges(true);
          }
          return { ...prevState, [key]: newValue };
        }
        return prevState;
      });
    },
    []
  );

  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (unsavedChanges) {
        event.preventDefault();
        event.returnValue = "";
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);

    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [unsavedChanges]);

  useEffect(() => {
    if (
      isMounted &&
      state.apiKeys.length === 0 &&
      window.location.href.includes("docetl.org")
    ) {
      toast({
        title: "No API Keys Found",
        description:
          "If you are accessing the playground using docetl.org, please add your API keys using Edit > Edit API Keys in the menu bar. Disregard this message if you are running DocETL locally.",
        duration: 5000,
        variant: "destructive",
      });
    }
  }, [isMounted, state.apiKeys]);

  const contextValue: PipelineContextType = {
    ...state,
    setOperations: useCallback(
      (value) => setStateAndUpdate("operations", value),
      [setStateAndUpdate]
    ),
    setCurrentFile: useCallback(
      (value) => setStateAndUpdate("currentFile", value),
      [setStateAndUpdate]
    ),
    setOutput: useCallback(
      (value) => setStateAndUpdate("output", value),
      [setStateAndUpdate]
    ),
    setTerminalOutput: useCallback(
      (value) => setStateAndUpdate("terminalOutput", value),
      [setStateAndUpdate]
    ),
    setIsLoadingOutputs: useCallback(
      (value) => setStateAndUpdate("isLoadingOutputs", value),
      [setStateAndUpdate]
    ),
    setIsDecomposing: useCallback(
      (value) => setStateAndUpdate("isDecomposing", value),
      [setStateAndUpdate]
    ),
    setNumOpRun: useCallback(
      (value) => setStateAndUpdate("numOpRun", value),
      [setStateAndUpdate]
    ),
    setPipelineName: useCallback(
      (value) => setStateAndUpdate("pipelineName", value),
      [setStateAndUpdate]
    ),
    setSampleSize: useCallback(
      (value) => setStateAndUpdate("sampleSize", value),
      [setStateAndUpdate]
    ),
    setFiles: useCallback(
      (value) => setStateAndUpdate("files", value),
      [setStateAndUpdate]
    ),
    setCost: useCallback(
      (value) => setStateAndUpdate("cost", value),
      [setStateAndUpdate]
    ),
    setDefaultModel: useCallback(
      (value) => setStateAndUpdate("defaultModel", value),
      [setStateAndUpdate]
    ),
    setOptimizerModel: useCallback(
      (value) => setStateAndUpdate("optimizerModel", value),
      [setStateAndUpdate]
    ),
    setOptimizerProgress: useCallback(
      (value) => setStateAndUpdate("optimizerProgress", value),
      [setStateAndUpdate]
    ),
    saveProgress,
    unsavedChanges,
    clearPipelineState,
    serializeState: useCallback(() => serializeState(stateRef.current), []),
    setAutoOptimizeCheck: useCallback(
      (value) => setStateAndUpdate("autoOptimizeCheck", value),
      [setStateAndUpdate]
    ),
    setHighLevelGoal: useCallback(
      (value) => setStateAndUpdate("highLevelGoal", value),
      [setStateAndUpdate]
    ),
    setSystemPrompt: useCallback(
      (value) => setStateAndUpdate("systemPrompt", value),
      [setStateAndUpdate]
    ),
    setNamespace: useCallback(
      (value) => setStateAndUpdate("namespace", value),
      [setStateAndUpdate]
    ),
    setApiKeys: useCallback(
      (value) => setStateAndUpdate("apiKeys", value),
      [setStateAndUpdate]
    ),
    setExtraPipelineSettings: useCallback(
      (value) => setStateAndUpdate("extraPipelineSettings", value),
      [setStateAndUpdate]
    ),
    onRequestDecompositionRef,
  };

  if (!isLoaded) return null;

  return (
    <PipelineContext.Provider value={contextValue}>
      {children}
    </PipelineContext.Provider>
  );
};

export const usePipelineContext = () => {
  const context = useContext(PipelineContext);
  if (context === undefined) {
    throw new Error(
      "usePipelineContext must be used within a PipelineProvider"
    );
  }
  return context;
};
