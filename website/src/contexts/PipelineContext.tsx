import React, { useState, useCallback, createContext, useContext, useEffect, useRef } from 'react';
import { Operation, File, OutputType } from '@/app/types';
import { mockFiles, initialOperations, mockSampleSize, mockPipelineName } from '@/mocks/mockData';
import * as localStorageKeys from '@/app/localStorageKeys';

interface PipelineState {
  operations: Operation[];
  currentFile: File | null;
  output: OutputType | null;
  terminalOutput: string;
  isLoadingOutputs: boolean;
  numOpRun: number;
  pipelineName: string;
  sampleSize: number | null;
  files: File[];
  cost: number;
  defaultModel: string;
}

interface PipelineContextType extends PipelineState {
  setOperations: React.Dispatch<React.SetStateAction<Operation[]>>;
  setCurrentFile: React.Dispatch<React.SetStateAction<File | null>>;
  setOutput: React.Dispatch<React.SetStateAction<OutputType | null>>;
  setTerminalOutput: React.Dispatch<React.SetStateAction<string>>;
  setIsLoadingOutputs: React.Dispatch<React.SetStateAction<boolean>>;
  setNumOpRun: React.Dispatch<React.SetStateAction<number>>;
  setPipelineName: React.Dispatch<React.SetStateAction<string>>;
  setSampleSize: React.Dispatch<React.SetStateAction<number | null>>;
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
  setCost: React.Dispatch<React.SetStateAction<number>>;
  setDefaultModel: React.Dispatch<React.SetStateAction<string>>;
  saveProgress: () => void;
  unsavedChanges: boolean;
  clearPipelineState: () => void;
}

const PipelineContext = createContext<PipelineContextType | undefined>(undefined);

const loadFromLocalStorage = <T,>(key: string, defaultValue: T): T => {
  if (typeof window !== "undefined") {
    const storedValue = localStorage.getItem(key);
    return storedValue ? JSON.parse(storedValue) : defaultValue;
  }
  return defaultValue;
};

export const PipelineProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<PipelineState>(() => ({
    operations: loadFromLocalStorage(localStorageKeys.OPERATIONS_KEY, initialOperations),
    currentFile: loadFromLocalStorage(localStorageKeys.CURRENT_FILE_KEY, mockFiles[0]),
    output: loadFromLocalStorage(localStorageKeys.OUTPUT_KEY, null),
    terminalOutput: loadFromLocalStorage(localStorageKeys.TERMINAL_OUTPUT_KEY, ''),
    isLoadingOutputs: loadFromLocalStorage(localStorageKeys.IS_LOADING_OUTPUTS_KEY, false),
    numOpRun: loadFromLocalStorage(localStorageKeys.NUM_OP_RUN_KEY, 0),
    pipelineName: loadFromLocalStorage(localStorageKeys.PIPELINE_NAME_KEY, mockPipelineName),
    sampleSize: loadFromLocalStorage(localStorageKeys.SAMPLE_SIZE_KEY, mockSampleSize),
    files: loadFromLocalStorage(localStorageKeys.FILES_KEY, mockFiles),
    cost: loadFromLocalStorage(localStorageKeys.COST_KEY, 0),
    defaultModel: loadFromLocalStorage(localStorageKeys.DEFAULT_MODEL_KEY, "gpt-4o-mini"),
  }));

  const [unsavedChanges, setUnsavedChanges] = useState(false);
  const stateRef = useRef(state);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const saveProgress = useCallback(() => {
    Object.entries(stateRef.current).forEach(([key, value]) => {
      localStorage.setItem(localStorageKeys[`${key.toUpperCase()}_KEY` as keyof typeof localStorageKeys], JSON.stringify(value));
    });
    setUnsavedChanges(false);
    console.log('Progress saved!');
  }, []);

  const setStateAndUpdate = useCallback(<K extends keyof PipelineState>(
    key: K,
    value: PipelineState[K] | ((prevState: PipelineState[K]) => PipelineState[K])
  ) => {
    setState(prevState => {
      const newValue = typeof value === 'function' ? (value as Function)(prevState[key]) : value;
      if (newValue !== prevState[key]) {
        setUnsavedChanges(true);
        return { ...prevState, [key]: newValue };
      }
      return prevState;
    });
  }, []);

  const clearPipelineState = useCallback(() => {
    Object.values(localStorageKeys).forEach(key => {
      localStorage.removeItem(key);
    });
    setState({
      operations: initialOperations,
      currentFile: mockFiles[0],
      output: null,
      terminalOutput: '',
      isLoadingOutputs: false,
      numOpRun: 0,
      pipelineName: mockPipelineName,
      sampleSize: mockSampleSize,
      files: mockFiles,
      cost: 0,
      defaultModel: "gpt-4o-mini",
    });
    setUnsavedChanges(false);
    console.log('Pipeline state cleared!');
  }, []);

  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (unsavedChanges) {
        event.preventDefault();
        event.returnValue = '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [unsavedChanges]);

  const contextValue: PipelineContextType = {
    ...state,
    setOperations: useCallback((value) => setStateAndUpdate('operations', value), [setStateAndUpdate]),
    setCurrentFile: useCallback((value) => setStateAndUpdate('currentFile', value), [setStateAndUpdate]),
    setOutput: useCallback((value) => setStateAndUpdate('output', value), [setStateAndUpdate]),
    setTerminalOutput: useCallback((value) => setStateAndUpdate('terminalOutput', value), [setStateAndUpdate]),
    setIsLoadingOutputs: useCallback((value) => setStateAndUpdate('isLoadingOutputs', value), [setStateAndUpdate]),
    setNumOpRun: useCallback((value) => setStateAndUpdate('numOpRun', value), [setStateAndUpdate]),
    setPipelineName: useCallback((value) => setStateAndUpdate('pipelineName', value), [setStateAndUpdate]),
    setSampleSize: useCallback((value) => setStateAndUpdate('sampleSize', value), [setStateAndUpdate]),
    setFiles: useCallback((value) => setStateAndUpdate('files', value), [setStateAndUpdate]),
    setCost: useCallback((value) => setStateAndUpdate('cost', value), [setStateAndUpdate]),
    setDefaultModel: useCallback((value) => setStateAndUpdate('defaultModel', value), [setStateAndUpdate]),
    saveProgress,
    unsavedChanges,
    clearPipelineState,
  };

  return (
    <PipelineContext.Provider value={contextValue}>
      {children}
    </PipelineContext.Provider>
  );
};

export const usePipelineContext = () => {
  const context = useContext(PipelineContext);
  if (context === undefined) {
    throw new Error('usePipelineContext must be used within a PipelineProvider');
  }
  return context;
};