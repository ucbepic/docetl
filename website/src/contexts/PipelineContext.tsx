import React, { createContext, useContext, useState, useEffect } from 'react';
import { Operation, File, OutputType } from '@/app/types';
import { mockFiles, initialOperations, mockSampleSize, mockPipelineName } from '@/mocks/mockData';

interface PipelineContextType {
  operations: Operation[];
  currentFile: File | null;
  setOperations: React.Dispatch<React.SetStateAction<Operation[]>>;
  setCurrentFile: React.Dispatch<React.SetStateAction<File | null>>;
  output: OutputType | null;
  setOutput: React.Dispatch<React.SetStateAction<OutputType | null>>;
  isLoadingOutputs: boolean;
  setIsLoadingOutputs: React.Dispatch<React.SetStateAction<boolean>>;
  numOpRun: number;
  setNumOpRun: React.Dispatch<React.SetStateAction<number>>;
  pipelineName: string;
  setPipelineName: React.Dispatch<React.SetStateAction<string>>;
  sampleSize: number | null;
  setSampleSize: React.Dispatch<React.SetStateAction<number | null>>;
  files: File[];
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
  cost: number;
  setCost: React.Dispatch<React.SetStateAction<number>>;
  defaultModel: string;
  setDefaultModel: React.Dispatch<React.SetStateAction<string>>;
  terminalOutput: string;
  setTerminalOutput: React.Dispatch<React.SetStateAction<string>>;
}

const PipelineContext = createContext<PipelineContextType | undefined>(undefined);

const loadFromLocalStorage = <T,>(key: string, defaultValue: T): T => {
  if (typeof window !== "undefined") {
    const storedValue = localStorage.getItem(`docetl_${key}`);
    return storedValue ? JSON.parse(storedValue) : defaultValue;
  }
  return defaultValue;
};

export const PipelineProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [operations, setOperations] = useState<Operation[]>(() => loadFromLocalStorage('operations', initialOperations));
  const [currentFile, setCurrentFile] = useState<File | null>(() => loadFromLocalStorage('currentFile', mockFiles[0]));
  const [output, setOutput] = useState<OutputType | null>(() => loadFromLocalStorage('output', null));
  const [terminalOutput, setTerminalOutput] = useState<string>(() => loadFromLocalStorage('terminalOutput', ''));
  const [isLoadingOutputs, setIsLoadingOutputs] = useState<boolean>(() => loadFromLocalStorage('isLoadingOutputs', false));
  const [numOpRun, setNumOpRun] = useState<number>(() => loadFromLocalStorage('numOpRun', 0));
  const [pipelineName, setPipelineName] = useState<string>(() => loadFromLocalStorage('pipelineName', mockPipelineName));
  const [sampleSize, setSampleSize] = useState<number | null>(() => loadFromLocalStorage('sampleSize', mockSampleSize));
  const [files, setFiles] = useState<File[]>(() => loadFromLocalStorage('files', mockFiles));
  const [cost, setCost] = useState<number>(() => loadFromLocalStorage('cost', 0));
  const [defaultModel, setDefaultModel] = useState<string>(() => loadFromLocalStorage('defaultModel', "gpt-4o-mini"));

  useEffect(() => {
    localStorage.setItem('docetl_operations', JSON.stringify(operations));
  }, [operations]);

  useEffect(() => {
    localStorage.setItem('docetl_currentFile', JSON.stringify(currentFile));
  }, [currentFile]);

  useEffect(() => {
    localStorage.setItem('docetl_output', JSON.stringify(output));
  }, [output]);

  useEffect(() => {
    localStorage.setItem('docetl_terminalOutput', JSON.stringify(terminalOutput));
  }, [terminalOutput]);

  useEffect(() => {
    localStorage.setItem('docetl_isLoadingOutputs', JSON.stringify(isLoadingOutputs));
  }, [isLoadingOutputs]);

  useEffect(() => {
    localStorage.setItem('docetl_numOpRun', JSON.stringify(numOpRun));
  }, [numOpRun]);

  useEffect(() => {
    localStorage.setItem('docetl_pipelineName', JSON.stringify(pipelineName));
  }, [pipelineName]);

  useEffect(() => {
    localStorage.setItem('docetl_sampleSize', JSON.stringify(sampleSize));
  }, [sampleSize]);

  useEffect(() => {
    localStorage.setItem('docetl_files', JSON.stringify(files));
  }, [files]);

  useEffect(() => {
    localStorage.setItem('docetl_cost', JSON.stringify(cost));
  }, [cost]);

  useEffect(() => {
    localStorage.setItem('docetl_defaultModel', JSON.stringify(defaultModel));
  }, [defaultModel]);

  return (
    <PipelineContext.Provider value={{
      operations,
      currentFile,
      setOperations,
      setCurrentFile,
      output,
      setOutput,
      isLoadingOutputs,
      setIsLoadingOutputs,
      numOpRun,
      setNumOpRun,
      pipelineName,
      setPipelineName,
      sampleSize,
      setSampleSize,
      files,
      setFiles,
      cost,
      setCost,
      defaultModel,
      setDefaultModel,
      terminalOutput,
      setTerminalOutput,
    }}>
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