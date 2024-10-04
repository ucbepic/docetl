import React, { createContext, useContext, useState } from 'react';
import { Operation, File, OutputRow } from '@/app/types';
import { initialOperations } from '@/mocks/mockData';

interface PipelineContextType {
  operations: Operation[];
  currentFile: File | null;
  setOperations: React.Dispatch<React.SetStateAction<Operation[]>>;
  setCurrentFile: React.Dispatch<React.SetStateAction<File | null>>;
  outputs: OutputRow[];
  setOutputs: React.Dispatch<React.SetStateAction<OutputRow[]>>;
  isLoadingOutputs: boolean;
  setIsLoadingOutputs: React.Dispatch<React.SetStateAction<boolean>>;
}

const PipelineContext = createContext<PipelineContextType | undefined>(undefined);

export const PipelineProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [operations, setOperations] = useState<Operation[]>(initialOperations);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [outputs, setOutputs] = useState<OutputRow[]>([]);
  const [isLoadingOutputs, setIsLoadingOutputs] = useState<boolean>(false);
  return (
    <PipelineContext.Provider value={{
      operations,
      currentFile,
      setOperations,
      setCurrentFile,
      outputs,
      setOutputs,
      isLoadingOutputs,
      setIsLoadingOutputs,
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