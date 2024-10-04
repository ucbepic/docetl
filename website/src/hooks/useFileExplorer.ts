import { useState, useEffect } from 'react';
import { File } from '@/app/types';
import { mockFiles } from '@/mocks/mockData';

export const useFileExplorer = () => {
  const [files, setFiles] = useState<File[]>(mockFiles);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileClick = (file: File) => {
    setSelectedFile(file);
  };

  const handleFileUpload = (file: File) => {
    setFiles(prevFiles => [...prevFiles, file]);
  };

  const handleFilesUpdate = (updatedFiles: File[]) => {
    setFiles(updatedFiles);
  };

  useEffect(() => {
    // Fetch debate transcripts
    fetch('/debate_transcripts.json')
      .then(response => response.json())
      .then(data => {
        const updatedFiles = files.map(file => 
          file.name === 'debate_transcripts.json' 
            ? { ...file, content: JSON.stringify(data, null, 2) } 
            : file
        );
        handleFilesUpdate(updatedFiles);
      });
  }, []);

  return {
    files,
    selectedFile,
    handleFileClick,
    handleFileUpload,
    handleFilesUpdate,
  };
};