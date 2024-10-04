import React, { useEffect } from 'react';
import { FileText, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { File } from '@/app/types';

interface FileExplorerProps {
  files: File[];
  onFileClick: (file: File) => void;
  onFileUpload: (file: File) => void;
  onFilesUpdate: (updatedFiles: File[]) => void;
}

export const FileExplorer: React.FC<FileExplorerProps> = ({
  files,
  onFileClick,
  onFileUpload,
  onFilesUpdate,
}) => {
  useEffect(() => {
    fetch('/debate_transcripts.json')
      .then(response => response.json())
      .then(data => {
        const updatedFiles = files.map(file => 
          file.name === 'debate_transcripts.json' 
            ? { ...file, content: JSON.stringify(data, null, 2) } 
            : file
        );
        onFilesUpdate(updatedFiles);
      });
  }, []);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile && uploadedFile.type === 'application/json') {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        onFileUpload({ name: uploadedFile.name, content });
      };
      reader.readAsText(uploadedFile);
    } else {
      alert('Please upload a JSON file');
    }
  };

  return (
    <div className="h-full p-4 bg-white">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-sm font-bold mb-2 flex items-center uppercase whitespace-nowrap overflow-x-auto">
          File Explorer
        </h2>
        <Button variant="ghost" size="icon" className="rounded-sm flex-shrink-0">
          <Input 
            type="file" 
            accept=".json" 
            onChange={handleFileUpload} 
            className="hidden" 
            id="file-upload" 
          />
          <label htmlFor="file-upload" className="cursor-pointer">
            <Upload size={16} />
          </label>
        </Button>
      </div>
      <div className="overflow-x-auto">
        {files.map((file) => (
          <div key={file.name} className="cursor-pointer hover:bg-gray-100 p-1 whitespace-nowrap" onClick={() => onFileClick(file)}>
            <FileText className="inline mr-2" size={16} />
            {file.name}
          </div>
        ))}
      </div>
    </div>
  );
};