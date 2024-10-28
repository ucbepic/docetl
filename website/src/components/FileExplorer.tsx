import React from "react";
import { FileText, Upload, Trash2, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { File } from "@/app/types";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";

interface FileExplorerProps {
  files: File[];
  onFileClick: (file: File) => void;
  onFileUpload: (file: File) => void;
  onFileDelete: (file: File) => void;
  currentFile: File | null;
  setCurrentFile: (file: File | null) => void;
  setShowDatasetView: (show: boolean) => void;
}

export const FileExplorer: React.FC<FileExplorerProps> = ({
  files,
  onFileClick,
  onFileUpload,
  onFileDelete,
  currentFile,
  setCurrentFile,
  setShowDatasetView,
}) => {
  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile && uploadedFile.type === "application/json") {
      const formData = new FormData();
      formData.append("file", uploadedFile);

      try {
        const response = await fetch("/api/uploadFile", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Upload failed");
        }

        const data = await response.json();
        const filePath = data.path; // This will be the absolute path from the server

        onFileUpload({ name: uploadedFile.name, path: filePath });
        setCurrentFile({ name: uploadedFile.name, path: filePath });
      } catch (error) {
        alert("Failed to upload file");
        console.error(error);
      }
    } else {
      alert("Please upload a JSON file");
    }
  };

  const handleFileSelection = (file: File) => {
    setCurrentFile(file);
    onFileClick(file);
  };

  return (
    <div className="h-full p-4">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-sm font-bold mb-2 flex items-center uppercase whitespace-nowrap overflow-x-auto">
          File Explorer
        </h2>
        <Button
          variant="ghost"
          size="icon"
          className="rounded-sm flex-shrink-0"
        >
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
          <ContextMenu key={file.name}>
            <ContextMenuTrigger
              className={`flex w-full cursor-pointer hover:bg-gray-100 p-1 whitespace-nowrap ${
                currentFile?.name === file.name ? "bg-blue-100" : ""
              }`}
              onClick={() => handleFileSelection(file)}
            >
              <FileText className="inline mr-2" size={16} />
              {file.name}
            </ContextMenuTrigger>
            <ContextMenuContent className="w-64">
              <ContextMenuItem
                onClick={() => {
                  handleFileSelection(file);
                  setShowDatasetView(true);
                }}
              >
                <Eye className="mr-2 h-4 w-4" />
                <span>View File</span>
              </ContextMenuItem>
              <ContextMenuItem onClick={() => onFileDelete(file)}>
                <Trash2 className="mr-2 h-4 w-4" />
                <span>Delete</span>
              </ContextMenuItem>
            </ContextMenuContent>
          </ContextMenu>
        ))}
      </div>
    </div>
  );
};
