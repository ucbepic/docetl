import React, { useEffect, useState } from "react";
import {
  FileText,
  Upload,
  Trash2,
  Eye,
  FolderUp,
  Download,
  Loader2,
  X,
  Folder,
  Database,
  AlertTriangle,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { File } from "@/app/types";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { DocumentViewer } from "./DocumentViewer";
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
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";

interface FileExplorerProps {
  files: File[];
  onFileClick: (file: File) => void;
  onFileUpload: (file: File) => void;
  onFileDelete: (file: File) => void;
  onFolderDelete?: (folderName: string) => void;
  currentFile: File | null;
  setCurrentFile: (file: File | null) => void;
  setShowDatasetView: (show: boolean) => void;
  namespace: string;
}

function mergeFileList(
  existing: FileList | null,
  newFiles: FileList
): FileList {
  const dt = new DataTransfer();

  // Add existing files
  if (existing) {
    Array.from(existing).forEach((file) => dt.items.add(file));
  }

  // Add new files
  Array.from(newFiles).forEach((file) => dt.items.add(file));

  return dt.files;
}

interface FileWithPath extends File {
  relativePath?: string;
}

const SUPPORTED_EXTENSIONS = [
  ".pdf",
  ".docx",
  ".doc",
  ".txt",
  ".html",
  ".pptx",
  ".md",
] as const;

async function getAllFiles(entry: FileSystemEntry): Promise<FileWithPath[]> {
  const files: FileWithPath[] = [];

  async function processEntry(
    entry: FileSystemEntry,
    path: string = ""
  ): Promise<void> {
    if (entry.isFile) {
      const fileEntry = entry as FileSystemFileEntry;
      const file = await new Promise<File>((resolve, reject) => {
        // @ts-expect-error FileSystemFileEntry type definitions are incomplete
        fileEntry.file(resolve, reject);
      });

      // Check if file has supported extension
      if (
        SUPPORTED_EXTENSIONS.some((ext) =>
          file.name.toLowerCase().endsWith(ext)
        )
      ) {
        // Create a new file with the full path
        const fullPath = path ? `${path}/${file.name}` : file.name;
        // @ts-expect-error File constructor with path is not in type definitions
        const newFile = new File([file], fullPath, {
          type: file.type,
        }) as FileWithPath;
        Object.defineProperty(newFile, "relativePath", { value: fullPath });
        files.push(newFile);
      }
    } else if (entry.isDirectory) {
      const dirEntry = entry as FileSystemDirectoryEntry;
      const dirReader = dirEntry.createReader();

      const entries = await new Promise<FileSystemEntry[]>(
        (resolve, reject) => {
          dirReader.readEntries(resolve, reject);
        }
      );

      const newPath = path ? `${path}/${entry.name}` : entry.name;
      await Promise.all(entries.map((e) => processEntry(e, newPath)));
    }
  }

  await processEntry(entry);
  return files;
}

type ConversionMethod = "local" | "azure" | "docetl";

async function validateJsonDataset(file: Blob): Promise<void> {
  const text = await file.text();
  let data: unknown;

  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("Invalid JSON format");
  }

  // Check if it's an array
  if (!Array.isArray(data)) {
    throw new Error(
      "Dataset must be an array of objects, like this: [{key: value}, {key: value}]"
    );
  }

  // Check if array is not empty
  if (data.length === 0) {
    throw new Error("Dataset cannot be empty");
  }

  // Check if first item is an object
  if (typeof data[0] !== "object" || data[0] === null) {
    throw new Error("Dataset must contain objects");
  }

  // Get keys of first object
  const firstObjectKeys = Object.keys(data[0]).sort();

  // Check if all objects have the same keys
  const hasConsistentKeys = data.every((item) => {
    if (typeof item !== "object" || item === null) return false;
    const currentKeys = Object.keys(item).sort();
    return (
      currentKeys.length === firstObjectKeys.length &&
      currentKeys.every((key, index) => key === firstObjectKeys[index])
    );
  });

  if (!hasConsistentKeys) {
    throw new Error("All objects in dataset must have the same keys");
  }
}

export const FileExplorer: React.FC<FileExplorerProps> = ({
  files,
  onFileClick,
  onFileUpload,
  onFileDelete,
  onFolderDelete,
  currentFile,
  setCurrentFile,
  setShowDatasetView,
  namespace,
}) => {
  const { toast } = useToast();
  const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [draggedFiles, setDraggedFiles] = useState<number>(0);
  const [viewingDocument, setViewingDocument] = useState<File | null>(null);
  const [folderToDelete, setFolderToDelete] = useState<string | null>(null);
  const [conversionMethod, setConversionMethod] =
    useState<ConversionMethod>("local");
  const [azureEndpoint, setAzureEndpoint] = useState("");
  const [azureKey, setAzureKey] = useState("");

  const { uploadingFiles, uploadDataset } = useDatasetUpload({
    namespace,
    onFileUpload,
    setCurrentFile,
  });

  // Group files by folder
  const groupedFiles = files.reduce((acc: { [key: string]: File[] }, file) => {
    const folder = file.parentFolder || "root";
    if (!acc[folder]) {
      acc[folder] = [];
    }
    acc[folder].push(file);
    return acc;
  }, {});

  const handleFileClick = (file: File) => {
    if (file.type === "document") {
      setViewingDocument(file);
    } else {
      handleFileSelection(file);
    }
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const uploadedFile = event.target.files?.[0];
    if (!uploadedFile) {
      console.log("No file selected");
      return;
    }
    const fileToUpload: File = {
      name: uploadedFile.name,
      path: uploadedFile.name,
      type: "json",
      blob: uploadedFile,
    };
    await uploadDataset(fileToUpload);
  };

  const handleFileSelection = (file: File) => {
    setCurrentFile(file);
    onFileClick(file);
  };

  const handleFolderUpload = async (
    fileList: FileList | DataTransferItemList
  ) => {
    const files: FileWithPath[] = [];

    const processItems = async () => {
      // @ts-expect-error DataTransferItemList doesn't have proper type support
      const items = Array.from(fileList);

      for (const item of items) {
        if ("webkitGetAsEntry" in item) {
          // Handle drag and drop
          // @ts-expect-error webkitGetAsEntry is not in type definitions
          const entry = (item as DataTransferItem).webkitGetAsEntry();
          if (entry) {
            const entryFiles = await getAllFiles(entry);
            files.push(...entryFiles);
          }
        } else {
          // Handle regular file input
          // @ts-expect-error FileList type conversion needs explicit cast
          const file = item as FileWithPath;
          if (
            SUPPORTED_EXTENSIONS.some((ext) =>
              file.name.toLowerCase().endsWith(ext)
            )
          ) {
            files.push(file);
          }
        }
      }
    };

    await processItems();

    // Create a new FileList-like object with the collected files
    const dt = new DataTransfer();
    // @ts-expect-error DataTransfer.items.add type is incomplete
    files.forEach((file) => dt.items.add(file));
    setSelectedFiles((prevFiles) => mergeFileList(prevFiles, dt.files));
  };

  useEffect(() => {
    if (!isUploadDialogOpen && !viewingDocument && !folderToDelete) {
      // Reset pointer-events after the dialog closes
      document.body.style.pointerEvents = "auto";
    }
  }, [isUploadDialogOpen, viewingDocument, folderToDelete]);

  const handleDialogClose = () => {
    // Clear the state and close the dialog
    setIsUploadDialogOpen(false);
    setIsConverting(false);
    setSelectedFiles(null);
  };

  const handleConversion = async () => {
    if (!selectedFiles) return;

    setIsConverting(true);
    const formData = new FormData();

    // First, save all original documents and collect their paths
    const originalDocsFormData = new FormData();
    Array.from(selectedFiles).forEach((file) => {
      formData.append("files", file);
      originalDocsFormData.append(
        "files",
        new File([file], file.name, { type: file.type })
      );
    });
    originalDocsFormData.append("namespace", namespace);
    formData.append("conversion_method", conversionMethod);

    try {
      // First save the original documents
      const saveDocsResponse = await fetch("/api/saveDocuments", {
        method: "POST",
        body: originalDocsFormData,
      });

      if (!saveDocsResponse.ok) {
        throw new Error("Failed to save original documents");
      }

      const savedDocs = await saveDocsResponse.json();

      // Choose endpoint based on conversion method
      const conversionEndpoint = "/api/convertDocuments";

      // Prepare headers for Azure if needed
      const headers: HeadersInit = {};
      if (conversionMethod === "azure") {
        headers["azure-endpoint"] = azureEndpoint;
        headers["azure-key"] = azureKey;
      }

      // Then proceed with conversion
      const response = await fetch(conversionEndpoint, {
        method: "POST",
        body: formData,
        headers,
      });

      if (!response.ok) {
        throw new Error("Failed to convert documents");
      }

      const result = await response.json();

      // Create a folder name based on timestamp
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const folderName = `converted_${timestamp}`;

      // Add original documents to the file explorer
      savedDocs.files.forEach((savedDoc: { name: string; path: string }) => {
        const originalFile = {
          name: savedDoc.name,
          path: savedDoc.path,
          type: "document" as const,
          parentFolder: folderName,
        };
        onFileUpload(originalFile);
      });

      // Create and upload the JSON result file
      const jsonFile = new File(
        [JSON.stringify(result.documents, null, 2)],
        `docs_${timestamp}.json`,
        { type: "application/json" }
      );

      const jsonFormData = new FormData();
      jsonFormData.append("file", jsonFile);
      jsonFormData.append("namespace", namespace);
      const uploadResponse = await fetch("/api/uploadFile", {
        method: "POST",
        body: jsonFormData,
      });

      if (!uploadResponse.ok) {
        throw new Error("Failed to upload converted file");
      }

      const uploadData = await uploadResponse.json();

      const newFile = {
        name: jsonFile.name,
        path: uploadData.path,
        type: "json" as const,
        parentFolder: folderName,
      };

      onFileUpload(newFile);
      setCurrentFile(newFile);
      handleDialogClose();

      toast({
        title: "Success",
        description:
          "Documents converted and uploaded successfully. We recommend downloading your dataset json file.",
      });
    } catch (error) {
      console.error("Error processing files:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to process files. Please try again.",
      });
    } finally {
      setIsConverting(false);
    }
  };

  const handleFileDownload = async (file: File) => {
    try {
      const response = await fetch(
        `/api/readFile?path=${encodeURIComponent(file.path)}`
      );
      if (!response.ok) {
        throw new Error("Failed to download file");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.name;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Error downloading file:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to download file",
      });
    }
  };

  const handleFolderDeleteClick = (folderName: string) => {
    setFolderToDelete(folderName);
  };

  const handleFolderDeleteConfirm = () => {
    if (!folderToDelete) return;

    // Get all files in the folder
    const filesInFolder = files.filter(
      (file) => file.parentFolder === folderToDelete
    );

    // If current file is in this folder, clear it
    if (currentFile && currentFile.parentFolder === folderToDelete) {
      setCurrentFile(null);
    }

    // Delete all files in the folder
    filesInFolder.forEach((file) => {
      onFileDelete(file);
    });

    // Call the optional folder delete callback
    if (onFolderDelete) {
      onFolderDelete(folderToDelete);
    }

    setFolderToDelete(null);
  };

  return (
    <div className="h-full flex flex-col p-4 bg-background">
      <div className="flex justify-between items-center mb-4 border-b pb-3">
        <h2 className="text-base font-bold flex items-center">
          <Folder className="mr-2" size={14} />
          FILES
        </h2>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="icon" className="h-8 w-8">
              <Upload size={14} />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onSelect={(e) => e.preventDefault()}>
              <div className="flex items-center w-full">
                <Input
                  type="file"
                  accept=".json,application/json"
                  onChange={(e) => {
                    handleFileUpload(e);
                    e.currentTarget.value = "";
                  }}
                  className="hidden"
                  id="file-upload"
                  disabled={uploadingFiles.size > 0}
                />
                <label
                  htmlFor="file-upload"
                  className={`flex items-center w-full cursor-pointer ${
                    uploadingFiles.size > 0
                      ? "opacity-50 cursor-not-allowed"
                      : ""
                  }`}
                >
                  {uploadingFiles.size > 0 ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Database className="mr-2 h-4 w-4" />
                  )}
                  <span>
                    {uploadingFiles.size > 0
                      ? "Uploading dataset..."
                      : "Upload dataset.json"}
                  </span>
                </label>
              </div>
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setIsUploadDialogOpen(true)}>
              <FolderUp className="mr-2 h-4 w-4" />
              <span>Upload Files or Folder</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <div className="text-xs mb-4 bg-muted/50 p-2 rounded-md">
        <span className="text-muted-foreground font-medium">Tip: </span>
        Right-click files to view, download or delete them
      </div>

      <div className="overflow-y-auto flex-grow">
        {Object.entries(groupedFiles).map(([folder, folderFiles]) => (
          <div key={folder}>
            {folder !== "root" && (
              <ContextMenu>
                <ContextMenuTrigger className="flex items-center p-2 text-sm text-gray-600 relative hover:bg-gray-100 rounded-md">
                  <div className="absolute left-2">
                    <Folder className="h-4 w-4" />
                  </div>
                  <span className="pl-6 font-medium">{folder}</span>
                </ContextMenuTrigger>
                <ContextMenuContent className="w-64">
                  <ContextMenuItem
                    onClick={() => handleFolderDeleteClick(folder)}
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    <span>Delete Folder</span>
                  </ContextMenuItem>
                </ContextMenuContent>
              </ContextMenu>
            )}
            {folderFiles.map((file) => (
              <ContextMenu key={file.path}>
                <ContextMenuTrigger
                  className={`
                    group relative flex items-center w-full cursor-pointer 
                    hover:bg-accent p-2 rounded-md
                    ${currentFile?.path === file.path ? "bg-primary/10" : ""} 
                    ${folder !== "root" ? "ml-4" : ""}
                  `}
                  onClick={() => handleFileClick(file)}
                >
                  <div className="absolute left-2">
                    {uploadingFiles.has(file.name) ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : file.type === "json" ? (
                      <Database className="h-4 w-4" />
                    ) : (
                      <FileText className="h-4 w-4" />
                    )}
                  </div>
                  <div className="w-full pl-6 pr-2">
                    <span className="block text-sm">{file.name}</span>
                  </div>
                </ContextMenuTrigger>
                <ContextMenuContent className="w-64">
                  {file.type === "json" ? (
                    <ContextMenuItem
                      onClick={() => {
                        handleFileSelection(file);
                        setShowDatasetView(true);
                      }}
                    >
                      <Eye className="mr-2 h-4 w-4" />
                      <span>View Dataset</span>
                    </ContextMenuItem>
                  ) : (
                    <ContextMenuItem onClick={() => setViewingDocument(file)}>
                      <Eye className="mr-2 h-4 w-4" />
                      <span>View Document</span>
                    </ContextMenuItem>
                  )}
                  <ContextMenuItem onClick={() => handleFileDownload(file)}>
                    <Download className="mr-2 h-4 w-4" />
                    <span>Download</span>
                  </ContextMenuItem>
                  <ContextMenuItem onClick={() => onFileDelete(file)}>
                    <Trash2 className="mr-2 h-4 w-4" />
                    <span>Delete</span>
                  </ContextMenuItem>
                </ContextMenuContent>
              </ContextMenu>
            ))}
            {Array.from(uploadingFiles).map((filename) => {
              if (!folderFiles.some((f) => f.name === filename)) {
                return (
                  <div
                    key={`uploading-${filename}`}
                    className="group relative flex items-center w-full p-1"
                  >
                    <div className="absolute left-1">
                      <Loader2 className="h-4 w-4 animate-spin" />
                    </div>
                    <div className="w-full pl-5 pr-2 overflow-x-auto">
                      <span className="block whitespace-nowrap text-xs text-gray-500">
                        {filename}
                      </span>
                    </div>
                  </div>
                );
              }
              return null;
            })}
          </div>
        ))}
      </div>

      {viewingDocument && (
        <DocumentViewer
          isOpen={!!viewingDocument}
          onClose={() => setViewingDocument(null)}
          filePath={viewingDocument.path}
          fileName={viewingDocument.name}
        />
      )}

      {isUploadDialogOpen && (
        <Dialog
          open={isUploadDialogOpen}
          onOpenChange={(isOpen) => {
            if (!isOpen) {
              handleDialogClose();
            }
          }}
        >
          <DialogContent className="max-w-5xl w-full overflow-hidden max-h-[80vh] flex flex-col">
            <DialogHeader>
              <DialogTitle>Upload Documents</DialogTitle>
            </DialogHeader>

            <div className="space-y-3 w-full flex-1 overflow-hidden flex flex-col">
              <div>
                <Label className="text-sm font-medium text-gray-700">
                  Processing Method
                </Label>

                <RadioGroup
                  value={conversionMethod}
                  onValueChange={(value) =>
                    setConversionMethod(value as ConversionMethod)
                  }
                  className="mt-2 grid grid-cols-3 gap-2"
                >
                  <div className="flex flex-col space-y-1 p-2 rounded-md transition-colors hover:bg-gray-50 cursor-pointer border border-gray-100">
                    <div className="flex items-start space-x-2.5">
                      <RadioGroupItem
                        value="local"
                        id="local-server"
                        className="mt-0.5"
                      />
                      <Label
                        htmlFor="local-server"
                        className="text-sm font-medium cursor-pointer"
                      >
                        Local Server
                      </Label>
                    </div>
                    <p className="text-xs text-muted-foreground pl-6">
                      Process documents privately on your machine (this can be
                      slow for many documents)
                    </p>
                    {conversionMethod === "local" && (
                      <div className="bg-destructive/10 text-destructive rounded-md p-2 mt-1 text-xs">
                        <div className="flex gap-2">
                          <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                          <div>
                            <p className="font-medium">Local Server Required</p>
                            <p className="mt-1">
                              This option requires running the application
                              locally with the server component.
                            </p>
                            <button
                              className="text-destructive underline hover:opacity-80 mt-1.5 font-medium"
                              onClick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                              }}
                            >
                              Continue anyway if running locally
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="flex flex-col space-y-1 p-2 rounded-md transition-colors hover:bg-gray-50 cursor-pointer border border-gray-100">
                    <div className="flex items-start space-x-2.5">
                      <RadioGroupItem
                        value="docetl"
                        id="docetl-server"
                        className="mt-0.5"
                      />
                      <Label
                        htmlFor="docetl-server"
                        className="text-sm font-medium cursor-pointer"
                      >
                        Docling Server{" "}
                        <a
                          href="https://github.com/DS4SD/docling"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-blue-600 hover:underline"
                        >
                          (GitHub ↗)
                        </a>
                      </Label>
                    </div>
                    <p className="text-xs text-muted-foreground pl-6">
                      Use our hosted server for fast and accurate processing
                      across many documents
                    </p>
                  </div>

                  <div className="flex flex-col space-y-1 p-2 rounded-md transition-colors hover:bg-gray-50 cursor-pointer border border-gray-100">
                    <div className="flex items-start space-x-2.5">
                      <RadioGroupItem
                        value="azure"
                        id="azure-di"
                        className="mt-0.5"
                      />
                      <Label
                        htmlFor="azure-di"
                        className="text-sm font-medium cursor-pointer"
                      >
                        Azure Document Intelligence
                      </Label>
                    </div>
                    <p className="text-xs text-muted-foreground pl-6">
                      Enterprise-grade cloud processing
                    </p>
                  </div>
                </RadioGroup>
              </div>

              {conversionMethod === "azure" && (
                <div className="grid gap-2 animate-in fade-in slide-in-from-top-1">
                  <div className="space-y-1">
                    <Label htmlFor="azure-endpoint" className="text-sm">
                      Azure Endpoint
                    </Label>
                    <Input
                      id="azure-endpoint"
                      placeholder="https://your-resource.cognitiveservices.azure.com/"
                      value={azureEndpoint}
                      onChange={(e) => setAzureEndpoint(e.target.value)}
                      className="h-8"
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-1">
                      <Label htmlFor="azure-key" className="text-sm">
                        Azure Key
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
                          </TooltipTrigger>
                          <TooltipContent className="bg-amber-50 border-amber-200 text-amber-900 w-[150px] text-xs">
                            <p>
                              Warning: Key is passed in plaintext to your local
                              server
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <Input
                      id="azure-key"
                      type="password"
                      placeholder="Enter your Azure Document Intelligence key"
                      value={azureKey}
                      onChange={(e) => setAzureKey(e.target.value)}
                      className="h-8"
                    />
                  </div>
                </div>
              )}

              <div
                className={`
                  border-2 border-dashed rounded-lg transition-colors relative flex-shrink-0
                  ${
                    selectedFiles && selectedFiles.length > 0
                      ? "border-border bg-accent/50 p-6"
                      : "border-border p-8 hover:border-primary"
                  }
                `}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  e.currentTarget.classList.add(
                    "border-blue-400",
                    "bg-blue-50/50"
                  );
                  if (e.dataTransfer.items) {
                    setDraggedFiles(e.dataTransfer.items.length);
                  }
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  e.currentTarget.classList.remove(
                    "border-blue-400",
                    "bg-blue-50/50"
                  );
                  setDraggedFiles(0);
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  e.currentTarget.classList.remove(
                    "border-blue-400",
                    "bg-blue-50/50"
                  );
                  setDraggedFiles(0);
                  if (e.dataTransfer.items) {
                    handleFolderUpload(e.dataTransfer.items);
                  } else if (e.dataTransfer.files) {
                    handleFolderUpload(e.dataTransfer.files);
                  }
                }}
              >
                {draggedFiles > 0 && (
                  <div className="absolute inset-0 flex items-center justify-center bg-blue-50/90 backdrop-blur-[1px] rounded-lg border-2 border-blue-400">
                    <div className="flex flex-col items-center space-y-1">
                      <Upload className="w-6 h-6 text-blue-500 animate-bounce" />
                      <span className="text-sm font-medium text-blue-600">
                        Drop {draggedFiles} file{draggedFiles !== 1 ? "s" : ""}{" "}
                        to convert
                      </span>
                    </div>
                  </div>
                )}

                <div className="text-center">
                  <Upload className="w-10 h-10 mx-auto text-gray-400 mb-4" />
                  <div className="space-y-2">
                    <p className="text-sm text-gray-600">
                      Drag and drop your documents here or
                    </p>
                    <label>
                      <input
                        type="file"
                        multiple
                        className="hidden"
                        accept={SUPPORTED_EXTENSIONS.join(",")}
                        onChange={(e) => {
                          if (e.target.files) {
                            handleFolderUpload(e.target.files);
                          }
                        }}
                      />
                      <span className="text-sm text-primary hover:text-blue-600 cursor-pointer">
                        browse files
                      </span>
                    </label>
                    <p className="text-xs text-gray-500">
                      Supported formats: PDF, DOCX, DOC, TXT, HTML, PPTX, MD
                    </p>
                    <p className="text-xs text-gray-500">
                      Processing may take up to 2 minutes
                    </p>
                  </div>
                </div>
              </div>

              {selectedFiles && selectedFiles.length > 0 && (
                <div className="space-y-4 flex-1 min-h-0 flex flex-col">
                  <div className="border rounded-lg divide-y max-w-full flex-1 overflow-y-auto">
                    {Array.from(selectedFiles).map((file, index) => (
                      <div
                        key={`${file.name}-${index}`}
                        className="flex items-center py-2.5 px-4 group hover:bg-gray-50"
                      >
                        <FileText className="h-4 w-4 flex-shrink-0 text-gray-400 mr-3" />
                        <div className="min-w-0 flex-1 overflow-hidden">
                          <div className="flex items-center">
                            <p className="text-sm font-medium text-gray-700 truncate">
                              {/* @ts-expect-error FileWithPath type is not fully defined */}
                              {(file as FileWithPath).relativePath || file.name}
                            </p>
                          </div>
                          <p className="text-xs text-gray-500">
                            {(file.size / 1024).toFixed(1)} KB
                          </p>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="ml-2 h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                          onClick={() => {
                            const dt = new DataTransfer();
                            Array.from(selectedFiles).forEach((f, i) => {
                              if (i !== index) dt.items.add(f);
                            });
                            setSelectedFiles(dt.files);
                          }}
                        >
                          <X className="h-4 w-4 text-gray-500 hover:text-gray-700" />
                        </Button>
                      </div>
                    ))}
                  </div>

                  <div className="flex items-center justify-between pt-2 flex-shrink-0">
                    <div className="flex items-center space-x-4">
                      <p className="text-sm text-gray-600">
                        {selectedFiles.length} file
                        {selectedFiles.length !== 1 ? "s" : ""} selected
                      </p>
                      <label className="text-sm text-primary hover:text-blue-600 cursor-pointer">
                        <input
                          type="file"
                          multiple
                          className="hidden"
                          accept={SUPPORTED_EXTENSIONS.join(",")}
                          onChange={(e) => {
                            if (e.target.files) {
                              handleFolderUpload(e.target.files);
                            }
                          }}
                        />
                        Add more
                      </label>
                    </div>
                    <Button
                      onClick={handleConversion}
                      disabled={
                        isConverting ||
                        (conversionMethod === "azure" &&
                          (!azureEndpoint || !azureKey))
                      }
                      className="min-w-[100px]"
                    >
                      {isConverting ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Converting
                        </>
                      ) : (
                        "Convert"
                      )}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>
      )}

      {folderToDelete && (
        <AlertDialog
          open={!!folderToDelete}
          onOpenChange={() => setFolderToDelete(null)}
        >
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete Folder</AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete this folder and all its
                contents? This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={handleFolderDeleteConfirm}
                className="bg-red-600 hover:bg-red-700"
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
};
