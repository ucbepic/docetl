import { useState } from "react";
import { useToast } from "@/hooks/use-toast";
import type { File } from "@/app/types";

interface UseDatasetUploadOptions {
  namespace: string;
  onFileUpload: (file: File) => void;
  setCurrentFile: (file: File | null) => void;
}

export function useDatasetUpload({
  namespace,
  onFileUpload,
  setCurrentFile,
}: UseDatasetUploadOptions) {
  const { toast } = useToast();
  const [uploadingFiles, setUploadingFiles] = useState<Set<string>>(new Set());

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

  const uploadDataset = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".json")) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please upload a JSON file",
      });
      return;
    }

    // Add loading indicator immediately
    toast({
      title: "Uploading dataset...",
      description: "This may take a few seconds",
    });

    // Add to uploading files set to show spinner in file list
    setUploadingFiles((prev) => new Set(prev).add(file.name));

    try {
      // Validate JSON structure before uploading
      await validateJsonDataset(file.blob);

      const formData = new FormData();
      formData.append("file", file.blob);
      formData.append("namespace", namespace);

      const response = await fetch("/api/uploadFile", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();

      const newFile = {
        name: file.name,
        path: data.path,
        type: "json" as const,
        parentFolder: "root",
      };

      onFileUpload(newFile);
      setCurrentFile(newFile);

      toast({
        title: "Success",
        description: "Dataset uploaded successfully",
      });
    } catch (error) {
      console.error(error);
      toast({
        variant: "destructive",
        title: "Error",
        description:
          error instanceof Error ? error.message : "Failed to upload file",
      });
    } finally {
      setUploadingFiles((prev) => {
        const next = new Set(prev);
        next.delete(file.name);
        return next;
      });
    }
  };

  return {
    uploadingFiles,
    uploadDataset,
  };
}
