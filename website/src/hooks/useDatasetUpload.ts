import { useState } from "react";
import { useToast } from "@/hooks/use-toast";
import type { File } from "@/app/types";
import { getBackendUrl } from "@/lib/api-config";
import Papa from "papaparse";

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

  async function validateJsonDataset(data: unknown): Promise<void> {
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

  const convertCsvToJson = (csvText: string): Promise<unknown[]> => {
    return new Promise((resolve, reject) => {
      Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          if (results.errors.length > 0) {
            reject(
              new Error(`CSV parsing error: ${results.errors[0].message}`)
            );
          } else {
            resolve(results.data);
          }
        },
        error: (error) => {
          reject(new Error(`CSV parsing error: ${error.message}`));
        },
      });
    });
  };

  const uploadLocalDataset = async (file: File) => {
    const fileExtension = file.name.toLowerCase().split(".").pop();

    if (!["json", "csv"].includes(fileExtension || "")) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please upload a JSON or CSV file",
      });
      return;
    }

    toast({
      title: "Uploading dataset...",
      description: "This may take a few seconds",
    });

    setUploadingFiles((prev) => new Set(prev).add(file.name));

    try {
      let jsonData: unknown;

      if (fileExtension === "csv") {
        const csvText = await file.blob.text();
        jsonData = await convertCsvToJson(csvText);
      } else {
        const text = await file.blob.text();
        try {
          jsonData = JSON.parse(text);
        } catch {
          throw new Error("Invalid JSON format");
        }
      }

      await validateJsonDataset(jsonData);

      // Convert the JSON data back to a blob for upload
      const jsonBlob = new Blob([JSON.stringify(jsonData)], {
        type: "application/json",
      });
      const formData = new FormData();
      // Always save as .json regardless of input format
      const fileName = file.name.replace(/\.(json|csv)$/, ".json");
      formData.append("file", jsonBlob, fileName);
      formData.append("namespace", namespace);

      const response = await fetch(`${getBackendUrl()}/fs/upload-file`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();

      const newFile = {
        name: fileName,
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

  const uploadRemoteDataset = async (url: string) => {
    const fileName = url.split("/").pop() || "dataset.json";
    setUploadingFiles((prev) => new Set(prev).add(fileName));

    try {
      toast({
        title: "Downloading remote dataset...",
        description: "This may take a few seconds",
      });

      const formData = new FormData();
      formData.append("url", url);
      formData.append("namespace", namespace);

      const response = await fetch(`${getBackendUrl()}/fs/upload-file`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        // Get the response details
        const errorDetails = await response.json();
        throw new Error(
          errorDetails.detail || "Failed to fetch remote dataset"
        );
      }

      const data = await response.json();

      const newFile = {
        name: fileName.replace(/\.(json|csv)$/, ".json"),
        path: data.path,
        type: "json" as const,
        parentFolder: "root",
      };

      onFileUpload(newFile);
      setCurrentFile(newFile);

      toast({
        title: "Success",
        description: "Remote dataset downloaded and processed successfully",
      });
    } catch (error) {
      console.error(error);
      toast({
        variant: "destructive",
        title: "Error",
        description:
          error instanceof Error
            ? error.message
            : "Failed to fetch remote dataset",
      });
    } finally {
      setUploadingFiles((prev) => {
        const next = new Set(prev);
        next.delete(fileName);
        return next;
      });
    }
  };

  return {
    uploadingFiles,
    uploadLocalDataset,
    uploadRemoteDataset,
  };
}
