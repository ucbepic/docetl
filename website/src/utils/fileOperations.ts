import { getBackendUrl } from "@/lib/api-config";
import type { File } from "@/app/types";

interface SaveFileOptions {
  currentFile: File | null;
  namespace: string;
}

interface RestoreFileHandlers {
  setFiles: (updater: (prev: File[]) => File[]) => void;
  setCurrentFile: (file: File | null) => void;
}

// Define a type for the data structure being saved and loaded
interface PipelineData {
  currentFile?: {
    name: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

// Type for file contents stored in saved pipeline data
interface FileContentsMap {
  [fileName: string]: {
    content: Record<string, unknown>;
    metadata: File;
  };
}

// Type for data being saved with file contents
interface PipelineDataWithFileContents extends PipelineData {
  __fileContents?: FileContentsMap;
}

export const saveToFile = async (
  data: PipelineData,
  defaultFilename: string,
  options?: SaveFileOptions
) => {
  try {
    // Include the dataset content if currentFile is provided
    const dataToSave = { ...data };

    if (options?.currentFile && options.namespace) {
      try {
        const fileResponse = await fetch(
          `${getBackendUrl()}/fs/read-file?path=${encodeURIComponent(
            options.currentFile.path
          )}`
        );

        if (fileResponse.ok) {
          const fileContent = await fileResponse.json();
          dataToSave["__fileContents"] = {
            [options.currentFile.name]: {
              content: fileContent,
              metadata: options.currentFile,
            },
          };
        }
      } catch (err) {
        console.error("Failed to fetch file content:", err);
        // Continue with saving other data even if file content fetch fails
      }
    }

    // Use the showSaveFilePicker API to let user choose save location and filename
    const handle = await window.showSaveFilePicker({
      suggestedName: defaultFilename,
      types: [
        {
          description: "DocETL Pipeline File",
          accept: {
            "application/json": [".dtl"],
          },
        },
      ],
    });

    // Create a writable stream and write the data
    const writable = await handle.createWritable();
    await writable.write(JSON.stringify(dataToSave, null, 2));
    await writable.close();
  } catch (err) {
    // User cancelled or other error
    console.error("Error saving file:", err);
    throw err;
  }
};

export const loadFromFile = async (
  handlers?: RestoreFileHandlers
): Promise<PipelineData> => {
  try {
    // Use the showOpenFilePicker API for a better file selection experience
    const [handle] = await window.showOpenFilePicker({
      types: [
        {
          description: "DocETL Pipeline File",
          accept: {
            "application/json": [".dtl"],
          },
        },
      ],
      multiple: false,
    });

    const file = await handle.getFile();
    const text = await file.text();
    const data = JSON.parse(text);

    // Process file contents if they exist and handlers are provided
    await processFileContents(data, handlers);

    // Remove the file contents from the data before returning
    if (data.__fileContents) {
      // Use _ to indicate we're intentionally not using this variable
      const { __fileContents: _, ...restData } = data;
      return restData;
    }

    return data;
  } catch (err) {
    // User cancelled or other error
    console.error("Error loading file:", err);
    throw err;
  }
};

// Fallback for browsers that don't support the File System Access API
export const saveToFileClassic = async (
  data: PipelineData,
  defaultFilename: string,
  options?: SaveFileOptions
) => {
  // Include the dataset content if currentFile is provided
  const dataToSave = { ...data };

  if (options?.currentFile && options.namespace) {
    try {
      const fileResponse = await fetch(
        `${getBackendUrl()}/fs/read-file?path=${encodeURIComponent(
          options.currentFile.path
        )}`
      );

      if (fileResponse.ok) {
        const fileContent = await fileResponse.json();
        dataToSave["__fileContents"] = {
          [options.currentFile.name]: {
            content: fileContent,
            metadata: options.currentFile,
          },
        };
      }
    } catch (err) {
      console.error("Failed to fetch file content:", err);
      // Continue with saving other data even if file content fetch fails
    }
  }

  const blob = new Blob([JSON.stringify(dataToSave, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = defaultFilename;

  // Append to document, click, and cleanup
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export const loadFromFileClassic = async (
  handlers?: RestoreFileHandlers
): Promise<PipelineData> => {
  return new Promise((resolve, reject) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".dtl";

    input.onchange = async (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) {
        reject(new Error("No file selected"));
        return;
      }

      try {
        const text = await file.text();
        const data = JSON.parse(text);

        // Process file contents if they exist and handlers are provided
        await processFileContents(data, handlers);

        // Remove the file contents from the data before returning
        if (data.__fileContents) {
          // Use _ to indicate we're intentionally not using this variable
          const { __fileContents: _, ...restData } = data;
          resolve(restData);
        } else {
          resolve(data);
        }
      } catch (error) {
        reject(error);
      }
    };

    input.click();
  });
};

// Helper function to process file contents
async function processFileContents(
  data: PipelineDataWithFileContents,
  handlers?: RestoreFileHandlers
): Promise<void> {
  if (!data.__fileContents || !handlers) return;

  const fileContents = data.__fileContents as FileContentsMap;

  // For each saved file, recreate it in the current session
  for (const [fileName, fileData] of Object.entries(fileContents)) {
    try {
      // Create a blob from the content
      const fileContent = JSON.stringify(fileData.content);
      const fileBlob = new Blob([fileContent], { type: "application/json" });

      // Create a Form for the upload
      const formData = new FormData();
      formData.append("file", fileBlob, fileName);
      formData.append("namespace", fileData.metadata.parentFolder || "root");

      // Upload to server
      const response = await fetch(`${getBackendUrl()}/fs/upload-file`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const uploadResult = await response.json();

        // Create the file object with the new path
        const newFile = {
          ...fileData.metadata,
          path: uploadResult.path,
        };

        // Update the files list
        handlers.setFiles((prev) => [...prev, newFile]);

        // Set as current file if appropriate
        const currentFileName = data.currentFile?.name;
        if (currentFileName === fileName) {
          handlers.setCurrentFile(newFile);
        }
      }
    } catch (error) {
      console.error("Error restoring file:", fileName, error);
    }
  }
}
