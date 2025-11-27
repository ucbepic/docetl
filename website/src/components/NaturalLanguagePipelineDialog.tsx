import React, { useState, useEffect, useMemo } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  AlertCircle,
  CheckCircle2,
  FileText,
  Loader2,
  Sparkles,
  Upload,
  Trash2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { File as FileType, Operation, OutputType } from "@/app/types";
import { toast } from "@/hooks/use-toast";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";
import { useRestorePipeline } from "@/hooks/useRestorePipeline";

// Extract content from XML-style tags
function extractTagContent(text: string, tag: string): string | null {
  // Create a regex that works with any character including newlines
  const pattern = "<" + tag + ">([\\s\\S]*?)</" + tag + ">";
  const regex = new RegExp(pattern, "i");
  const match = text.match(regex);

  // If we find a match, clean it up by trimming whitespace
  let content = match && match[1] ? match[1].trim() : null;

  // Remove ```yaml and ``` if present
  if (content) {
    content = content.replace(/^```yaml\n/, "").replace(/```$/, "");
    content = content.trim();
  }

  // Debug the extraction to help with troubleshooting
  if (content && tag === "name") {
    console.debug(
      `[extractTagContent] Found name tag with content: "${content}"`
    );
  }

  return content;
}

// Helper function to extract a sample of document data for the LLM
const extractSampleData = (jsonData: unknown[], sampleSize = 5): string => {
  if (!Array.isArray(jsonData) || jsonData.length === 0) {
    return "No data available";
  }

  // Get all unique keys from the first few documents
  const keys = Object.keys(jsonData[0]);

  // Take a sample of documents
  const samples = jsonData.slice(0, sampleSize);

  // Format the data for the prompt
  let formattedData = `Dataset has ${
    jsonData.length
  } documents with the following fields: ${keys.join(", ")}\n\n`;
  formattedData += "Sample documents:\n";

  samples.forEach((doc, index) => {
    formattedData += `\nDocument ${index + 1}:\n`;
    Object.entries(doc).forEach(([key, value]) => {
      // Limit each field to 500 characters
      const stringValue =
        typeof value === "string" ? value : JSON.stringify(value);
      const truncatedValue =
        stringValue.length > 500
          ? stringValue.substring(0, 500) + "..."
          : stringValue;
      formattedData += `  ${key}: ${truncatedValue}\n`;
    });
  });

  return formattedData;
};

const formatFileSize = (bytes?: number | null): string => {
  if (!bytes || bytes <= 0) {
    return "Unknown size";
  }
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1
  );
  const value = bytes / Math.pow(1024, exponent);
  return `${value % 1 === 0 ? value : value.toFixed(1)} ${units[exponent]}`;
};

const getFileExtension = (input?: string | null): string => {
  if (!input) {
    return "";
  }
  const lastSegment = input.split("/").pop() || input;
  const parts = lastSegment.split(".");
  if (parts.length < 2) {
    return "";
  }
  return parts.pop()?.toLowerCase() || "";
};

const parseDatasetContent = (text: string, extension: string): unknown[] => {
  if (extension === "csv") {
    const lines = text.split("\n").map((line) => line.trim());
    if (lines.length === 0) {
      return [];
    }
    const headers = lines[0]
      .split(",")
      .map((header) => header.trim())
      .filter(Boolean);
    if (headers.length === 0) {
      return [];
    }
    const rows: Record<string, string>[] = [];
    for (let i = 1; i < lines.length; i += 1) {
      const line = lines[i];
      if (!line) {
        continue;
      }
      const values = line.split(",");
      const record: Record<string, string> = {};
      headers.forEach((header, index) => {
        record[header] = values[index]?.trim() ?? "";
      });
      rows.push(record);
    }
    return rows;
  }

  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) {
      return parsed;
    }
    if (parsed && typeof parsed === "object") {
      return [parsed];
    }
  } catch (error) {
    console.error("Failed to parse dataset content:", error);
  }

  return [];
};

interface NaturalLanguagePipelineDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  namespace: string;
  onFileUpload: (file: FileType) => void;
  setCurrentFile: (file: FileType | null) => void;
  setOperations: (operations: Operation[]) => void;
  setPipelineName: (name: string) => void;
  setSampleSize: (size: number | null) => void;
  setDefaultModel: (model: string) => void;
  setFiles: (files: FileType[]) => void;
  setSystemPrompt: (prompt: {
    datasetDescription: string | null;
    persona: string | null;
  }) => void;
  currentFile: FileType | null;
  files: FileType[];
  setOutput: (output: OutputType | null) => void;
  defaultModel: string;
}

const NaturalLanguagePipelineDialog: React.FC<
  NaturalLanguagePipelineDialogProps
> = ({
  open,
  onOpenChange,
  namespace,
  onFileUpload,
  setCurrentFile,
  setOperations,
  setPipelineName,
  setSampleSize,
  setDefaultModel,
  setFiles,
  setSystemPrompt,
  currentFile,
  files,
  defaultModel,
  setOutput,
}) => {
  const [activeTab, setActiveTab] = useState<string>("upload");
  const [uploadedFile, setUploadedFile] = useState<FileType | null>(null);
  const [fileData, setFileData] = useState<unknown[]>([]);
  const [userPrompt, setUserPrompt] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generatedYaml, setGeneratedYaml] = useState<string>("");
  const [generatedName, setGeneratedName] = useState<string>("");
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [generatedOutput, setGeneratedOutput] = useState<string>("");
  const [instructions, setInstructions] = useState<string>("");

  const steps = [
    {
      id: "upload",
      label: "Upload & Describe",
      description: "Provide your dataset and describe the outcome you need.",
    },
    {
      id: "review",
      label: "Review & Apply",
      description: "Inspect the generated pipeline and apply it to the workspace.",
    },
  ] as const;

  const activeStepIndex = steps.findIndex((step) => step.id === activeTab);
  const canAccessReview = Boolean(generatedYaml || generatedOutput);

  const datasetName =
    uploadedFile?.name ?? currentFile?.name ?? "No dataset selected yet";
  const datasetSize = uploadedFile?.blob
    ? formatFileSize(uploadedFile.blob.size)
    : currentFile?.blob
    ? formatFileSize(currentFile.blob.size)
    : null;

  const operationCount = useMemo(() => {
    if (!generatedYaml) {
      return 0;
    }
    const matches = generatedYaml.match(/^\s{0,6}-\sname:/gm);
    return matches ? matches.length : 0;
  }, [generatedYaml]);

  const handleStepChange = (stepId: (typeof steps)[number]["id"]) => {
    if (stepId === "review" && !canAccessReview) {
      return;
    }
    setActiveTab(stepId);
  };

  const previewText = isGenerating
    ? "Generating pipeline..."
    : generatedYaml ||
      generatedOutput ||
      "No pipeline generated yet. Upload a dataset and describe your goal to draft one.";

  const statusLabel = isGenerating
    ? "Generating"
    : canAccessReview
    ? "Ready to apply"
    : "Draft";

  const { uploadLocalDataset } = useDatasetUpload({
    namespace,
    onFileUpload: (file: FileType) => {
      setUploadedFile(file);
      onFileUpload(file);
      setCurrentFile(file);
    },
    setCurrentFile,
  });

    const { restoreFromYAML } = useRestorePipeline({
      setOperations,
      setPipelineName,
      setSampleSize,
      setDefaultModel,
      setFiles,
      setCurrentFile,
      setSystemPrompt,
      files,
      setOutput,
    });

    // Setup request headers
    const getRequestHeaders = useMemo(() => {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        "x-source": "nl-pipeline-generator",
        "x-model": "gpt-5",
        "x-namespace": namespace,
      };

      // Check if personal OpenAI API key should be used
      const usePersonalOpenAI =
        localStorage.getItem("USE_PERSONAL_OPENAI") === "true";

      if (usePersonalOpenAI) {
        headers["x-use-openai"] = "true";
        // Try to get API key from localStorage
        const openAiKey = localStorage.getItem("OPENAI_API_KEY");
        if (openAiKey) {
          headers["x-openai-key"] = openAiKey;
        }
      }

      return headers;
    }, [namespace]);

  // Get instructions on component mount
  useEffect(() => {
    const fetchInstructions = async () => {
      try {
        const response = await fetch("/llms-full.txt");
        if (!response.ok) {
          throw new Error("Failed to fetch LLM instructions");
        }
        const text = await response.text();
        setInstructions(text);
      } catch (error) {
        console.error("Error loading LLM instructions:", error);
        setInstructions("Error loading instructions. Please try again.");
      }
    };

    fetchInstructions();
  }, []);

  // Generate pipeline using the new endpoint
  const handleGenerate = async () => {
    if (!uploadedFile && !currentFile) {
      toast({
        title: "No file uploaded",
        description: "Please upload a data file first.",
        variant: "destructive",
      });
      return;
    }

    if (!userPrompt.trim()) {
      toast({
        title: "No prompt provided",
        description: "Please describe what you want to do with your data.",
        variant: "destructive",
      });
      return;
    }

    setIsGenerating(true);
    setGeneratedOutput("");
    setGeneratedName(""); // Reset generated values
    setGeneratedYaml("");

    // Move to the review tab immediately
    setActiveTab("review");

    try {
      const datasetFile = uploadedFile ?? currentFile;
      let datasetRecords = fileData;

      if (
        datasetRecords.length === 0 &&
        datasetFile?.path
      ) {
        try {
          const response = await fetch(
            `/api/readFile?path=${encodeURIComponent(datasetFile.path)}`
          );
          if (response.ok) {
            const text = await response.text();
            const extension =
              getFileExtension(datasetFile.path) ||
              getFileExtension(datasetFile.name);
            datasetRecords = parseDatasetContent(
              text,
              extension || "json"
            );
            if (datasetRecords.length > 0) {
              setFileData(datasetRecords);
            }
          } else {
            console.error(
              "Failed to read dataset for pipeline generation:",
              response.statusText
            );
          }
        } catch (error) {
          console.error("Error loading dataset for pipeline generation:", error);
        }
      }

      if (datasetRecords.length === 0 && uploadedFile?.blob) {
        try {
          const text = await uploadedFile.blob.text();
          const extension =
            getFileExtension(uploadedFile.path) ||
            getFileExtension(uploadedFile.name);
          datasetRecords = parseDatasetContent(text, extension || "json");
          if (datasetRecords.length > 0) {
            setFileData(datasetRecords);
          }
        } catch (error) {
          console.error("Error reading uploaded dataset blob:", error);
        }
      }

      const sampleData = extractSampleData(datasetRecords);

      // Create the combined user message
      const promptText = `You are a DocETL pipeline generator. Your task is to create a YAML pipeline configuration based on the user's requirements and data.

Follow these DocETL guidelines:
${instructions}

I need to create a DocETL pipeline for the following data:

${sampleData}

The task I need a pipeline for is: ${userPrompt}

Please provide:
1. A short, descriptive name for this pipeline (less than 30 characters)
2. A complete DocETL pipeline YAML that accomplishes this task including:
   - Datasets configuration with input path set to "DATASET_PATH_PLACEHOLDER" (exactly this string, which will be replaced with the real path)
   - Operations (map, reduce, etc.) with appropriate prompts that are jinja2 templates that reference the dataset keys, and simple output schemas
   - Refer to keys in input by using input.keyname; not just saying "keyname". In jinja2 for loops, use "for input in inputs" to iterate over the input documents.
   - All operation prompts should be jinja2 templates that reference the dataset keys
   - All operation prompts should instruct the LLM to provide examples and provenance when possible, and output schemas should be simple with open-ended fields if the user might want to know why the LLM made a certain decision
   - Output configuration (if needed) with output path set to "DATASET_PATH_PLACEHOLDER_OUTPUT" and intermediate_dir as "DATASET_PATH_PLACEHOLDER_INTERMEDIATES"
   - System prompt with dataset description and persona for the LLM to adopt when processing the data
   - Do not include backticks in the YAML
   - Use the model ${defaultModel} for the default_model
3. Do not include any resolve operations unless the user has explicitly requested it. Only use code operations if the data or logic needed is not fuzzy.
4. Start with a sample of 5 documents by saying "sample: 5" in the first operation.
5. Reduce operations should have key "_all" unless the user has explicitly requested for different answers for different groups of documents.
6. The last operation should be a reduce operation that summarizes the output of the previous operation in a presentable format (along with details like filenames or document names if relevant), unless the user has explicitly requested otherwise.

Format your response exactly as follows:
<name>
[The pipeline name you've created]
</name>

<pipeline>
[The complete YAML configuration]
</pipeline>`;

      // Call the new API endpoint
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: getRequestHeaders,
        body: JSON.stringify({ prompt: promptText }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      const generatedText = data.text;

      // Set the generated output
      setGeneratedOutput(generatedText);

      // Extract name and YAML
      const name = extractTagContent(generatedText, "name");
      const yaml = extractTagContent(generatedText, "pipeline");

      if (name) setGeneratedName(name);
      if (yaml) setGeneratedYaml(yaml);

      toast({
        title: "Pipeline generated",
        description: "Your pipeline has been generated successfully!",
      });
    } catch (error) {
      console.error("Error generating pipeline:", error);
      toast({
        title: "Error generating pipeline",
        description:
          error instanceof Error ? error.message : "An unknown error occurred",
        variant: "destructive",
      });

      // Go back to upload tab on error
      setActiveTab("upload");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;

    setIsUploading(true);
    try {
      const uploadedFile = e.target.files[0];
      const fileExtension =
        uploadedFile.name.split(".").pop()?.toLowerCase() || "json";

      if (!["json", "csv"].includes(fileExtension)) {
        toast({
          title: "Invalid file type",
          description: "Please upload a JSON or CSV file",
          variant: "destructive",
        });
        return;
      }

      // Create file object for uploadLocalDataset
      const fileToUpload: FileType = {
        name: uploadedFile.name,
        path: uploadedFile.name,
        type: "json",
        blob: uploadedFile,
      };

      // Read file for sample data extraction
      const text = await uploadedFile.text();
      const parsedData = parseDatasetContent(text, fileExtension);

      setFileData(parsedData);
      await uploadLocalDataset(fileToUpload);

      toast({
        title: "File uploaded",
        description: "Your data has been uploaded successfully",
      });
    } catch (error) {
      console.error("Error uploading file:", error);
      toast({
        title: "Upload failed",
        description:
          error instanceof Error ? error.message : "Failed to upload file",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleApplyPipeline = async () => {
    try {
      const datasetFile = uploadedFile ?? currentFile;
      if (!datasetFile) {
        toast({
          title: "No dataset available",
          description: "Please upload or select a dataset before applying the pipeline.",
          variant: "destructive",
        });
        return;
      }

      if (!generatedYaml) {
        toast({
          title: "Pipeline not ready",
          description: "Generate a pipeline before applying it.",
          variant: "destructive",
        });
        return;
      }

      // Use the generated name or default
      const pipelineName = generatedName || "generated-pipeline";

      const datasetPath = datasetFile.path || datasetFile.name;
      const pathWithoutExtension = datasetPath.replace(/\.(json|csv)$/i, "");

      // Replace placeholders in the YAML with actual paths
      const updatedYaml = generatedYaml
        .replace(/DATASET_PATH_PLACEHOLDER/g, datasetPath)
        .replace(
          /DATASET_PATH_PLACEHOLDER_OUTPUT/g,
          `${pathWithoutExtension}_output.json`
        )
        .replace(
          /DATASET_PATH_PLACEHOLDER_INTERMEDIATES/g,
          `${pathWithoutExtension}_intermediates`
        );

      // Create pipeline file object with the generated name
      const pipelineBlob = new Blob([updatedYaml], {
        type: "application/x-yaml",
      });
      const pipelineFile: FileType = {
        name: `${pipelineName}.yaml`,
        path: `${pipelineName}.yaml`,
        type: "pipeline-yaml",
        blob: pipelineBlob,
      };

      // Restore pipeline from the YAML
      await restoreFromYAML(pipelineFile);

      toast({
        title: "Pipeline Applied",
        description: `The "${pipelineName}" pipeline has been applied successfully.`,
      });

      onOpenChange(false);
    } catch (error) {
      console.error("Error applying pipeline:", error);
      toast({
        title: "Error",
        description: "Failed to apply the pipeline. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleDeleteFile = () => {
    // Reset file states
    setUploadedFile(null);
    setFileData([]);

    toast({
      title: "File deleted",
      description: "The uploaded dataset has been removed",
    });

    // Reset to first step if on review tab
    if (activeTab === "review") {
      setActiveTab("upload");
      setGeneratedYaml("");
      setGeneratedName("");
      setGeneratedOutput("");
    }
  };

    return (
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader className="space-y-3">
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              Generate Pipeline from Natural Language
            </DialogTitle>
            <DialogDescription>
              Upload your data and describe what you want to do with it. We&apos;ll
              use AI to generate a DocETL pipeline. Review the prompts before
              running anything in production. You can close this dialog at any time
              to upload PDFs or author your pipeline manually in the workspace.
            </DialogDescription>
          </DialogHeader>

          <Alert className="border-blue-200 bg-blue-50/50">
            <AlertCircle className="h-4 w-4 text-blue-600" />
            <AlertTitle className="text-sm font-semibold text-blue-900">
              Need to upload PDFs or author manually?
            </AlertTitle>
            <AlertDescription className="text-xs text-blue-800">
              You can close this dialog at any time to return to the workspace, where
              you can upload PDFs or manually author your pipeline using the visual
              editor.
            </AlertDescription>
          </Alert>

          <div className="space-y-6 pb-2">
            <div className="flex flex-col gap-2">
                <div className="flex flex-wrap items-center gap-2">
                {steps.map((step, index) => {
                  const isActive = index === activeStepIndex;
                  const isComplete = index < activeStepIndex;
                  const disabled = step.id === "review" && !canAccessReview;
                  const baseClasses =
                      "flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm transition-colors";
                  const stateClasses = isActive
                    ? "border-primary bg-primary/10 text-primary shadow-sm"
                    : isComplete
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                      : "border-border/60 bg-muted text-muted-foreground";
                  return (
                    <React.Fragment key={step.id}>
                      <button
                        type="button"
                        onClick={() => handleStepChange(step.id)}
                        disabled={disabled}
                        className={`${baseClasses} ${stateClasses} ${
                          disabled
                            ? "cursor-not-allowed opacity-60"
                            : "hover:border-primary/40"
                        }`}
                        >
                          <span className="flex h-6 w-6 items-center justify-center rounded-md border border-current text-xs font-semibold">
                          {index + 1}
                        </span>
                        <span className="whitespace-nowrap text-sm font-medium">
                          {step.label}
                        </span>
                      </button>
                      {index < steps.length - 1 && (
                        <span
                          className="hidden h-px w-8 rounded bg-border md:block"
                          aria-hidden="true"
                        />
                      )}
                    </React.Fragment>
                  );
                })}
              </div>
              <p className="text-xs text-muted-foreground">
                {steps[activeStepIndex]?.description}
              </p>
            </div>

            {activeTab === "upload" ? (
              <div className="grid gap-6 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)]">
                <div className="space-y-4">
                  <Card className="border shadow-sm">
                    <CardHeader className="space-y-2">
                      <CardTitle className="flex items-center gap-2 text-sm font-semibold">
                        <Upload className="h-4 w-4 text-primary" />
                        Dataset
                      </CardTitle>
                      <CardDescription className="text-xs">
                        Upload a CSV or JSON file to ground the generated pipeline.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <label
                          htmlFor="file-upload"
                          className={`relative flex min-h-[180px] cursor-pointer flex-col items-center justify-center gap-4 rounded-lg border border-dashed transition-colors ${
                            uploadedFile
                              ? "border-primary/60 bg-primary/5"
                              : "border-border bg-muted/30 hover:border-primary/60"
                          }`}
                        >
                        <input
                          type="file"
                          id="file-upload"
                          className="hidden"
                          accept=".csv,.json"
                          onChange={handleFileChange}
                          disabled={isUploading}
                        />
                          <div className="flex h-14 w-14 items-center justify-center rounded-lg bg-white shadow-sm">
                          <Upload className="h-6 w-6 text-primary" />
                        </div>
                        <div className="text-center">
                          <p className="text-sm font-medium text-foreground">
                            {uploadedFile ? "Replace dataset" : "Drag & drop your file"}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            or click to browse — supports CSV and JSON
                          </p>
                        </div>
                      </label>

                      {isUploading && (
                        <Alert className="border-primary/20 bg-primary/5">
                          <Loader2 className="h-4 w-4 animate-spin text-primary" />
                          <AlertTitle className="text-sm">
                            Uploading dataset
                          </AlertTitle>
                          <AlertDescription className="text-xs">
                            We&apos;ll start generating once the upload finishes.
                          </AlertDescription>
                        </Alert>
                      )}

                      {uploadedFile && !isUploading && (
                        <div className="rounded-md border border-primary/20 bg-primary/5 p-3 text-sm">
                          <div className="flex items-start justify-between gap-2">
                            <div className="space-y-1">
                              <div className="flex items-center gap-2 font-medium text-primary">
                                <FileText className="h-4 w-4" />
                                <span className="truncate">{uploadedFile.name}</span>
                              </div>
                              <p className="text-xs uppercase tracking-wide text-primary/70">
                                {datasetSize ?? "Size unavailable"}
                              </p>
                            </div>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-7 w-7 text-muted-foreground hover:text-destructive"
                              onClick={handleDeleteFile}
                              title="Remove dataset"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      )}
                    </CardContent>
                    <CardFooter className="flex flex-col gap-2 text-xs text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <AlertCircle className="h-3.5 w-3.5" />
                        Aim for a representative sample (under ~5MB) for faster
                        iteration.
                      </div>
                      {currentFile && !uploadedFile && (
                        <div className="flex items-center gap-2 text-emerald-600">
                          <CheckCircle2 className="h-3.5 w-3.5" />
                          Using dataset from your current workspace.
                        </div>
                      )}
                    </CardFooter>
                  </Card>
                </div>

                <div className="space-y-4">
                  <Alert className="border-primary/25 bg-primary/5">
                    <Sparkles className="h-4 w-4 text-primary" />
                    <AlertTitle className="text-sm font-semibold">
                      AI-assisted pipeline builder
                    </AlertTitle>
                    <AlertDescription className="text-xs">
                      We use <span className="font-medium">gpt-5</span> to draft your
                      pipeline. Review prompts and file paths before applying them.
                    </AlertDescription>
                  </Alert>

                  <Card className="border shadow-sm">
                    <CardHeader className="space-y-2">
                      <CardTitle className="text-sm font-semibold">
                        Describe your goal
                      </CardTitle>
                      <CardDescription className="text-xs">
                        Include the desired outcome, key columns, and any quality
                        checks you need.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <Textarea
                        placeholder="Example: Extract customer sentiment and key issues from these support tickets, then summarize by category."
                        value={userPrompt}
                        onChange={(e) => setUserPrompt(e.target.value)}
                        className="min-h-[200px] resize-none"
                      />
                      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
                        <span>
                          Tip: mention schema fields explicitly (e.g.,
                          <code className="ml-1 rounded bg-muted px-1">ticket_body</code>).
                        </span>
                        <Button
                          onClick={handleGenerate}
                          disabled={
                            (!uploadedFile && !currentFile) ||
                            !userPrompt.trim() ||
                            isGenerating
                          }
                          size="sm"
                        >
                          {isGenerating ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                              Generating
                            </>
                          ) : (
                            <>
                              <Sparkles className="mr-2 h-4 w-4" />
                              Generate pipeline
                            </>
                          )}
                        </Button>
                      </div>
                    </CardContent>
                  </Card>

                    <div className="rounded-md border border-dashed border-border bg-muted/20 p-3 text-xs text-muted-foreground">
                      Want to inspect our full prompt scaffolding?{" "}
                      <a
                        href="https://docetl.org/llms-full.txt"
                        target="_blank"
                        rel="noreferrer"
                        className="font-medium text-primary underline-offset-4 hover:underline"
                      >
                        Open the reference guide
                      </a>
                      .
                    </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {isGenerating && (
                  <Alert className="border-primary/20 bg-primary/5">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    <AlertTitle className="text-sm font-semibold">
                      Crafting your pipeline
                    </AlertTitle>
                    <AlertDescription className="text-xs">
                      gpt-5 is assembling operations, prompts, and schemas based on
                      your request.
                    </AlertDescription>
                  </Alert>
                )}

                <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_280px]">
                  <Card className="border shadow-sm flex flex-col overflow-hidden">
                    <CardHeader className="space-y-2">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <CardTitle className="text-sm font-semibold">
                          {generatedName || "Generated pipeline"}
                        </CardTitle>
                        <Badge
                          variant={isGenerating ? "secondary" : "default"}
                          className="flex items-center gap-1"
                        >
                          {isGenerating && (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          )}
                          {statusLabel}
                        </Badge>
                      </div>
                      <CardDescription className="text-xs">
                        Review the YAML before applying it. Adjust prompts or dataset
                        paths if needed.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="flex-1 overflow-auto pt-0">
                      <div className="rounded-md border border-dashed border-border bg-muted/40 p-4">
                        <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
                          {previewText}
                        </pre>
                      </div>
                    </CardContent>
                    <CardFooter className="sticky bottom-0 left-0 right-0 flex flex-wrap items-center justify-between gap-2 border-t bg-card/95 px-6 py-3 supports-[backdrop-filter]:bg-card/80">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => setActiveTab("upload")}
                        disabled={isGenerating}
                      >
                        Back to upload
                      </Button>
                      <div className="flex items-center gap-2">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            if (!generatedYaml || isGenerating) {
                              return;
                            }
                            if (
                              typeof navigator === "undefined" ||
                              !navigator.clipboard
                            ) {
                              toast({
                                title: "Copy unavailable",
                                description:
                                  "Clipboard access is not available in this environment.",
                                variant: "destructive",
                              });
                              return;
                            }
                            navigator.clipboard
                              .writeText(generatedYaml)
                              .then(() =>
                                toast({
                                  title: "Pipeline copied",
                                  description:
                                    "The generated YAML has been copied to your clipboard.",
                                })
                              )
                              .catch(() =>
                                toast({
                                  title: "Copy failed",
                                  description:
                                    "We couldn’t copy the YAML automatically. Please try again.",
                                  variant: "destructive",
                                })
                              );
                          }}
                          disabled={!generatedYaml || isGenerating}
                        >
                          Copy YAML
                        </Button>
                        <Button
                          onClick={handleApplyPipeline}
                          size="sm"
                          disabled={isGenerating || !generatedYaml}
                        >
                          Apply pipeline
                        </Button>
                      </div>
                    </CardFooter>
                  </Card>

                  <Card className="border shadow-sm h-fit">
                    <CardHeader className="space-y-2">
                      <CardTitle className="text-sm font-semibold">
                        Pipeline snapshot
                      </CardTitle>
                      <CardDescription className="text-xs">
                        Quick details about this draft.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4 text-sm">
                      <div className="space-y-1">
                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                          Dataset
                        </p>
                        <div className="flex items-center gap-2">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <span className="truncate">{datasetName}</span>
                        </div>
                        {datasetSize && (
                          <p className="text-xs text-muted-foreground">
                            {datasetSize}
                          </p>
                        )}
                      </div>
                      <div className="space-y-1">
                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                          Operations
                        </p>
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
                          <span>
                            {operationCount > 0
                              ? `${operationCount} configured`
                              : "Awaiting generation"}
                          </span>
                        </div>
                      </div>
                      <div className="space-y-1">
                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                          Model
                        </p>
                        <Badge variant="outline" className="text-xs">
                          {defaultModel}
                        </Badge>
                      </div>
                      <div className="space-y-1">
                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                          Next steps
                        </p>
                        <p className="text-xs text-muted-foreground">
                          Apply the pipeline to send it into the workspace, then review
                          prompts and schema inside the visual editor.
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    );
};

export default NaturalLanguagePipelineDialog;
