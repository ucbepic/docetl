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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Loader2, Sparkles, Upload, Trash2 } from "lucide-react";
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
      "x-model": "o1-mini",
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
    if (!uploadedFile) {
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
      // Extract sample data
      const sampleData = extractSampleData(fileData);

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
      const fileExtension = uploadedFile.name.split(".").pop()?.toLowerCase();

      if (!["json", "csv"].includes(fileExtension || "")) {
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
      let parsedData: unknown[] = [];

      if (fileExtension === "json") {
        parsedData = JSON.parse(text);
      } else {
        // For CSV, we'll use a simple parser
        // In a real implementation, you'd want to use Papa.parse like in useDatasetUpload
        const lines = text.split("\n");
        const headers = lines[0].split(",").map((h) => h.trim());

        for (let i = 1; i < lines.length; i++) {
          if (!lines[i].trim()) continue;
          const values = lines[i].split(",");
          const obj: Record<string, string> = {};
          headers.forEach((header, index) => {
            obj[header] = values[index]?.trim() || "";
          });
          parsedData.push(obj);
        }
      }

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
      if (!uploadedFile) {
        toast({
          title: "No file uploaded",
          description: "Cannot apply pipeline without an uploaded file.",
          variant: "destructive",
        });
        return;
      }

      // Use the generated name or default
      const pipelineName = generatedName || "generated-pipeline";

      // Replace placeholders in the YAML with actual paths
      const updatedYaml = generatedYaml
        .replace(/DATASET_PATH_PLACEHOLDER/g, uploadedFile.path)
        .replace(
          /DATASET_PATH_PLACEHOLDER_OUTPUT/g,
          `${uploadedFile.path.replace(/\.(json|csv)$/i, "")}_output.json`
        )
        .replace(
          /DATASET_PATH_PLACEHOLDER_INTERMEDIATES/g,
          `${uploadedFile.path.replace(/\.(json|csv)$/i, "")}_intermediates`
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
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            Generate Pipeline from Natural Language
          </DialogTitle>
          <DialogDescription>
            Upload your data and describe what you want to do with it.
            We&apos;ll use AI to generate a DocETL pipeline. Note that the
            pipeline might have errors, so you should review the prompts before
            you run it.
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="upload">1. Upload & Describe</TabsTrigger>
            <TabsTrigger
              value="review"
              disabled={!generatedOutput && !generatedYaml}
            >
              2. Review & Apply
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-4 mt-4">
            <Card className="border shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Upload Dataset
                </CardTitle>
                <CardDescription className="text-xs">
                  Upload your CSV or JSON data file to generate a pipeline.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="border border-dashed rounded-md p-3 text-center transition-colors border-border hover:border-primary">
                  <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept=".csv,.json"
                    onChange={handleFileChange}
                    disabled={isUploading}
                  />
                  <div className="flex flex-col items-center justify-center gap-1.5">
                    <Upload className="h-5 w-5 mx-auto text-gray-400 mb-1" />
                    <div className="flex items-center text-sm text-gray-500">
                      <span>Drag and drop files here, or</span>
                      <Label
                        htmlFor="file-upload"
                        className="ml-0.5 text-primary cursor-pointer hover:text-blue-600"
                      >
                        browse files
                      </Label>
                    </div>
                    <p className="mt-0.5 text-[10px] text-gray-400">
                      Accepted file types: .csv, .json
                    </p>
                  </div>
                </div>
                {isUploading && (
                  <div className="mt-2 flex items-center text-xs text-gray-600">
                    <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                    Uploading...
                  </div>
                )}
                {uploadedFile && !isUploading && (
                  <div className="mt-2 text-xs flex items-center justify-between">
                    <div className="text-green-600 flex items-center">
                      <div className="bg-green-100 text-green-600 p-0.5 rounded-full mr-1.5">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-3 w-3"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      </div>
                      {uploadedFile.name} uploaded successfully
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-5 w-5 text-gray-400 hover:text-red-500 rounded-full -m-1"
                      onClick={handleDeleteFile}
                      title="Delete dataset"
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="border shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Describe Your Goal
                </CardTitle>
                <CardDescription className="text-xs">
                  Tell us what you want to do with your data, and we&apos;ll
                  generate a pipeline for you.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-0">
                <Textarea
                  placeholder="Example: Extract customer sentiment and key issues from these support tickets, then summarize by category."
                  value={userPrompt}
                  onChange={(e) => setUserPrompt(e.target.value)}
                  className="min-h-[120px] resize-none"
                />
              </CardContent>
              <CardFooter className="pt-0">
                <Button
                  onClick={handleGenerate}
                  disabled={!uploadedFile || !userPrompt.trim() || isGenerating}
                  className="ml-auto"
                  size="sm"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Generate Pipeline
                    </>
                  )}
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="review" className="space-y-4 mt-4">
            <Card className="border shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-1.5">
                  {isGenerating ? (
                    <>
                      <Loader2 className="h-4 w-4 text-primary animate-spin" />
                      {generatedName ||
                        "Generating Pipeline with o1-mini (this may take a minute)..."}
                    </>
                  ) : generatedName ? (
                    <>
                      <Sparkles className="h-4 w-4 text-primary" />
                      {generatedName}
                    </>
                  ) : (
                    "Generated Pipeline"
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="bg-slate-50 rounded-md p-3 overflow-hidden border border-slate-100">
                  <pre className="text-xs overflow-x-auto whitespace-pre-wrap text-slate-700">
                    {isGenerating
                      ? "Generating pipeline..."
                      : generatedYaml || generatedOutput}
                  </pre>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between pt-0">
                <Button
                  variant="outline"
                  onClick={() => setActiveTab("upload")}
                  size="sm"
                  disabled={isGenerating}
                >
                  Back to Upload
                </Button>
                <Button
                  onClick={handleApplyPipeline}
                  size="sm"
                  disabled={isGenerating || !generatedYaml}
                >
                  Apply Pipeline
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
};

export default NaturalLanguagePipelineDialog;
