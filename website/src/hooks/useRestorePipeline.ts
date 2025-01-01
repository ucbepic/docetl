import { useCallback } from "react";
import yaml from "js-yaml";
import { v4 as uuidv4 } from "uuid";
import path from "path";
import { Operation, File } from "@/app/types";
import { schemaDictToItemSet } from "@/components/utils";
import { useToast } from "@/hooks/use-toast";

interface Dataset {
  type: string;
  path: string;
}

interface YAMLOperation {
  id?: string;
  type: string;
  name?: string;
  prompt?: string;
  output?: {
    schema: Record<string, unknown>;
  };
  validate?: unknown;
  sample?: unknown;
  [key: string]: unknown;
}

interface YAMLContent {
  operations?: YAMLOperation[];
  datasets?: Record<string, { type: string; path: string }>;
  default_model?: string;
  system_prompt?: {
    dataset_description: string | null;
    persona: string | null;
  };
}

interface RestorePipelineConfig {
  setOperations: (operations: Operation[]) => void;
  setPipelineName: (name: string) => void;
  setSampleSize: (size: number | null) => void;
  setDefaultModel: (model: string) => void;
  setFiles: (files: File[]) => void;
  setCurrentFile: (file: File | null) => void;
  setSystemPrompt: (prompt: {
    datasetDescription: string | null;
    persona: string | null;
  }) => void;
  currentFile: File | null;
  files: File[];
}

export const useRestorePipeline = ({
  setOperations,
  setPipelineName,
  setSampleSize,
  setDefaultModel,
  setFiles,
  setCurrentFile,
  setSystemPrompt,
  currentFile,
  files,
}: RestorePipelineConfig) => {
  const { toast } = useToast();

  const restoreFromYAML = useCallback(
    async (file: File) => {
      const reader = new FileReader();

      return new Promise<void>((resolve, reject) => {
        reader.onload = async (e) => {
          const content = e.target?.result;
          if (typeof content === "string") {
            try {
              const yamlFileName = file.name.split("/").pop()?.split(".")[0];
              const yamlContent = yaml.load(content) as YAMLContent;
              setOperations([]);

              // Update operations from YAML
              setOperations(
                (yamlContent.operations || []).map((op) => {
                  const {
                    id,
                    type,
                    name,
                    prompt,
                    output,
                    validate,
                    sample,
                    ...otherKwargs
                  } = op;

                  // If the operation type is 'reduce', ensure reduce_key is a list
                  if (type === "reduce" && otherKwargs.reduce_key) {
                    otherKwargs.reduce_key = Array.isArray(
                      otherKwargs.reduce_key
                    )
                      ? otherKwargs.reduce_key
                      : [otherKwargs.reduce_key];
                  }

                  return {
                    id: id || uuidv4(),
                    llmType:
                      type === "map" ||
                      type === "reduce" ||
                      type === "resolve" ||
                      type === "filter" ||
                      type === "parallel_map"
                        ? "LLM"
                        : "non-LLM",
                    type: type as Operation["type"],
                    name: name || "Untitled Operation",
                    prompt,
                    output: output
                      ? {
                          schema: schemaDictToItemSet(
                            output.schema as Record<string, string>
                          ),
                        }
                      : undefined,
                    validate,
                    sample,
                    otherKwargs,
                    visibility: true,
                  } as Operation;
                })
              );

              setPipelineName(yamlFileName || "Untitled Pipeline");
              setSampleSize(
                (yamlContent.operations?.[0]?.sample as number) || null
              );
              setDefaultModel(yamlContent.default_model || "gpt-4o-mini");
              setSystemPrompt({
                datasetDescription:
                  yamlContent.system_prompt?.dataset_description || null,
                persona: yamlContent.system_prompt?.persona || null,
              });

              // Look for paths in all datasets
              const datasetPaths = Object.values(yamlContent.datasets || {})
                .filter(
                  (dataset: Dataset) => dataset.type === "file" && dataset.path
                )
                .map((dataset: Dataset) => dataset.path);

              if (datasetPaths.length > 0) {
                const requiredPath = datasetPaths[0]; // Take the first dataset path
                const existingFile = files.find(
                  (file) => file.path === requiredPath
                );

                if (existingFile) {
                  // If the file exists, set it as current
                  setCurrentFile(existingFile);
                } else {
                  // If the file doesn't exist, show a toast message
                  toast({
                    title: "Dataset Required",
                    description: `This pipeline requires a dataset at path: ${requiredPath}. Please upload the dataset using the file explorer.`,
                    variant: "destructive",
                  });
                }
              }

              toast({
                title: "Pipeline Loaded",
                description:
                  "Your pipeline configuration has been loaded successfully.",
                duration: 3000,
              });

              resolve();
            } catch (error) {
              console.error("Error parsing YAML:", error);
              toast({
                title: "Error",
                description: "Failed to parse the uploaded YAML file.",
                variant: "destructive",
              });
              reject(error);
            }
          }
        };

        reader.onerror = (error) => reject(error);
        reader.readAsText(file.blob);
      });
    },
    [
      setOperations,
      setPipelineName,
      setSampleSize,
      setDefaultModel,
      setFiles,
      setCurrentFile,
      files,
      toast,
    ]
  );

  return { restoreFromYAML };
};
