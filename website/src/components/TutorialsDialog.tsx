import React, { useState, useEffect } from "react";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";
import { useRestorePipeline } from "@/hooks/useRestorePipeline";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { File as FileType, Operation } from "@/app/types";

interface Tutorial {
  id: string;
  title: string;
  description: string;
  datasetUrl: string;
  datasetDescription: string;
  operations: string[];
  pipelineTemplate: string;
}

const TUTORIALS: Tutorial[] = [
  {
    id: "supreme-court",
    title: "Supreme Court Transcript Analysis",
    description:
      "Analyze Supreme Court oral arguments to understand how different Justices approach legal problems and how lawyers adapt their arguments.",
    datasetUrl:
      "https://drive.google.com/file/d/1n-muIvBYb3VGfZOYOBqJUYcO1swsBELt/view?usp=share_link",
    datasetDescription:
      "Collection of Supreme Court oral argument transcripts from 2024, covering various legal domains",
    operations: [
      "Map: Analyze reasoning patterns, justice opinions, and notable exchanges in individual transcripts",
      "Reduce: Synthesize patterns across multiple cases into a magazine-style article",
    ],
    pipelineTemplate: `datasets:
  input:
    type: file
    path: DATASET_PATH_PLACEHOLDER
    source: local
default_model: gpt-4o-mini
operations:
  - type: map
    name: find_patterns_in_reasoning
    prompt: >-
      You are analyzing a Supreme Court oral argument transcript to identify
      interesting patterns in legal reasoning. Your analysis will be read by
      people without legal training, so explain everything in clear, everyday
      language.

      Here is the transcript:

      {{ input.content }}

      Please analyze this transcript in everyday language:

      1. First, in 1-2 sentences, what's the key question this case is trying to
      answer? What area of law is this (e.g. tech privacy, free speech)?

      2. Find 4-5 notable exchanges between Justices and attorneys that show
      different ways of reasoning. For each exchange:
         - Quote the actual back-and-forth
         - Explain what's interesting about how they're thinking through the problem
         - Which Justice is asking the questions and what's distinctive about their approach?
         - What everyday situation would help understand this exchange?

      3. Justice Focus: Look at each Justice who spoke substantially:
         - What kinds of questions do they tend to ask?
         - What seems to concern them most?
         - How do lawyers adapt to their particular style?

      4. Looking at all these exchanges, what seems to be the main way the
      Justices are approaching this problem? Are they focused on real-world
      impacts, strict interpretation of laws, historical examples, etc?

      Avoid legal jargon - if you need to use a legal term, explain it like you
      would to a friend. Use concrete examples that anyone could understand.
    output:
      schema:
        summary: string
        legal_domain: string
    validate: []
  - type: reduce
    name: analyze_common_patterns
    prompt: >-
      You're writing an engaging magazine article about how the Supreme Court
      works in 2024. Below are detailed analyses of several oral arguments:

      {% for input in inputs %}

      Legal Domain: {{ input.legal_domain }}

      Case Analysis:

      {{ input.summary }}

      {% endfor %}

      Write an analysis that reveals fascinating patterns in how Supreme Court
      arguments work. Structure your analysis like a magazine feature article
      that explores:

      1. Different Justices, Different Styles
         - What's distinctive about how each Justice approaches problems?
         - Do certain Justices focus more on specific aspects (practical effects, text, precedent)?
         - How do their questioning styles differ?
         - Use specific exchanges to show these personal styles

      2. How Fields Shape the Conversation
         - How do tech cases differ from free speech cases? From business regulation?
         - Do certain Justices become more active in particular types of cases?
         - What unique challenges come up in each field?

      3. The Art of Persuasion
         - How do lawyers adapt their arguments for different Justices?
         - What strategies work across all cases vs. specific domains?
         - Show examples of lawyers skillfully handling different Justice's styles

      4. Bigger Patterns
         - What does this tell us about how the Court approaches problems?
         - Where do they prioritize practical effects vs. theoretical concerns?
         - Are there surprising similarities across very different cases?

      Write this for an intelligent but non-legal audience - help them
      understand the fascinating human dynamics of how America's highest court
      works. Use specific quotes and moments from the arguments, but explain
      everything like you're telling a story to an interested friend.
    output:
      schema:
        analysis: string
    reduce_key:
      - _all
pipeline:
  steps:
    - name: data_processing
      input: input
      operations:
        - find_patterns_in_reasoning
        - analyze_common_patterns
  output:
    type: file
    path: DATASET_PATH_PLACEHOLDER_OUTPUT
    intermediate_dir: DATASET_PATH_PLACEHOLDER_INTERMEDIATES
system_prompt:
  dataset_description: "a collection of Supreme Court oral argument transcripts"
  persona: "a legal analyst skilled at breaking down complex legal concepts for general audiences. You have extensive experience studying Supreme Court arguments and can identify key patterns while explaining them in clear, engaging language that helps non-lawyers understand the fascinating dynamics at play"`,
  },
];

interface TutorialsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedTutorial?: Tutorial;
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
}

export function TutorialsDialog({
  open,
  onOpenChange,
  selectedTutorial,
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
}: TutorialsDialogProps) {
  const { toast } = useToast();
  const { uploadDataset } = useDatasetUpload({
    namespace,
    onFileUpload,
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
    currentFile,
    files,
  });

  // Add state to track the uploaded dataset path
  const [uploadedDatasetPath, setUploadedDatasetPath] = useState<string | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(false);

  // Use effect to watch for currentFile changes
  useEffect(() => {
    if (uploadedDatasetPath && currentFile?.path) {
      const finishTutorialLoad = async () => {
        try {
          // Create pipeline YAML with the correct dataset path
          const pipelineYaml = selectedTutorial?.pipelineTemplate
            .replace(/DATASET_PATH_PLACEHOLDER/g, currentFile.path)
            .replace(
              /DATASET_PATH_PLACEHOLDER_OUTPUT/g,
              `${currentFile.path.replace(".json", "")}_output.json`
            )
            .replace(
              /DATASET_PATH_PLACEHOLDER_INTERMEDIATES/g,
              `${currentFile.path.replace(".json", "")}_intermediates`
            );

          if (!pipelineYaml) {
            throw new Error("Pipeline template not found");
          }

          // Create pipeline file object
          const pipelineBlob = new Blob([pipelineYaml], {
            type: "application/x-yaml",
          });
          const pipelineFile: FileType = {
            name: `${selectedTutorial?.id}-pipeline.yaml`,
            path: `${selectedTutorial?.id}-pipeline.yaml`,
            type: "pipeline-yaml",
            blob: pipelineBlob,
          };

          // Restore pipeline from the YAML
          await restoreFromYAML(pipelineFile);

          toast({
            title: "Tutorial Loaded",
            description:
              "The tutorial pipeline and dataset have been loaded successfully.",
          });

          // Reset states
          setUploadedDatasetPath(null);
          setIsLoading(false);
          onOpenChange(false);
        } catch (error) {
          console.error("Error loading tutorial:", error);
          toast({
            title: "Error",
            description: "Failed to load the tutorial. Please try again.",
            variant: "destructive",
          });
          setIsLoading(false);
        }
      };

      finishTutorialLoad();
    }
  }, [currentFile, uploadedDatasetPath, selectedTutorial]);

  if (!selectedTutorial) return null;

  const loadTutorial = async () => {
    try {
      setIsLoading(true);

      // Get file ID from Google Drive URL
      const fileId = selectedTutorial.datasetUrl.split("/")[5];

      // Download dataset through our API route
      const datasetResponse = await fetch(
        `/api/downloadTutorialDataset?fileId=${fileId}`
      );
      if (!datasetResponse.ok) {
        throw new Error("Failed to download dataset");
      }

      const datasetFileName = `${selectedTutorial.id}-dataset.json`;
      const datasetBlob = new File(
        [await datasetResponse.blob()],
        datasetFileName,
        {
          type: "application/json",
        }
      );

      // Create file object
      const datasetFile: FileType = {
        name: datasetFileName,
        path: datasetFileName,
        type: "json",
        blob: datasetBlob,
      };

      // Set the path we're expecting
      setUploadedDatasetPath(datasetFileName);

      // Upload dataset and wait for currentFile to update
      await uploadDataset(datasetFile);
    } catch (error) {
      console.error("Error loading tutorial:", error);
      toast({
        title: "Error",
        description: "Failed to load the tutorial. Please try again.",
        variant: "destructive",
      });
      setIsLoading(false);
    }
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent className="max-w-2xl">
        <AlertDialogHeader>
          <AlertDialogTitle className="text-xl font-bold">
            Load Example Pipeline
          </AlertDialogTitle>
          <AlertDialogDescription className="space-y-4">
            <div className="mt-4 space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-foreground">
                  {selectedTutorial.title}
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  {selectedTutorial.description}
                </p>
              </div>

              <div className="rounded-lg border border-border bg-muted/50 p-4 space-y-3">
                <div>
                  <h4 className="text-sm font-medium text-foreground">
                    Dataset
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    {selectedTutorial.datasetDescription}
                  </p>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-foreground">
                    Pipeline Operations
                  </h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    {selectedTutorial.operations.map((op, index) => (
                      <li key={index}>{op}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-4">
                <p className="text-sm text-yellow-800">
                  Loading this example will replace your current pipeline
                  configuration.
                </p>
              </div>
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isLoading} className="border-border">
            Cancel
          </AlertDialogCancel>
          <Button
            onClick={loadTutorial}
            disabled={isLoading}
            className={`${
              isLoading ? "opacity-50 cursor-not-allowed" : ""
            } bg-primary`}
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading Example...
              </>
            ) : (
              "Load Example"
            )}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

export { TUTORIALS };
