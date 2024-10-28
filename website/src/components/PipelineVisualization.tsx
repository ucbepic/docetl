import React, { useState, useCallback, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import {
  Sparkles,
  Cog,
  ArrowRight,
  ArrowDown,
  Play,
  ChevronRight,
  ChevronLeft,
  Zap,
  Pencil,
} from "lucide-react";
// import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  exampleDebate,
  reduceInput,
  reduceOutput,
} from "@/components/DebateContent";

const ignoreCols = ["url", "id", "year"];
const colOrder = [
  "date",
  "title",
  "content",
  "themes",
  "theme",
  "viewpoints",
  "report",
];

const opTypeToDocLink = {
  map: "https://ucbepic.github.io/docetl/operators/map/",
  reduce: "https://ucbepic.github.io/docetl/operators/reduce/",
  unnest: "https://ucbepic.github.io/docetl/operators/unnest/",
  resolve: "https://ucbepic.github.io/docetl/operators/resolve/",
};

export const pipelineSteps = [
  {
    synthesized: false,
    filename: "/debate_intermediates/extract_themes_and_viewpoints.json",
    name: "Map Each Transcript to Themes",
    type: "map",
    color: "blue",
    isLLM: true,
    outputDescription: "Covers 339 distinct themes",
    outputCols: ["themes"],
    exampleVars: {
      "input.title": exampleDebate["title"],
      "input.date": exampleDebate["date"],
      "input.content": exampleDebate["content"],
    },
    exampleOutput: exampleDebate["themes"],
    prompt: `Analyze the following debate transcript for {{ input.title }} on {{ input.date }}:

{{ input.content }}

Extract the main themes discussed in this debate and the viewpoints of the candidates on these themes.
Return a list of themes and corresponding viewpoints in the following format:
[
  {
    "theme": "Theme 1",
    "viewpoints": "Candidate A's viewpoint... Candidate B's viewpoint..."
  },
  {
    "theme": "Theme 2",
    "viewpoints": "Candidate A's viewpoint... Candidate B's viewpoint..."
  },
  ...
]`,
  },
  {
    synthesized: false,
    filename: "/debate_intermediates/unnest_themes.json",
    name: "Unnest Themes",
    type: "unnest",
    outputCols: ["theme", "viewpoints"],
    color: "orange",
    isLLM: false,
    exampleVars: { transcript: exampleDebate["themes"] },
    exampleOutput: exampleDebate["unnested_themes"],
    description:
      "This step flattens the nested theme structure. The input to this step consists of one document per debate transcript, but the output of this step will consist of a document per theme identified.",
  },
  {
    synthesized: true,
    filename: "/debate_intermediates/synthesized_resolve_0.json",
    name: "Deduplicate and Merge Themes",
    type: "resolve",
    color: "yellow",
    isLLM: true,
    exampleVars: {
      "input1.theme": "Medicare and Social Security",
      "input1.viewpoints": `Vice President Biden: The cuts proposed would harm millions; we have extended Medicare's
life by making smart cuts, not vouchers. Congressman Ryan: Medicare and Social Security are going bankrupt; we must
reform them to protect future generations.`,
      "input2.theme": "Social Security and Medicare",
      "input2.viewpoints": `Dole speaks to the need for reform and preserving benefits, stating, "We want to make 
certain we protect those in the pipeline just as we did back in 1983." Clinton assures the sustainability of these 
programs, mentioning, "Social Security is stable until... at least the third decade of the next century."`,
    },
    exampleOutput: [
      { equal: "True", "(resolved) theme": "Social Security and Medicare" },
    ],
    outputDescription: "Reduces number of themes by 55%",
    outputCols: ["theme"],
    prompt: `Compare the following two debate themes:

[Entity 1]:
Theme: {{ input1.theme }}
Viewpoints: {{ input1.viewpoints }}

[Entity 2]:
Theme: {{ input2.theme }}
Viewpoints: {{ input2.viewpoints }}

Are these themes likely referring to the same concept? Consider the following attributes:
- The core subject matter being discussed
- The context in which the theme is presented
- The viewpoints of the candidates associated with each theme

Respond with "True" if they are likely the same theme, or "False" if they are likely different themes.`,
  },
  {
    synthesized: false,
    filename: "/debate_intermediates/summarize_theme_evolution.json",
    name: "Summarize Themes Over Time",
    type: "reduce",
    color: "green",
    outputCols: ["report"],
    exampleVars: { "inputs[0].theme": "Abortion Rights", inputs: reduceInput },
    exampleOutput: [reduceOutput],
    outputDescription: "Generates 152 reports averaging 730 words each",
    isLLM: true,
    prompt: `Analyze the following viewpoints on the theme "{{ inputs[0].theme }}" from various debates over the years:

{% for item in inputs %}
Year: {{ item.year }}
Date: {{ item.date }}
Title: {{ item.title }}
Viewpoints: {{ item.viewpoints }}

{% endfor %}

Generate a comprehensive summary of how Democratic and Republican viewpoints on this theme have evolved through the years. Include supporting quotes from the debates to illustrate key points or shifts in perspective.

Your summary should:
1. Identify all major trends or shifts in each party's stance over time
2. Highlight any significant agreements or disagreements between the parties
3. Note any external events or factors that may have influenced changes in viewpoints
4. Use specific quotes to support your analysis
5. The title should contain the start and end years of the analysis

Format your response as a well-structured report.`,
  },
];

const PipelineVisualization = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [streamedContent, setStreamedContent] = useState("");
  const [showNextButton, setShowNextButton] = useState(false);
  const [showPreviousButton, setShowPreviousButton] = useState(false);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [showOutputs, setShowOutputs] = useState(false);
  const [outputs, setOutputs] = useState<any[]>([]);
  const [isLoadingOutputs, setIsLoadingOutputs] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [isStreaming, setIsStreaming] = useState(false);
  const rowsPerPage = 5;

  const streamingRef = useRef<boolean>(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const streamContent = useCallback(async (content: string) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    setIsStreaming(true);
    streamingRef.current = true;
    setStreamedContent("");

    try {
      for (let i = 0; i < content.length; i++) {
        if (signal.aborted) {
          break;
        }
        setStreamedContent((prev) => prev + content[i]);
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
    } catch (error: any) {
      if (error.name === "AbortError") {
        console.log("Streaming aborted");
      } else {
        console.error("Streaming error:", error);
      }
    } finally {
      setIsStreaming(false);
      streamingRef.current = false;
    }
  }, []);

  const runPipeline = () => {
    setIsRunning(true);
    setCompletedSteps([]);
    handleStepClick(0);
  };

  const handleStepClick = useCallback(
    async (index: number) => {
      if (streamingRef.current) {
        if (abortControllerRef.current) {
          abortControllerRef.current.abort();
        }
        await new Promise((resolve) => setTimeout(resolve, 50)); // Small delay to ensure abortion is complete
      }

      setCurrentStep(index);
      setShowNextButton(false);
      setShowPreviousButton(false);
      setShowOutputs(false);
      setOutputs([]);
      setCurrentPage(1);
      await streamContent(
        pipelineSteps[index].prompt || pipelineSteps[index].description || "",
      );
      setShowNextButton(true);
      setShowPreviousButton(index > 0);
      if (!completedSteps.includes(index)) {
        setCompletedSteps((prev) => [...prev, index]);
      }
    },
    [completedSteps, streamContent],
  );

  const handleNextStep = useCallback(async () => {
    const nextStep = currentStep + 1;
    if (nextStep < pipelineSteps.length) {
      await handleStepClick(nextStep);
    } else {
      setIsRunning(false);
      setCurrentStep(-1);
      setStreamedContent("");
      setShowNextButton(false);
      setShowPreviousButton(false);
    }
  }, [currentStep, handleStepClick]);

  const handlePreviousStep = useCallback(() => {
    if (currentStep > 0) {
      handleStepClick(currentStep - 1);
    }
  }, [currentStep, handleStepClick]);

  const fetchOutputs = useCallback(async () => {
    setIsLoadingOutputs(true);
    try {
      const response = await fetch(pipelineSteps[currentStep].filename);
      let data = await response.json();
      if (data.length > 0 && "date" in data[0]) {
        data.sort(
          (a: { date: string }, b: { date: string }) =>
            new Date(b.date).getTime() - new Date(a.date).getTime(),
        );
      }
      // Order the columns based on colOrder
      const orderedData = data.map((item: { [x: string]: any }) => {
        const orderedItem: { [key: string]: any } = {};
        colOrder.forEach((col) => {
          if (col in item) {
            orderedItem[col] = item[col];
          }
        });
        // Add any remaining columns that are not in colOrder
        Object.keys(item).forEach((key) => {
          if (!(key in orderedItem)) {
            orderedItem[key] = item[key];
          }
        });
        return orderedItem;
      });
      data = orderedData;
      setOutputs(data);
    } catch (error) {
      console.error("Error fetching outputs:", error);
    } finally {
      setIsLoadingOutputs(false);
    }
  }, [currentStep]);

  useEffect(() => {
    if (showOutputs) {
      fetchOutputs();
    }
  }, [showOutputs, fetchOutputs]);

  useEffect(() => {
    if (!isRunning) {
      setStreamedContent("");
      setShowNextButton(false);
      setShowPreviousButton(false);
      setShowOutputs(false);
      setOutputs([]);
      setCurrentPage(1);
    }
  }, [isRunning]);

  const renderTableCell = (value: any) => {
    const stringValue =
      typeof value === "object" && value !== null
        ? JSON.stringify(value)
        : String(value);
    const truncatedValue =
      stringValue.substring(0, 100) + (stringValue.length > 100 ? "..." : "");

    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="cursor-help">{truncatedValue}</span>
          </TooltipTrigger>
          <TooltipContent>
            <p className="max-w-md overflow-auto max-h-60 whitespace-pre-wrap">
              {stringValue}
            </p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  };

  const totalPages = Math.ceil(outputs.length / rowsPerPage);
  const paginatedOutputs = outputs.slice(
    (currentPage - 1) * rowsPerPage,
    currentPage * rowsPerPage,
  );

  const renderExampleVar = (key: string, value: any) => (
    <TooltipProvider key={key}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="border-b-2 border-dotted border-primary cursor-help">
            {key}
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <pre className="max-w-md overflow-auto max-h-60 whitespace-pre-wrap">
            {value}
          </pre>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );

  const renderPromptWithExampleVars = (
    prompt: string,
    exampleVars: { [key: string]: any },
  ) => {
    return (
      <span>
        {prompt.split(/(\s+)/).map((part) => {
          const key = Object.keys(exampleVars).find((k) => part.includes(k));
          if (key) {
            return renderExampleVar(key, exampleVars[key]);
          }
          return part;
        })}
      </span>
    );
  };

  const renderStepIcon = (step: any) => {
    if (step.synthesized) {
      return <Zap className="w-3 h-3 text-yellow-500 inline" />;
    } else {
      return <Pencil className="w-3 h-3 text-muted-foreground inline" />;
    }
  };

  const renderExampleOutput = (output: any) => (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="ml-2 text-sm text-muted-foreground border-b-2 border-dotted border-primary cursor-help">
            (Example Output)
          </span>
        </TooltipTrigger>
        <TooltipContent side="right" align="start" className="max-w-md">
          <pre className="text-sm whitespace-pre-wrap overflow-auto max-h-60">
            {JSON.stringify(output, null, 2)}
          </pre>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start mb-2">
        <div className="text-left">
          <h2 className="text-2xl font-bold mb-2">
            US Presidential Debate Analysis
          </h2>
          <p className="text-xs font-semibold text-primary italic">
            Powered by DocETL: a declarative system for LLM-powered data
            processing pipelines. Define pipelines in YAML, optimize
            automatically, and seamlessly integrate LLM and non-LLM operations.
          </p>
        </div>
        <Button
          onClick={runPipeline}
          disabled={isRunning}
          className="bg-primary text-primary-foreground hover:bg-primary/90"
        >
          {isRunning ? (
            <>
              <Play className="mr-2 h-4 w-4 animate-spin" />
              Running
            </>
          ) : (
            <>
              <Play className="mr-2 h-4 w-4" />
              Run
            </>
          )}
        </Button>
      </div>

      <p className="text-sm text-muted-foreground text-left">
        This pipeline analyzes themes in US presidential debates dating back to
        1960, summarizing the evolution of the viewpoints of Democrats and
        Republicans for each theme. It cost us $0.29 to run (and $0.86 to
        optimize).
      </p>
      <p className="text-sm text-muted-foreground text-left">
        The combined debate transcripts span 738,094 words, making it tricky to
        analyze in a single prompt. For example, when given the entire dataset
        in a single prompt, Gemini-1.5-Pro-002 (released September 24, 2024)
        only reports on the evolution of five themes (throughout all the
        documents).
      </p>

      <div className="flex flex-col md:flex-row justify-center items-center gap-2 mb-8 overflow-x-auto py-4">
        {pipelineSteps.map((step, index) => (
          <React.Fragment key={step.name}>
            <motion.div
              className={`p-2 rounded-lg shadow-sm cursor-pointer flex-shrink-0 w-full md:w-44
                ${
                  index === currentStep
                    ? "bg-primary/10 border-2 border-primary"
                    : completedSteps.includes(index)
                      ? "bg-secondary/10 border-2 border-primary"
                      : "bg-background border border-border"
                }
                ${!step.isLLM ? "scale-95" : ""}`}
              animate={
                isRunning && index === currentStep
                  ? { scale: [1, 1.05, 1] }
                  : {}
              }
              transition={{
                duration: 1,
                repeat: Infinity,
                repeatType: "reverse",
              }}
              onClick={() => handleStepClick(index)}
            >
              <div className="flex flex-row md:flex-col items-center justify-start md:justify-center text-left md:text-center h-full">
                <div className="mr-2 md:mr-0 md:mb-1">
                  {step.isLLM ? (
                    <Sparkles
                      className={`w-4 h-4 ${completedSteps.includes(index) || index === currentStep ? "text-primary" : "text-secondary"}`}
                    />
                  ) : (
                    <Cog
                      className={`w-4 h-4 ${completedSteps.includes(index) || index === currentStep ? "text-primary" : "text-secondary"}`}
                    />
                  )}
                </div>
                <div>
                  <p className="text-xs font-medium">{step.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {step.isLLM ? "(LLM)" : "(Non-LLM)"}
                    {/* {renderStepIcon(step)} */}
                  </p>
                  {step.isLLM && step.outputDescription && (
                    <p className="text-xs italic text-muted-foreground mt-1 line-clamp-2">
                      {step.outputDescription}
                    </p>
                  )}
                </div>
              </div>
            </motion.div>
            {index < pipelineSteps.length - 1 && (
              <>
                <ArrowRight
                  size={16}
                  className="text-muted-foreground hidden md:block flex-shrink-0"
                />
                <ArrowDown
                  size={16}
                  className="text-muted-foreground md:hidden flex-shrink-0"
                />
              </>
            )}
          </React.Fragment>
        ))}
      </div>

      {currentStep !== -1 && (
        <div className="border-t border-border pt-6 text-left">
          <h3 className="text-xl font-semibold mb-2">
            {pipelineSteps[currentStep].name}
          </h3>
          <p className="text-sm text-muted-foreground mb-4">
            Operator type:{" "}
            <a
              href={
                opTypeToDocLink[
                  pipelineSteps[currentStep]
                    .type as keyof typeof opTypeToDocLink
                ]
              }
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              {pipelineSteps[currentStep].type}
            </a>
            {pipelineSteps[currentStep].synthesized
              ? " (Synthesized by DocETL's optimizer"
              : " (User-defined"}{" "}
            {renderStepIcon(pipelineSteps[currentStep])})
          </p>

          <div className="mb-6">
            <h4 className="text-sm font-semibold mb-2">
              {pipelineSteps[currentStep].isLLM
                ? "LLM Prompt:"
                : "Description:"}
              {pipelineSteps[currentStep].exampleOutput &&
                renderExampleOutput(pipelineSteps[currentStep].exampleOutput)}
            </h4>
            <pre className="text-sm whitespace-pre-wrap overflow-x-auto bg-muted p-4 rounded-md">
              {pipelineSteps[currentStep].exampleVars
                ? renderPromptWithExampleVars(
                    streamedContent,
                    pipelineSteps[currentStep].exampleVars,
                  )
                : streamedContent}
            </pre>
          </div>

          {!isStreaming && (
            <div className="flex flex-col sm:flex-row justify-between items-center mb-6 space-y-2 sm:space-y-0">
              <div className="flex items-center space-x-2">
                {showPreviousButton && (
                  <Button
                    onClick={handlePreviousStep}
                    variant="outline"
                    size="sm"
                  >
                    <ChevronLeft className="mr-1 h-4 w-4" />
                    Previous Step
                  </Button>
                )}
                {showNextButton && (
                  <Button onClick={handleNextStep} variant="outline" size="sm">
                    {currentStep === pipelineSteps.length - 1
                      ? "Finish"
                      : "Next Step"}
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </Button>
                )}
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="show-outputs"
                  checked={showOutputs}
                  onCheckedChange={setShowOutputs}
                />
                <label htmlFor="show-outputs" className="text-sm">
                  Show All Step Outputs
                </label>
              </div>
            </div>
          )}

          {showOutputs && !isStreaming && (
            <div>
              {isLoadingOutputs ? (
                <p className="text-center text-muted-foreground">
                  Loading outputs...
                </p>
              ) : outputs.length > 0 ? (
                <>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {Object.keys(outputs[0])
                          .filter((key) => !ignoreCols.includes(key))
                          .map((key) => (
                            <TableHead key={key}>
                              {pipelineSteps[currentStep].outputCols.includes(
                                key,
                              ) ? (
                                <strong>{key}</strong>
                              ) : (
                                key
                              )}
                            </TableHead>
                          ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {paginatedOutputs.map((row, index) => (
                        <TableRow key={index}>
                          {Object.entries(row)
                            .filter(([key]) => !ignoreCols.includes(key))
                            .map(([key, value], cellIndex) => (
                              <TableCell key={cellIndex}>
                                {pipelineSteps[currentStep].outputCols.includes(
                                  key,
                                ) ? (
                                  <strong>{renderTableCell(value)}</strong>
                                ) : (
                                  renderTableCell(value)
                                )}
                              </TableCell>
                            ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  <div className="flex justify-between items-center mt-4">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        setCurrentPage((prev) => Math.max(prev - 1, 1))
                      }
                      disabled={currentPage === 1}
                    >
                      <ChevronLeft className="mr-2 h-4 w-4" /> Previous
                    </Button>
                    <span className="text-sm text-muted-foreground">
                      Page {currentPage} of {totalPages}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        setCurrentPage((prev) => Math.min(prev + 1, totalPages))
                      }
                      disabled={currentPage === totalPages}
                    >
                      Next <ChevronRight className="ml-2 h-4 w-4" />
                    </Button>
                  </div>
                </>
              ) : (
                <p className="text-center text-muted-foreground">
                  No outputs available.
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PipelineVisualization;
