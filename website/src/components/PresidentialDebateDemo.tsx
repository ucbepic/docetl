import React, { useEffect, useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Copy, ChevronDown } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import PipelineVisualization from "./PipelineVisualization";
import ReactMarkdown from "react-markdown";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useToast } from "@/hooks/use-toast";
import {
  Tooltip,
  TooltipProvider,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const pipelineCode = `datasets:
  debates:
    type: file
    path: "data.json"

default_model: gpt-4o-mini

operations:
  - name: extract_themes_and_viewpoints # Blue
    type: map
    output:
      schema:
        themes: "list[{theme: str, viewpoints: str}]"
    prompt: |
      Analyze the following debate transcript for {{ input.title }} on {{ input.date }}:

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
      ]
    validate:
      - 'len(output["themes"]) > 0'

  - name: unnest_themes
    type: unnest
    unnest_key: themes
    recursive: true

  - name: synthesized_resolve_0
    type: resolve
    blocking_keys:
      - theme
    blocking_threshold: 0.6465
    compare_batch_size: 1000
    comparison_model: gpt-4o-mini
    comparison_prompt: |
      Compare the following two topics discussed in a presidental debate:

      [Theme 1]:
      Theme: {{ input1.theme }}
      Viewpoints: {{ input1.viewpoints }}

      [Theme 2]:
      Theme: {{ input2.theme }}
      Viewpoints: {{ input2.viewpoints }}

      Should these topics be merged into one topic, or are they different topics? Respond with "True" if they can be merged, or "False" if they are different topics.
    embedding_model: text-embedding-3-small
    output:
      schema:
        theme: string
    resolution_model: gpt-4o-mini
    resolution_prompt: |
      Analyze the following duplicate themes:

      {% for key in inputs %}
      Entry {{ loop.index }}:
      {{ key.theme }}

      {% endfor %}

      Create a single, consolidated key that combines the information from all duplicate entries. When merging, follow these guidelines:
      1. Prioritize the most comprehensive and detailed viewpoint available among the duplicates. If multiple entries discuss the same theme with varying details, select the entry that includes the most information.
      2. Ensure clarity and coherence in the merged key; if key terms or phrases are duplicated, synthesize them into a single statement or a cohesive description that accurately represents the theme.

      Ensure that the merged key conforms to the following schema:
      {
        "theme": "string"
      }

      Return the consolidated key as a single JSON object.

  - name: summarize_theme_evolution
    type: reduce
    reduce_key: theme
    optimize: true
    output:
      schema:
        theme: str
        report: str
    prompt: |
      Analyze the following viewpoints on the theme "{{ inputs[0].theme }}" from various debates over the years:

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

      Format your response as a well-structured report.
    gleaning:
      num_rounds: 1
      validation_prompt: |
        1. Does the output adequately summarize the evolution of viewpoints on the theme based on the provided debate texts? Are all critical shifts and trends mentioned?
        2. Are there any crucial quotes or data points missing from the output that were present in the debate transcripts that could reinforce the analysis?
        3. Is the output well-structured and easy to follow, following any
        formatting guidelines specified in the prompt, such as using headings for sections or maintaining a coherent narrative flow?

pipeline:
  steps:
    - name: debate_analysis
      input: debates
      operations:
        - extract_themes_and_viewpoints
        - unnest_themes
        - synthesized_resolve_0
        - summarize_theme_evolution

  output:
    type: file
    path: "theme_evolution_analysis.json"
    intermediate_dir: "checkpoints"`;

const PresidentialDebateDemo = () => {
  const [reports, setReports] = useState<
    Array<{ theme: string; report: string }>
  >([]);
  const [selectedTheme, setSelectedTheme] = useState<string>("");

  const [inputData, setInputData] = useState<
    Array<{
      year: string;
      date: string;
      title: string;
      url: string;
      content: string;
    }>
  >([]);
  const [selectedDebate, setSelectedDebate] = useState<string>("");
  const [geminiResult, setGeminiResult] = useState<string>("");
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState<string>("demo-input");
  const [isCollapsibleOpen, setIsCollapsibleOpen] = useState(false);

  useEffect(() => {
    // Update the URL hash when the active tab changes, but only when the collapsible is open
    if (isCollapsibleOpen) {
      window.location.hash = activeTab;
    }
  }, [activeTab, isCollapsibleOpen]);

  useEffect(() => {
    // Set the active tab based on the URL hash on initial load
    const hash = window.location.hash.replace("#", "");
    if (
      hash &&
      [
        "demo-input",
        "demo-docetl-output",
        "demo-gemini-output",
        "demo-code",
      ].includes(hash)
    ) {
      setActiveTab(hash);
      setIsCollapsibleOpen(true);
    }
  }, []);

  useEffect(() => {
    // Fetch the JSON file
    fetch("/theme_evolution_analysis.json")
      .then((response) => response.json())
      .then((data) => {
        setReports(data);
        if (data.length > 0) {
          setSelectedTheme(data[0].theme);
        }
      })
      .catch((error) => console.error("Error loading theme reports:", error));

    // Fetch the input JSON file
    fetch("/debate_transcripts.json")
      .then((response) => response.json())
      .then((data) => {
        const transformedData = data
          .map(
            (item: {
              year: string;
              date: string;
              title: string;
              url: string;
              content: string;
            }) => ({
              ...item,
              title: `${item.title} (${item.year})`,
            }),
          )
          .sort(
            (a: { date: string }, b: { date: string }) =>
              new Date(b.date).getTime() - new Date(a.date).getTime(),
          );
        setInputData(transformedData);
        if (transformedData.length > 0) {
          setSelectedDebate(transformedData[0].title);
        }
      })
      .catch((error) => console.error("Error loading input data:", error));

    fetch("/debate_gemini_result.txt")
      .then((response) => response.text())
      .then((data) => {
        setGeminiResult(data);
      })
      .catch((error) => console.error("Error loading Gemini result:", error));
  }, []);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(pipelineCode);
    toast({
      description: "The pipeline code has been copied to your clipboard.",
    });
  };

  return (
    <Card className="max-w-8xl mx-auto bg-white shadow-md">
      <CardContent className="p-6 space-y-6">
        <PipelineVisualization />
        <p className="text-sm text-muted-foreground text-left mb-2">
          DocETL generates comprehensive reports for 152 distinct themes, each
          analyzing the evolution of Democratic and Republican viewpoints over
          time. You can explore the reports by selecting a theme from the
          dropdown menu.
        </p>

        <Collapsible
          open={isCollapsibleOpen}
          onOpenChange={setIsCollapsibleOpen}
        >
          <CollapsibleTrigger asChild>
            <Button variant="outline" className="w-full justify-between">
              <span>See Code, Transcripts, and Outputs</span>
              <ChevronDown
                className={`h-4 w-4 ml-2 inline-block ${isCollapsibleOpen ? "transform rotate-180" : ""}`}
              />
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-4">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList>
                <TabsTrigger value="demo-code">Code</TabsTrigger>
                <TabsTrigger value="demo-input">Input</TabsTrigger>
                <TabsTrigger value="demo-docetl-output">
                  DocETL Output
                </TabsTrigger>
                <TabsTrigger value="demo-gemini-output">
                  Gemini Output
                </TabsTrigger>
              </TabsList>
              <TabsContent value="demo-input" className="mt-4">
                <h3 className="text-lg font-semibold mb-4">
                  Debate Transcripts
                </h3>
                <Select
                  value={selectedDebate}
                  onValueChange={setSelectedDebate}
                >
                  <SelectTrigger className="w-full mb-4">
                    <SelectValue placeholder="Select a debate" />
                  </SelectTrigger>
                  <SelectContent>
                    {inputData.map((item) => (
                      <SelectItem key={item.title} value={item.title}>
                        {item.title}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <div className="h-[400px] overflow-y-auto bg-muted p-4 rounded-md text-left">
                  <pre className="text-sm whitespace-pre-wrap">
                    <code>
                      {
                        inputData.find((item) => item.title === selectedDebate)
                          ?.content
                      }
                    </code>
                  </pre>
                </div>
              </TabsContent>
              <TabsContent value="demo-docetl-output" className="mt-4">
                <h3 className="text-lg font-semibold mb-4">
                  DocETL-Generated Reports
                </h3>
                <Select value={selectedTheme} onValueChange={setSelectedTheme}>
                  <SelectTrigger className="w-full mb-4">
                    <SelectValue placeholder="Select a theme" />
                  </SelectTrigger>
                  <SelectContent>
                    {reports.map((item) => (
                      <SelectItem key={item.theme} value={item.theme}>
                        {item.theme}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <div className="h-[400px] overflow-y-auto bg-muted p-4 rounded-md text-left">
                  <ReactMarkdown
                    className="prose prose-sm max-w-none"
                    components={{
                      h1: ({ ...props }) => (
                        <h1
                          className="text-xl font-bold mt-4 mb-2"
                          {...props}
                        />
                      ),
                      h2: ({ ...props }) => (
                        <h2
                          className="text-lg font-semibold mt-3 mb-2"
                          {...props}
                        />
                      ),
                      h3: ({ ...props }) => (
                        <h3
                          className="text-base font-medium mt-2 mb-1"
                          {...props}
                        />
                      ),
                      h4: ({ ...props }) => (
                        <h4
                          className="text-sm font-medium mt-2 mb-1"
                          {...props}
                        />
                      ),
                      h5: ({ ...props }) => (
                        <h5
                          className="text-xs font-medium mt-1 mb-1"
                          {...props}
                        />
                      ),
                      h6: ({ ...props }) => (
                        <h6
                          className="text-xs font-medium mt-1 mb-1"
                          {...props}
                        />
                      ),
                    }}
                  >
                    {reports.find((item) => item.theme === selectedTheme)
                      ?.report || ""}
                  </ReactMarkdown>
                </div>
              </TabsContent>
              <TabsContent value="demo-gemini-output" className="mt-4">
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  Gemini-1.5-Pro-002 Result ({geminiResult.split(/\s+/).length}{" "}
                  words)
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild className="cursor-help">
                        <span className="ml-2 text-sm text-muted-foreground underline cursor-help">
                          Hover to see prompt
                        </span>
                      </TooltipTrigger>
                      <TooltipContent className="p-4">
                        <pre className="text-sm whitespace-pre-wrap text-left">
                          {`Analyze the following presidential debate transcripts:

{% for debate in debates %}
Title: {{ debate.title }}
Date: {{ debate.date }}
Content: {{ debate.content }}

{% endfor %}

For each debate:
1. Extract the main themes discussed and the viewpoints of the candidates on these themes.

Then, for each unique theme across all debates:
2. Generate a comprehensive summary of how Democratic and Republican viewpoints on this theme have evolved through the years.

For each theme's summary:
- Identify all major trends or shifts in each party's stance over time
- Highlight any significant agreements or disagreements between the parties
- Note any external events or factors that may have influenced changes in viewpoints
- Use specific quotes from the debates to support your analysis
- Include a title containing the start and end years of the analysis

Ensure each summary is well-structured and comprehensive.

Provide your analysis as a detailed text response, organizing the information by themes and their evolution over time.`}
                        </pre>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </h3>
                <div className="h-[400px] overflow-y-auto bg-muted p-4 rounded-md text-left">
                  <ReactMarkdown
                    className="prose prose-sm max-w-none"
                    components={{
                      h1: ({ ...props }) => (
                        <h1
                          className="text-xl font-bold mt-4 mb-2"
                          {...props}
                        />
                      ),
                      h2: ({ ...props }) => (
                        <h2
                          className="text-lg font-semibold mt-3 mb-2"
                          {...props}
                        />
                      ),
                      h3: ({ ...props }) => (
                        <h3
                          className="text-base font-medium mt-2 mb-1"
                          {...props}
                        />
                      ),
                      h4: ({ ...props }) => (
                        <h4
                          className="text-sm font-medium mt-2 mb-1"
                          {...props}
                        />
                      ),
                      h5: ({ ...props }) => (
                        <h5
                          className="text-xs font-medium mt-1 mb-1"
                          {...props}
                        />
                      ),
                      h6: ({ ...props }) => (
                        <h6
                          className="text-xs font-medium mt-1 mb-1"
                          {...props}
                        />
                      ),
                    }}
                  >
                    {geminiResult}
                  </ReactMarkdown>
                </div>
              </TabsContent>
              <TabsContent value="demo-code" className="mt-4">
                <h3 className="text-lg font-semibold mb-4">Pipeline Code</h3>
                <div className="relative h-[400px] overflow-hidden">
                  <pre className="h-full overflow-y-auto bg-muted p-4 rounded-md text-left">
                    <code className="text-sm whitespace-pre-wrap">
                      {pipelineCode}
                    </code>
                  </pre>
                  <Button
                    className="absolute top-2 right-2"
                    variant="secondary"
                    onClick={copyToClipboard}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </TabsContent>
            </Tabs>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
};

export default PresidentialDebateDemo;
