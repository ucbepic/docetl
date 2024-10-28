import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

const pipelinePrompts = [
  {
    name: "Extract Themes and Viewpoints",
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
    name: "Synthesized Resolve",
    prompt: `Compare the following two debate themes:

[Entity 1]:
{{ input1.theme }}

[Entity 2]:
{{ input2.theme }}

Are these themes likely referring to the same concept? Consider the following attributes:
- The core subject matter being discussed
- The context in which the theme is presented
- The viewpoints of the candidates associated with each theme

Respond with "True" if they are likely the same theme, or "False" if they are likely different themes.`,
  },
  {
    name: "Summarize Theme Evolution",
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

const PipelinePrompts = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Pipeline Operation Prompts</CardTitle>
      </CardHeader>
      <CardContent>
        <Accordion type="single" collapsible>
          {pipelinePrompts.map((prompt, index) => (
            <AccordionItem key={index} value={`item-${index}`}>
              <AccordionTrigger>{prompt.name}</AccordionTrigger>
              <AccordionContent>
                <pre className="text-sm bg-gray-100 p-4 rounded-md whitespace-pre-wrap">
                  {prompt.prompt}
                </pre>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </CardContent>
    </Card>
  );
};

export default PipelinePrompts;
