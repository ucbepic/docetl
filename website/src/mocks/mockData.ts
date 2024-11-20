import { File, Operation } from "@/app/types";
import path from "path";

export const mockFiles = [];

export const initialOperations: Operation[] = [
  // {
  //   id: "1",
  //   llmType: "LLM",
  //   type: "map",
  //   name: "extract_funny_quotes",
  //   prompt:
  //     "list the funniest quotes in this presidential debate, {{ input.date }}. here is the transcript: \n {{ input.content }}",
  //   output: {
  //     schema: [
  //       {
  //         key: "quote",
  //         type: "list",
  //         subType: { key: "quote", type: "string" },
  //       },
  //     ],
  //   },
  // },
  // {
  //   id: '2',
  //   llmType: 'non-LLM',
  //   type: 'unnest',
  //   name: 'unnest_themes',
  //   otherKwargs: {
  //     unnest_key: 'theme',
  //   }
  // },
  // {
  //   id: '3',
  //   llmType: 'LLM',
  //   type: 'resolve',
  //   name: 'resolve_themes',
  //   otherKwargs: {
  //     comparison_prompt: 'Are {{ input1.theme }} and {{ input2.theme }} very related?',
  //     resolution_prompt: 'What is a canonical name for the theme? Canonicalize the following themes: {% for input in inputs %} {{ input.theme }} {% endfor %}',
  //   },
  //   output: {
  //     schema: [
  //       { key: 'theme', type: 'string' }
  //     ]
  //   }
  // },
  // {
  //   id: '4',
  //   llmType: 'LLM',
  //   type: 'reduce',
  //   name: 'reduce_themes',
  //   otherKwargs: {
  //     associative: true,
  //     reduce_key: ['year'],
  //   },
  //   prompt: 'summarize the themes discussed in the debate. here are the transcripts: \n {% for input in inputs %}{{ input.content }}\n{% endfor %}',
  //   output: {
  //     schema: [
  //       { key: 'summary', type: 'string' }
  //     ]
  //   }
  // }
];

export const mockSampleSize = 5;
export const mockPipelineName = "Untitled_Analysis";
