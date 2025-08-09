import React, { Suspense } from "react";
import type { Metadata } from "next";
import dynamic from "next/dynamic";
import { Info, Download, FileJson, Loader2, ArrowLeft, Scroll } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import Link from "next/link";

// Dynamic import for the prompts explorer component
const SystemPromptsExplorer = dynamic(
  () => import("@/components/showcase/system-prompts-explorer"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center p-8 text-gray-600">
        <Loader2 className="h-6 w-6 animate-spin mr-2" />
        Loading Prompts Explorer...
      </div>
    ),
  }
);

export const metadata: Metadata = {
  title: "AI System Prompt Engineering Patterns | DocETL",
  description:
    "Demo of DocETL analyzing leaked system prompts from popular AI assistants to uncover common prompt-engineering strategies.",
  keywords: [
    "AI prompt engineering",
    "system prompts analysis",
    "LLM data analysis",
    "prompt strategy",
  ],
  openGraph: {
    title: "AI System Prompt Engineering Patterns | DocETL",
    description:
      "Interactive demo exploring common strategies in system prompts across ChatGPT, Claude and more using DocETL.",
    url: "https://www.docetl.org/showcase/ai-system-prompts-analysis",
    type: "website",
    images: [{ url: "/docetl-favicon-color.png" }],
  },
};

export default function AiSystemPromptsAnalysisPage() {
  return (
    <main className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Header with Logo */}
      <div className="text-center mb-8">
        <Link href="/" className="inline-block">
          <div className="flex items-center justify-center mb-2">
            <Scroll
              className="w-10 h-10 sm:w-12 sm:h-12 mr-2 text-primary"
              strokeWidth={1.5}
            />
            <span className="logo-text text-2xl sm:text-3xl">DocETL</span>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <div className="mb-6 flex items-center gap-3">
        <Button variant="outline" size="sm" asChild>
          <Link href="/showcase">
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Showcase
          </Link>
        </Button>
        <Button size="sm" asChild>
          <Link
            href="https://github.com/ucbepic/docetl"
            target="_blank"
            rel="noopener noreferrer"
          >
            ⭐ Star on GitHub
          </Link>
        </Button>
      </div>

      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2 text-gray-900">
          Prompt Engineering Strategies from Popular AI Assistants
        </h1>
        <p className="text-gray-600">
          Analyzing system prompts from ChatGPT, Claude, and other AI systems to extract common strategies
        </p>
      </div>

      <Alert className="mb-6 bg-blue-50 border-blue-200">
        <Info className="h-4 w-4 text-blue-600" />
        <AlertTitle className="text-blue-900">About This Analysis</AlertTitle>
        <AlertDescription className="text-blue-800">
          <div className="space-y-3 mt-2">
            <p>
              Using the{" "}
              <a
                href="https://github.com/dontriskit/awesome-ai-system-prompts"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-700 underline hover:text-blue-900"
              >
                awesome-ai-system-prompts
              </a>{" "}
              repository as source data, DocETL analyzed leaked system prompts from major AI systems 
              to identify recurring patterns and approaches used across different assistants.
            </p>
            
            <p>
              The pipeline extracts <strong>common strategies</strong> (recurring patterns across systems), 
              <strong>implementation examples</strong> (how specific AI systems apply these strategies), and 
              <strong>strategy summaries</strong> (concise explanations and considerations). This analysis helps 
              prompt engineers and AI developers identify best practices for designing effective system prompts.
            </p>

            <p className="text-sm">
              <strong>Processing details:</strong> 19 AI systems analyzed • gpt-4o-mini • $0.18 total cost
            </p>

            <div className="p-3 border border-yellow-300 bg-yellow-50 rounded-md mt-3">
              <p className="text-sm font-medium text-yellow-800">
                <strong>Note:</strong> Strategies are extracted through automated analysis and may not capture 
                all nuances. Use these insights as a starting point for your prompt engineering work.
              </p>
            </div>
            
            <div className="flex flex-wrap gap-3 mt-4">
              <a
                href="https://docetlcloudbank.blob.core.windows.net/demos/prompts.json"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <FileJson className="h-4 w-4 mr-2" />
                  Download Pipeline Input
                </Button>
              </a>
              <a
                href="https://docetlcloudbank.blob.core.windows.net/demos/analyzed_strategies.json"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <Download className="h-4 w-4 mr-2" />
                  Download Pipeline Output
                </Button>
              </a>
              <a
                href="/demos/prompts_pipeline.yaml"
                download
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <FileJson className="h-4 w-4 mr-2" />
                  Download Pipeline YAML
                </Button>
              </a>
            </div>
          </div>
        </AlertDescription>
      </Alert>

      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-900">
          Explore System Prompt Strategies
        </h3>
        <Suspense
          fallback={
            <div className="flex items-center justify-center p-8 text-gray-600">
              <Loader2 className="h-6 w-6 animate-spin mr-2" />
              Loading Prompts Explorer...
            </div>
          }
        >
          <SystemPromptsExplorer />
        </Suspense>
      </div>
    </main>
  );
}