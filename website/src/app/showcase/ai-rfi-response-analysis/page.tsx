import React, { Suspense } from "react";
import type { Metadata } from "next";
import dynamic from "next/dynamic";
import { Info, Download, FileJson, Loader2, ArrowLeft, Scroll } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import Link from "next/link";

// Simplify the dynamic import completely
const RfiResponseExplorer = dynamic(
  () => import("@/components/showcase/rfi-response-explorer"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center p-8 text-gray-600">
        <Loader2 className="h-6 w-6 animate-spin mr-2" />
        Loading Data Explorer...
      </div>
    ),
  }
);

export const metadata: Metadata = {
  title: "AI Policy RFI Response Analysis | DocETL",
  description:
    "DocETL demo that extracts structured insights from 10,000 public responses to the U.S. AI RFI using LLMs and ranks them by topics.",
  keywords: [
    "AI policy analysis",
    "LLM data extraction",
    "public comment analysis",
    "AI document analytics",
  ],
  openGraph: {
    title: "AI Policy RFI Response Analysis | DocETL",
    description:
      "Interactive demo: see how DocETL uses AI to extract proposals, demographics and more from thousands of policy responses.",
    url: "https://www.docetl.org/showcase/ai-rfi-response-analysis",
    type: "website",
    images: [{ url: "/docetl-favicon-color.png" }],
  },
};

export default function AiRfiResponseAnalysisPage() {
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
          AI RFI Response Analysis
        </h1>
        <p className="text-gray-600">
          Analyzing 10,000+ public responses to the U.S. Government's{" "}
          <a
            href="https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline"
          >
            Request for Information
          </a>{" "}
          on AI Action Plan Development
        </p>
      </div>

      <Alert className="mb-6 bg-blue-50 border-blue-200">
        <Info className="h-4 w-4 text-blue-600" />
        <AlertTitle className="text-blue-900">About This Analysis</AlertTitle>
        <AlertDescription className="text-blue-800">
          <div className="space-y-3 mt-2">
            <p>
              In February 2025, the U.S. Office of Science and Technology Policy requested public input on developing 
              an AI Action Plan as directed by{" "}
              <a
                href="https://www.federalregister.gov/documents/2025/01/31/2025-02172/removing-barriers-to-american-leadership-in-artificial-intelligence"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-700 underline hover:text-blue-900"
              >
                Executive Order 14179
              </a>
              . DocETL processed over 10,000 responses to extract structured insights about policy proposals, 
              demographic patterns, and key concerns from individuals, academia, industry, and other stakeholders.
            </p>
            
            <p>
              The pipeline identifies whether responses contain <strong>concrete proposals</strong>, detects if they're from 
              <strong>notable entities</strong> (well-known individuals or organizations), estimates <strong>age demographics</strong>, 
              and classifies the <strong>primary topic</strong> of concern while generating concise summaries.
            </p>

            <p className="text-sm">
              <strong>Processing details:</strong> 10,068 documents • gpt-4o-mini • $3.01 total cost • 
              Extracted from public PDFs via OCR
            </p>

            <div className="p-3 border border-yellow-300 bg-yellow-50 rounded-md mt-3">
              <p className="text-sm font-medium text-yellow-800">
                <strong>Disclaimer:</strong> This analysis may contain errors from OCR processing and LLM limitations. 
                For illustrative purposes only.
              </p>
            </div>
            
            <div className="flex flex-wrap gap-3 mt-4">
              <a
                href="https://docetlcloudbank.blob.core.windows.net/demos/processed_responses.json"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <FileJson className="h-4 w-4 mr-2" />
                  Download Pipeline Input
                </Button>
              </a>
              <a
                href="https://docetlcloudbank.blob.core.windows.net/demos/summarized_responses.json"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <Download className="h-4 w-4 mr-2" />
                  Download Pipeline Output
                </Button>
              </a>
              <a
                href="/demos/rfi_pipeline.yaml"
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
          Explore the Output
        </h3>
        <Suspense
          fallback={
            <div className="flex items-center justify-center p-8 text-gray-600">
              <Loader2 className="h-6 w-6 animate-spin mr-2" />
              Loading Data Explorer...
            </div>
          }
        >
          <RfiResponseExplorer />
        </Suspense>
      </div>
    </main>
  );
}