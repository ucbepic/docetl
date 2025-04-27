"use client";

import React, { Suspense } from "react";
import Link from "next/link";
import Image from "next/image";
import dynamic from "next/dynamic";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button"; // <-- Add Button import
import { ExternalLink, Loader2, Scroll, ArrowLeft } from "lucide-react"; // <-- Add Scroll, ArrowLeft

// Simplify the dynamic import completely
const RfiResponseExplorer = dynamic(
  () => import("@/components/showcase/rfi-response-explorer"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center p-8 text-muted-foreground">
        <Loader2 className="h-6 w-6 animate-spin mr-2" />
        Loading Data Explorer...
      </div>
    ),
  }
);

export default function AiRfiResponseAnalysisPage() {
  const handleDownloadPipeline = () => {
    const link = document.createElement("a");
    link.href = "/demos/rfi_pipeline.yaml";
    link.download = "rfi_pipeline.yaml";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4 sm:p-8">
      <div className="max-w-6xl w-full">
        {" "}
        {/* Increased max-width for explorer */}
        {/* Header */}
        <div className="text-center mb-8 sm:mb-12">
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
        {/* Back Link */}
        <div className="mb-6">
          <Button variant="outline" size="sm" asChild>
            <Link href="/showcase">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Showcase
            </Link>
          </Button>
        </div>
        {/* Demo Content */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="text-xl">AI RFI Response Analysis</CardTitle>
            <CardDescription>
              <em>Analyzing public responses to the U.S. Government&apos;s </em>
              <Link
                href="https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Request for Information
              </Link>{" "}
              on AI Action Plan Development.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              In February 2025, the U.S. Office of Science and Technology Policy
              requested public input on developing an AI Action Plan as directed
              by{" "}
              <Link
                href="https://www.federalregister.gov/documents/2025/01/31/2025-02172/removing-barriers-to-american-leadership-in-artificial-intelligence"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Executive Order 14179
              </Link>
              . The government received over 10,000 responses from individuals,
              academia, industry, and other stakeholders.
            </p>
            <p className="text-sm text-muted-foreground">
              This demo showcases DocETL&apos;s capabilities by processing these
              responses to extract structured insights. Our pipeline uses
              gpt-4o-mini to analyze each submission to identify:
            </p>
            <ul className="text-sm text-muted-foreground list-disc pl-5 space-y-1">
              <li>
                <span className="font-medium">Concrete proposals</span>: Whether
                the response contains actionable policy suggestions
              </li>
              <li>
                <span className="font-medium">Notable entity</span>: Whether the
                submission comes from a notable entity (a well-known individual,
                e.g., researcher, industry leader, celebrity, etc., or
                organization) or other
              </li>
              <li>
                <span className="font-medium">Age demographics</span>: Estimated
                age bracket of the submitter
              </li>
              <li>
                <span className="font-medium">Main topic</span>: Primary concern
                discussed in the response
              </li>
              <li>
                <span className="font-medium">Summary</span>: Concise summary of
                the submission&apos;s key points
              </li>
            </ul>
            <p className="text-sm text-muted-foreground mb-4 italic">
              The entire pipeline processing cost just $3.01 to run on all
              10,068 documents.
            </p>
            <div className="p-4 border border-yellow-300 bg-yellow-50 rounded-md mb-4">
              <p className="text-sm font-medium text-yellow-800">
                <strong>Disclaimer:</strong> Please note that this analysis may
                contain errors. Potential sources of inaccuracy include OCR
                processing of the PDF documents and limitations in the LLM agent
                used for data processing. This demonstration is for illustrative
                purposes only.
              </p>
            </div>
            <div className="flex flex-wrap gap-x-4 gap-y-2 text-sm">
              <Link
                href="https://www.nitrd.gov/coordination-areas/ai/90-fr-9088-responses/"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center text-blue-600 hover:underline"
              >
                View and Download Official RFI Responses as PDFs{" "}
                <ExternalLink className="ml-1 h-4 w-4" />
              </Link>
              <span className="text-muted-foreground hidden sm:inline">|</span>
              <Link
                href="https://docetl.blob.core.windows.net/showcase/processed_responses.json"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center text-blue-600 hover:underline"
              >
                Download DocETL Dataset{" "}
                <ExternalLink className="ml-1 h-4 w-4" />
              </Link>
              <span className="text-muted-foreground hidden sm:inline">|</span>
              <Link
                href="https://docetl.blob.core.windows.net/showcase/summarized_responses.json"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center text-blue-600 hover:underline"
              >
                Download DocETL Outputs{" "}
                <ExternalLink className="ml-1 h-4 w-4" />
              </Link>
              <span className="text-muted-foreground hidden sm:inline">|</span>
              <button
                onClick={handleDownloadPipeline}
                className="inline-flex items-center text-blue-600 hover:underline"
              >
                Download DocETL Pipeline YAML{" "}
                <ExternalLink className="ml-1 h-4 w-4" />
              </button>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-2 mt-4">
                Explore the Output:
              </h3>
              <Suspense
                fallback={
                  <div className="flex items-center justify-center p-8 text-muted-foreground">
                    <Loader2 className="h-6 w-6 animate-spin mr-2" />
                    Loading Data Explorer...
                  </div>
                }
              >
                <RfiResponseExplorer />
              </Suspense>
            </div>
          </CardContent>
        </Card>
        {/* Footer Logos */}
        <div className="mt-auto pt-8 flex justify-center items-center space-x-4 border-t">
          <a
            href="https://eecs.berkeley.edu"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              src="/berkeley.png"
              alt="UC Berkeley Logo"
              width={40}
              height={40}
              className="sm:w-[50px] sm:h-[50px]"
            />
          </a>
          <a
            href="https://epic.berkeley.edu"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              src="/epiclogo.png"
              alt="EPIC Lab Logo"
              width={120}
              height={40}
              className="sm:w-[150px] sm:h-[50px]"
            />
          </a>
        </div>
      </div>
    </main>
  );
}
