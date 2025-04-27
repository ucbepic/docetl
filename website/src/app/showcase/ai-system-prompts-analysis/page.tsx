"use client";

import React, { Suspense, useState } from "react";
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
import { Button } from "@/components/ui/button";
import { ExternalLink, Loader2, Scroll, ArrowLeft } from "lucide-react";

// Dynamic import for the prompts explorer component
const SystemPromptsExplorer = dynamic(
  () => import("@/components/showcase/system-prompts-explorer"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center p-8 text-muted-foreground">
        <Loader2 className="h-6 w-6 animate-spin mr-2" />
        Loading Prompts Explorer...
      </div>
    ),
  }
);

export default function AiSystemPromptsAnalysisPage() {
  const handleDownloadPipeline = () => {
    const link = document.createElement("a");
    link.href = "/demos/prompts_pipeline.yaml";
    link.download = "prompts_pipeline.yaml";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4 sm:p-8">
      <div className="max-w-6xl w-full">
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
            <CardTitle className="text-xl">
              Prompt Engineering Strategies from Popular AI Assistants
            </CardTitle>
            <CardDescription>
              <em>
                Analyzing system prompts from popular AI assistants to extract
                common prompt engineering strategies.
              </em>
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Using the{" "}
              <Link
                href="https://github.com/dontriskit/awesome-ai-system-prompts"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                awesome-ai-system-prompts
              </Link>{" "}
              repository as source data, which contains leaked system prompts
              from major AI systems like ChatGPT, Claude, Manus, and others,
              we&apos;ve used DocETL to analyze and extract structured
              information about strategies employed across these different AI
              assistants.
            </p>
            <p className="text-sm text-muted-foreground">
              Our pipeline identifies:
            </p>
            <ul className="text-sm text-muted-foreground list-disc pl-5 space-y-1">
              <li>
                <span className="font-medium">Common strategies</span>:
                Recurring patterns and approaches used across different AI
                systems
              </li>
              <li>
                <span className="font-medium">
                  Strategy implementation examples
                </span>
                : How specific AI systems implement these strategies
              </li>
              <li>
                <span className="font-medium">Strategy summaries</span>: Concise
                explanations of each strategy and its implementation
                considerations
              </li>
            </ul>
            <p className="text-sm text-muted-foreground mb-4">
              This analysis can help prompt engineers and AI developers identify
              best practices and effective approaches when designing system
              prompts for their own applications.
            </p>
            <p className="text-sm text-muted-foreground mb-4 italic">
              The entire pipeline processing cost just $0.18 to run for all 19
              systems.
            </p>
            <div className="p-4 border border-yellow-300 bg-yellow-50 rounded-md mb-4">
              <p className="text-sm font-medium text-yellow-800">
                <strong>Note:</strong> The strategies identified are extracted
                through automated analysis and may not capture all nuances or
                context. Use these insights as a starting point for your own
                prompt engineering work.
              </p>
            </div>
            <div className="flex flex-wrap gap-x-4 gap-y-2 text-sm">
              <Link
                href="https://github.com/dontriskit/awesome-ai-system-prompts"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center text-blue-600 hover:underline"
              >
                View Source Repository <ExternalLink className="ml-1 h-4 w-4" />
              </Link>
              <span className="text-muted-foreground hidden sm:inline">|</span>
              <Link
                href="https://docetl.blob.core.windows.net/showcase/prompts.json"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center text-blue-600 hover:underline"
              >
                Download DocETL Dataset{" "}
                <ExternalLink className="ml-1 h-4 w-4" />
              </Link>
              <span className="text-muted-foreground hidden sm:inline">|</span>
              <Link
                href="https://docetl.blob.core.windows.net/demos/analyzed_strategies.json"
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
                Explore System Prompt Strategies:
              </h3>
              <Suspense
                fallback={
                  <div className="flex items-center justify-center p-8 text-muted-foreground">
                    <Loader2 className="h-6 w-6 animate-spin mr-2" />
                    Loading Prompts Explorer...
                  </div>
                }
              >
                <SystemPromptsExplorer />
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
