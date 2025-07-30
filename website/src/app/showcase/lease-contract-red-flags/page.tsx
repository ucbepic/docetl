"use client";

import React, { Suspense } from "react";
import Head from "next/head";
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

const LeaseContractExplorer = dynamic(
  () => import("@/components/showcase/lease-contract-explorer"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center p-8 text-muted-foreground">
        <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading Explorer...
      </div>
    ),
  }
);

export default function LeaseContractRedFlagsPage() {
  const handleDownloadPipeline = () => {
    const link = document.createElement("a");
    link.href = "/demos/contracts_pipeline.yaml";
    link.download = "contracts_pipeline.yaml";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <>
      <Head>
        <title>Lease Contract Red-Flag Analysis | DocETL Showcase</title>
        <meta
          name="description"
          content="See how DocETL ranks lease agreements by the severity of red flags extracted using LLMs and the ALeaseBERT dataset."
        />
        <meta
          property="og:title"
          content="Lease Contract Red-Flag Analysis | DocETL"
        />
        <meta
          property="og:description"
          content="Interactive demo: rank lease agreements by risk, explore highlighted red-flag clauses, and learn how DocETL pipelines work."
        />
        <meta property="og:type" content="website" />
        <meta
          property="og:url"
          content="https://docetl.ai/showcase/lease-contract-red-flags"
        />
      </Head>
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
          {/* Back */}
          <div className="mb-6">
            <Button variant="outline" size="sm" asChild>
              <Link href="/showcase">
                <ArrowLeft className="mr-2 h-4 w-4" /> Back to Showcase
              </Link>
            </Button>
          </div>
          {/* Content Card */}
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="text-xl">
                Lease Contract Red-Flag Analysis
              </CardTitle>
              <CardDescription>
                Ranking lease agreements by severity of automatically extracted
                red flags using the{" "}
                <Link
                  href="https://arxiv.org/abs/2010.10386"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  ALeaseBERT
                </Link>{" "}
                dataset.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Many organizations need to{" "}
                <strong>
                  sort large collections of legal documents by latent attributes
                </strong>
                —for example, which contracts pose the greatest risk to the
                tenant. DocETL pipelines can express this kind of semantic
                ranking in a few declarative steps. In this demo we take 179
                lease agreements from the ALeaseBERT dataset and automatically{" "}
                <strong>
                  rank them by the severity&nbsp;of &quot;red flags&quot;
                </strong>{" "}
                &nbsp;– provisions that are potentially dangerous for the lessee
                (early-termination clauses, uncapped fee escalations, etc.).
                Contracts at the top of the list contain the most egregious red
                flags; those at the bottom are relatively benign.
              </p>
              <p className="text-sm text-muted-foreground">
                The pipeline first calls an LLM to{" "}
                <em>extract candidate red-flag sentences</em> for each contract,
                then calls an LLM to rank the contracts by the severity of the
                red flags. Finally we render the results below: select any
                contract to read it with highlights, and click a red flag on the
                right to jump to its location in the text. Note that there is no
                &quot;ground truth&quot; ranking for severity; ideally the most
                severe red flags would be at the top of the list, and the bottom
                would be relatively benign.
              </p>
              <p className="text-sm text-muted-foreground italic">
                Total cost for us to run: $4.24 (gpt-4.1-mini).
              </p>
              <div className="flex flex-wrap gap-x-4 gap-y-2 text-sm">
                <Link
                  href="https://uvaauas.figshare.com/articles/dataset/ALeaseBert/19732993"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center text-blue-600 hover:underline"
                >
                  Original Dataset <ExternalLink className="ml-1 h-4 w-4" />
                </Link>
                <span className="text-muted-foreground hidden sm:inline">
                  |
                </span>
                <Link
                  href="https://docetl.blob.core.windows.net/demos/lease_dataset.json"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center text-blue-600 hover:underline"
                >
                  DocETL Input <ExternalLink className="ml-1 h-4 w-4" />
                </Link>
                <span className="text-muted-foreground hidden sm:inline">
                  |
                </span>
                <Link
                  href="https://docetl.blob.core.windows.net/demos/red_flag_analysis.json"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center text-blue-600 hover:underline"
                >
                  DocETL Outputs <ExternalLink className="ml-1 h-4 w-4" />
                </Link>
                <span className="text-muted-foreground hidden sm:inline">
                  |
                </span>
                <button
                  onClick={handleDownloadPipeline}
                  className="inline-flex items-center text-blue-600 hover:underline"
                >
                  Download Pipeline YAML{" "}
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
                      <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading
                      Explorer...
                    </div>
                  }
                >
                  <LeaseContractExplorer />
                </Suspense>
              </div>
            </CardContent>
          </Card>

          {/* GitHub Star CTA */}
          <div className="mb-8 text-center">
            <Link
              href="https://github.com/epicLab/docetl"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-3 py-2 bg-gray-900 text-white rounded hover:bg-gray-800"
            >
              ⭐ Star DocETL on GitHub
            </Link>
          </div>
          {/* Footer */}
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
    </>
  );
}
