import React, { Suspense } from "react";
import type { Metadata } from "next";
import dynamic from "next/dynamic";
import { Info, Download, FileJson, Loader2, ArrowLeft, Scroll, ExternalLink } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import Link from "next/link";

const LeaseContractExplorer = dynamic(
  () => import("@/components/showcase/lease-contract-explorer"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center p-8 text-gray-600">
        <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading Explorer...
      </div>
    ),
  }
);

export const metadata: Metadata = {
  title: "AI-Powered Lease Contract Red-Flag Analysis | DocETL",
  description:
    "Interactive demo showing how DocETL uses LLMs to extract and rank red-flag clauses in lease agreements, helping teams sort legal documents by risk.",
  keywords: [
    "AI legal document analysis",
    "lease contract review",
    "red flag extraction",
    "LLM contract analysis",
    "AI document sorting",
    "generative AI legal tech",
  ],
  openGraph: {
    title: "AI-Powered Lease Contract Red-Flag Analysis | DocETL",
    description:
      "See lease agreements ranked by severity of risky clauses with interactive highlighting. Powered by DocETL pipelines and large language models.",
    url: "https://www.docetl.org/showcase/lease-contract-red-flags",
    type: "website",
    images: [
      {
        url: "/docetl-favicon-color.png",
      },
    ],
  },
};

export default function LeaseContractRedFlagsPage() {
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
          Lease Contract Red-Flag Analysis
        </h1>
        <p className="text-gray-600">
          Ranking 179 lease agreements by severity of automatically extracted red flags using the{" "}
          <a
            href="https://arxiv.org/abs/2010.10386"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline"
          >
            ALeaseBERT
          </a>{" "}
          dataset
        </p>
      </div>

      <Alert className="mb-6 bg-blue-50 border-blue-200">
        <Info className="h-4 w-4 text-blue-600" />
        <AlertTitle className="text-blue-900">About This Analysis</AlertTitle>
        <AlertDescription className="text-blue-800">
          <div className="space-y-3 mt-2">
            <p>
              Many organizations need to sort large collections of legal documents by latent attributes—for 
              example, which contracts pose the greatest risk to the tenant. DocETL pipelines can express 
              this kind of semantic ranking in a few declarative steps.
            </p>
            
            <p>
              This demo takes 179 lease agreements and automatically <strong>ranks them by the severity 
              of "red flags"</strong>—provisions that are potentially dangerous for the lessee 
              (early-termination clauses, uncapped fee escalations, etc.). The pipeline first extracts 
              candidate red-flag sentences for each contract, then ranks the contracts by severity. 
              Contracts at the top contain the most egregious red flags; those at the bottom are relatively benign.
            </p>

            <p className="text-sm">
              <strong>Processing details:</strong> 179 lease contracts • gpt-4.1-mini • $4.24 total cost
            </p>
            
            <div className="p-3 border border-yellow-300 bg-yellow-50 rounded-md mt-3">
              <p className="text-sm font-medium text-yellow-800">
                <strong>Note:</strong> There is no "ground truth" ranking for severity. The AI's assessment 
                is based on general legal principles and may not capture all nuances or jurisdiction-specific issues.
              </p>
            </div>
            
            <div className="flex flex-wrap gap-3 mt-4">
              <a
                href="https://docetlcloudbank.blob.core.windows.net/demos/lease_dataset.json"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <FileJson className="h-4 w-4 mr-2" />
                  Download Pipeline Input
                </Button>
              </a>
              <a
                href="https://docetlcloudbank.blob.core.windows.net/demos/red_flag_analysis.json"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <Download className="h-4 w-4 mr-2" />
                  Download Pipeline Output
                </Button>
              </a>
              <a
                href="/demos/contracts_pipeline.yaml"
                download
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <FileJson className="h-4 w-4 mr-2" />
                  Download Pipeline YAML
                </Button>
              </a>
              <a
                href="https://uvaauas.figshare.com/articles/dataset/ALeaseBert/19732993"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <ExternalLink className="h-4 w-4 mr-2" />
                  View Original Dataset
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
        <p className="text-sm text-gray-600 mb-4">
          Select any contract to read it with highlights. Click a red flag on the right to jump to its location in the text.
        </p>
        <Suspense
          fallback={
            <div className="flex items-center justify-center p-8 text-gray-600">
              <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading Explorer...
            </div>
          }
        >
          <LeaseContractExplorer />
        </Suspense>
      </div>
    </main>
  );
}