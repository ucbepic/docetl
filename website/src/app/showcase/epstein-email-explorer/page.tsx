import React, { Suspense } from "react";
import type { Metadata } from "next";
import dynamic from "next/dynamic";
import { Info, Loader2, ArrowLeft, Scroll, ExternalLink } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import Link from "next/link";

const EpsteinEmailExplorer = dynamic(
  () => import("@/components/showcase/epstein-email-explorer"),
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
  title: "Epstein Email Archive Explorer | DocETL Showcase",
  description:
    "Explore 2,322 emails from Jeffrey Epstein's correspondence released by the House Oversight Committee. Interactive analysis with AI-extracted metadata including participants, topics, flagged concerns, and victim mentions. Processed with DocETL for $8.04.",
  keywords: [
    "Epstein emails",
    "investigative journalism",
    "document analysis",
    "AI email analysis",
    "House Oversight Committee",
    "DocETL",
    "email metadata extraction",
  ],
  openGraph: {
    title: "Epstein Email Archive Explorer - 2,322 Emails Analyzed",
    description:
      "Interactive explorer of Jeffrey Epstein emails from House Oversight Committee release. AI-powered analysis: participants, organizations, topics, potential concerns. Download insights for AI chat. Built with DocETL ($8.04 processing cost).",
    url: "https://www.docetl.org/showcase/epstein-email-explorer",
    type: "website",
    siteName: "DocETL",
    images: [
      {
        url: "/docetl-favicon-color.png",
        width: 512,
        height: 512,
        alt: "DocETL Logo",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Epstein Email Archive Explorer - 2,322 Emails",
    description:
      "Interactive explorer of Jeffrey Epstein emails with AI-extracted metadata. Filter by participants, topics, concerns. Download for AI analysis. Built with DocETL.",
  },
};

export default function EpsteinEmailExplorerPage() {
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
          Epstein Email Archive Explorer
        </h1>
        <p className="text-gray-600">
          Interactive analysis of 2,322 emails from Jeffrey Epstein&apos;s correspondence
        </p>
      </div>

      <Alert className="mb-6 bg-gray-50 border-gray-200">
        <Info className="h-4 w-4 text-gray-600" />
        <AlertTitle className="text-gray-900">About This Dataset</AlertTitle>
        <AlertDescription className="text-gray-700">
          <div className="space-y-3 mt-2">
            <p>
              On November 12, 2025, the House Oversight Committee released email correspondence
              involving Jeffrey Epstein. This interactive explorer uses DocETL to analyze
              and structure this public dataset, enabling journalists and researchers to
              investigate connections, topics, and potentially concerning patterns.
            </p>

            <p>
              The pipeline extracted entities (people, organizations, locations), analyzed
              tone and topics, identified potential concerns, and summarized each email.
              Use the attribute-based filters to search within specific fields like participants,
              organizations, subject lines, or email content.
            </p>

            <p className="text-sm">
              We spent <strong>$8.04</strong> to run the DocETL pipeline.
            </p>

            <p className="text-sm text-gray-600 mt-3">
              If you use this tool or analysis in your work, please reference this website and the original
              House Oversight Committee data release. Thank you!
            </p>

            <div className="p-3 border border-yellow-300 bg-yellow-50 rounded-md mt-3">
              <p className="text-sm font-medium text-yellow-800">
                <strong>Disclaimer:</strong> This analysis is automated and provided for
                investigative research purposes. AI assessments may contain errors and should
                be verified against primary sources. The presence of flags does not constitute
                proof of wrongdoing.
              </p>
            </div>

            <div className="flex flex-wrap gap-3 mt-4">
              <a
                href="https://oversightdemocrats.house.gov/news/press-releases/house-oversight-committee-releases-jeffrey-epstein-email-correspondence-raising"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <ExternalLink className="h-4 w-4 mr-2" />
                  View Original Release
                </Button>
              </a>
            </div>
          </div>
        </AlertDescription>
      </Alert>

      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-900">
          Explore the Emails
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Use attribute-based filters to narrow results (e.g., select &quot;Participant&quot; then search for a name).
          Click any email to view full details. Click on person names to see all their related emails
          and add them as filters.
        </p>
        <Suspense
          fallback={
            <div className="flex items-center justify-center p-8 text-gray-600">
              <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading Explorer...
            </div>
          }
        >
          <EpsteinEmailExplorer />
        </Suspense>
      </div>
    </main>
  );
}
