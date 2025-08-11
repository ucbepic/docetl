import React from "react";
import HealthQuotesExplorer from "@/components/showcase/health-quotes-explorer";
import {
  Info,
  Download,
  FileJson,
  ExternalLink,
  ArrowLeft,
  Scroll,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function CongressionalHealthHearingsPage() {
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
          Congressional Health Hearings Analysis
        </h1>
        <p className="text-gray-600">
          How Congress Talks About Your Health — 243 Hearings from 2000-2025
        </p>
      </div>

      <Alert className="mb-6 bg-blue-50 border-blue-200">
        <Info className="h-4 w-4 text-blue-600" />
        <AlertTitle className="text-blue-900">About This Analysis</AlertTitle>
        <AlertDescription className="text-blue-800">
          <div className="space-y-3 mt-2">
            <p>
              DocETL analyzed 243 congressional hearing transcripts about health
              policy, extracting the most memorable quotes and tracking how
              different speakers engage with critical healthcare topics. The
              pipeline uses GPT-5 to identify impactful statements, measure
              evasion rates when witnesses dodge questions, and classify the
              emotional tone of responses.
            </p>

            <p>
              Each quote receives an <strong>impact score</strong> (0-100%)
              indicating how memorable or policy-relevant it is, while the{" "}
              <strong>evasion rate</strong> shows what percentage of questions
              went unanswered. Speaker roles distinguish between Congress
              members, committee chairs, and various witness types (industry
              representatives, physicians, academics, patients).
            </p>

            <p className="text-sm">
              <strong>Processing details:</strong> 243 transcripts • GPT-5
              series model • $17.33 total cost • Extracted from government API
              (2000-2025)
            </p>

            <div className="flex flex-wrap gap-3 mt-4">
              <a
                href="https://docetlcloudbank.blob.core.windows.net/demos/hearings_2000_plus.json"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <FileJson className="h-4 w-4 mr-2" />
                  Download Pipeline Input
                </Button>
              </a>
              <a
                href="https://docetlcloudbank.blob.core.windows.net/demos/health_quotes.json"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button variant="outline" size="sm" className="bg-white">
                  <Download className="h-4 w-4 mr-2" />
                  Download Pipeline Output
                </Button>
              </a>
              <a href="/demos/congress_health_pipeline.yaml" download>
                <Button variant="outline" size="sm" className="bg-white">
                  <FileJson className="h-4 w-4 mr-2" />
                  Download Pipeline YAML
                </Button>
              </a>
            </div>
          </div>
        </AlertDescription>
      </Alert>

      <HealthQuotesExplorer />
    </main>
  );
}
