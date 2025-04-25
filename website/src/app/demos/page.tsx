"use client";

import React, { Suspense } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { ExternalLink, Loader2 } from "lucide-react";
import Image from "next/image";
import { Scroll, ArrowRight } from "lucide-react";

const RfiResponseExplorer = dynamic(
  () =>
    import("@/components/demos/rfi-response-explorer").then(
      (mod) => mod.RfiResponseExplorer
    ),
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

export default function DemosIndexPage() {
  return (
    <main className="flex min-h-screen flex-col items-center p-4 sm:p-8">
      <div className="max-w-4xl w-full">
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
          <p className="text-lg sm:text-xl text-muted-foreground">
            Interactive Demos
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          <Link href="/demos/ai-rfi-response-analysis" className="block group">
            <Card className="h-full hover:shadow-lg transition-shadow duration-200 hover:border-primary/50">
              <CardHeader>
                <CardTitle className="text-lg group-hover:text-primary transition-colors">
                  AI RFI Response Analysis
                </CardTitle>
                <CardDescription>
                  Analyze public feedback on AI policy using DocETL for
                  structured data extraction.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex justify-end items-center text-sm text-primary opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                  View Demo <ArrowRight className="ml-1 h-4 w-4" />
                </div>
              </CardContent>
            </Card>
          </Link>
        </div>

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
