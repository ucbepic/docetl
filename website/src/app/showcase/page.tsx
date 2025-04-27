"use client";

import React from "react";
import Link from "next/link";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import Image from "next/image";
import { Scroll, ArrowRight } from "lucide-react";

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
            Interactive Showcase
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          <Link
            href="/showcase/ai-rfi-response-analysis"
            className="block group"
          >
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

          <Link
            href="/showcase/ai-system-prompts-analysis"
            className="block group"
          >
            <Card className="h-full hover:shadow-lg transition-shadow duration-200 hover:border-primary/50">
              <CardHeader>
                <CardTitle className="text-lg group-hover:text-primary transition-colors">
                  AI System Prompts Analysis
                </CardTitle>
                <CardDescription>
                  Explore common patterns and strategies in system prompts from
                  popular AI tools and agents.
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

        {/* Add Community Contribution Note */}
        <div className="text-center text-muted-foreground mt-8 mb-12">
          <p>
            Want to add your own demo to the showcase? We welcome community
            contributions!
          </p>
          <p>
            Join our{" "}
            <a
              href="https://discord.gg/fHp7B2X3xx"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              Discord server
            </a>{" "}
            to discuss your ideas.
          </p>
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
