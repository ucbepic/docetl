"use client";

import React, { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { Scroll, ChevronDown, ChevronUp } from "lucide-react";
import PresidentialDebateDemo from "@/components/PresidentialDebateDemo";
import { Button } from "@/components/ui/button";
import { sendGAEvent } from "@next/third-parties/google";
import { Card, CardContent } from "@/components/ui/card";

export default function Home() {
  const [showDemo, setShowDemo] = useState(true);
  const [showVision, setShowVision] = useState(false);

  const toggleDemo = () => {
    setShowDemo(!showDemo);
    if (!showDemo) {
      setShowVision(false);
    }
    sendGAEvent("event", "buttonClicked", { value: "demo" });
  };

  const toggleVision = () => {
    setShowVision(!showVision);
    if (!showVision) {
      setShowDemo(false);
    }
    sendGAEvent("event", "buttonClicked", { value: "vision" });
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 sm:p-8">
      <div className="max-w-4xl w-full">
        <div className="text-center">
          <div className="flex items-center justify-center mb-2">
            <Scroll
              className="w-12 h-12 sm:w-16 sm:h-16 mr-2 text-primary"
              strokeWidth={1.5}
            />
            <span className="logo-text text-2xl sm:text-3xl">docetl</span>
          </div>
          <p className="text-lg sm:text-xl mb-4 sm:mb-6">
            Powering complex document processing pipelines
          </p>

          <div className="max-w-lg mx-auto">
            <p className="text-sm sm:text-md mb-1 text-gray-600">
              <em>New IDE Released!</em>{" "}
              <a
                href="https://ucbepic.github.io/docetl/playground/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 underline"
              >
                Dec 2, 2024
              </a>
              ! Try out our new web-based IDE.
            </p>
            <p className="text-sm sm:text-md mb-6 text-gray-600">
              <em>New blog post!</em>{" "}
              <a
                href="https://data-people-group.github.io/blogs/2024/09/24/docetl/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 underline"
              >
                September 24, 2024
              </a>
            </p>
          </div>

          <div className="flex flex-wrap justify-center gap-4 mb-6 sm:mb-8">
            <Button
              onClick={toggleDemo}
              className="btn btn-primary flex items-center"
            >
              Demo
              {showDemo ? (
                <ChevronUp className="ml-2 h-4 w-4" />
              ) : (
                <ChevronDown className="ml-2 h-4 w-4" />
              )}
            </Button>

            <Button onClick={toggleVision} className="btn btn-secondary">
              Research Projects{" "}
              {showVision ? (
                <ChevronUp className="ml-2 h-4 w-4" />
              ) : (
                <ChevronDown className="ml-2 h-4 w-4" />
              )}
            </Button>

            {/* <Button asChild className="btn btn-secondary">
              <Link href="/blog">Blog</Link>
            </Button> */}
            <Button
              asChild
              className="btn btn-secondary"
              onClick={() =>
                sendGAEvent("event", "buttonClicked", { value: "discord" })
              }
            >
              <a
                href="https://discord.gg/fHp7B2X3xx"
                target="_blank"
                rel="noopener noreferrer"
              >
                Discord
              </a>
            </Button>
            <Button
              asChild
              className="btn btn-secondary"
              onClick={() =>
                sendGAEvent("event", "buttonClicked", { value: "github" })
              }
            >
              <a
                href="https://github.com/ucbepic/docetl"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </a>
            </Button>
            <Button
              asChild
              className="btn btn-secondary"
              onClick={() =>
                sendGAEvent("event", "buttonClicked", { value: "docs" })
              }
            >
              <Link href="https://ucbepic.github.io/docetl/" target="_blank">
                Docs
              </Link>
            </Button>
            <Button
              className="btn btn-secondary"
              onClick={() =>
                sendGAEvent("event", "buttonClicked", { value: "paper" })
              }
            >
              <Link href="https://arxiv.org/abs/2410.12189" target="_blank">
                Paper
              </Link>
            </Button>
          </div>
        </div>

        {showVision && (
          <Card className="max-w-4xl mx-auto bg-white shadow-md">
            <CardContent className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2">
                  Reimagining Data Systems for Semantic Operations
                </h2>
                <p className="text-sm text-muted-foreground mb-6">
                  While traditional database systems excel at structured data
                  processing, semantic operations powered by LLMs bring
                  unprecedented expressiveness and flexibility. However, these
                  operations introduce new challenges: they can be incorrect,
                  are computationally intensive, and typically rely on remote
                  API calls. We&apos;re reimagining data systems throughout the
                  stack to address these unique challenges. Here are some
                  projects we are working on:
                </p>

                <div className="grid gap-6">
                  <div className="border rounded-md p-4">
                    <h4 className="font-medium text-primary mb-2">
                      Query Optimizer
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      Current LLM-powered systems focus mainly on cost
                      reduction. But for complex tasks, even well-crafted
                      operations can produce inaccurate results. The DocETL
                      optimizer uses LLM agents to automatically rewrite
                      pipelines, by breaking operations down into smaller,
                      well-scoped tasks to improve accuracy.{" "}
                      <a
                        href="https://arxiv.org/abs/2410.12189"
                        className="text-blue-500 hover:underline inline-flex items-center"
                      >
                        Read our paper <span className="ml-1">→</span>
                      </a>
                    </p>
                  </div>

                  <div className="border rounded-md p-4">
                    <h4 className="font-medium text-primary mb-2">
                      Execution Engine
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      Our users consistently highlight map operations as the
                      most valuable feature, but these require at least one LLM
                      call per document—making them prohibitively expensive at
                      scale. We&apos;re exploring novel techniques to
                      dramatically reduce costs for open-ended map operations
                      without sacrificing accuracy.
                    </p>
                  </div>

                  <div className="border rounded-md p-4">
                    <h4 className="font-medium text-primary mb-2">
                      Interactive Interface
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      Semantic operations are highly expressive, but this power
                      comes with a challenge—they can be fuzzy and ambiguous in
                      practice. Consequently, users often need many iterations
                      to get semantic operations right. Through the DocETL IDE,
                      we&apos;re designing interfaces that help users explore
                      data, refine their intents, and quickly iterate on prompts
                      and operations.
                    </p>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground mt-6">
                  There are many domain-specific unstructured data processing
                  needs that can benefit from systems like DocETL. We work with
                  partners at universities, governments, and institutions to
                  explore how AI can improve data workflows, especially for
                  domain experts and those who may not have data or ML
                  expertise. If you&apos;d like to learn more (e.g., bring
                  DocETL to your team or join our case studies), please reach
                  out to{" "}
                  <a
                    href="mailto:shreyashankar@berkeley.edu"
                    className="text-blue-500 hover:underline"
                  >
                    shreyashankar@berkeley.edu
                  </a>
                  .
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {showDemo && (
          <div className="demo-wrapper show">
            <div className="demo-content">
              <div className="demo-inner">
                <PresidentialDebateDemo />
              </div>
            </div>
          </div>
        )}

        <div className="mt-6 sm:mt-8 flex justify-center items-center space-x-4">
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
