"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import PresidentialDebateDemo from "@/components/PresidentialDebateDemo";
import { sendGAEvent } from "@next/third-parties/google";
import { Scroll } from "lucide-react";

const papers = [
  {
    title: "DocETL: Agentic Query Rewriting and Evaluation for Complex Document Processing",
    venue: "VLDB 2025",
    url: "https://arxiv.org/abs/2410.12189",
  },
  {
    title: "Steering Semantic Data Processing With DocWrangler",
    venue: "UIST 2025",
    url: "https://arxiv.org/abs/2504.14764",
  },
  {
    title: "Multi-Objective Agentic Rewrites for Unstructured Data Processing",
    venue: "VLDB 2026",
    url: "https://arxiv.org/abs/2512.02289",
  },
];

export default function Home() {
  const [showDemo, setShowDemo] = useState(false);

  useEffect(() => {
    if (window.innerWidth >= 640) {
      setShowDemo(true);
    }
  }, []);

  const toggleDemo = () => {
    setShowDemo(!showDemo);
    sendGAEvent("event", "buttonClicked", { value: "demo" });
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center px-4 py-8 sm:px-8">
      <div className="max-w-3xl w-full min-w-0">
        <div className="flex items-center justify-center mb-4">
          <Scroll
            className="w-10 h-10 sm:w-12 sm:h-12 mr-2 text-primary"
            strokeWidth={1.5}
          />
          <span className="logo-text text-2xl sm:text-3xl">DocETL</span>
        </div>

        <p className="text-center text-sm sm:text-lg text-muted-foreground mb-6 break-words">
          Turn your unstructured documents into structured insights.
        </p>

        <div className="flex flex-wrap justify-center gap-x-5 gap-y-2 text-sm mb-8">
          <a
            href="https://ucbepic.github.io/docetl/"
            target="_blank"
            rel="noopener noreferrer"
            className="font-semibold text-primary hover:underline"
          >
            Documentation
          </a>
          <a
            href="https://github.com/ucbepic/docetl"
            target="_blank"
            rel="noopener noreferrer"
            className="font-semibold text-primary hover:underline"
          >
            GitHub
          </a>
          <Link
            href="/showcase"
            className="font-semibold text-primary hover:underline"
          >
            Showcase
          </Link>
          <Link
            href="/playground"
            className="font-semibold text-primary hover:underline"
          >
            Playground
          </Link>
          <a
            href="https://discord.gg/fHp7B2X3xx"
            target="_blank"
            rel="noopener noreferrer"
            className="font-semibold text-primary hover:underline"
          >
            Discord
          </a>
        </div>

        <div className="text-center mb-6">
          <button
            onClick={toggleDemo}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            {showDemo ? "Hide example" : "Show example"}
          </button>
        </div>

        {showDemo && (
          <div className="demo-wrapper show mb-8 overflow-x-auto">
            <div className="demo-content">
              <div className="demo-inner min-w-0">
                <PresidentialDebateDemo />
              </div>
            </div>
          </div>
        )}

        <div className="border-t pt-6">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
            Research
          </h3>
          <ul className="space-y-1.5">
            {papers.map((paper) => (
              <li key={paper.url} className="text-sm">
                <a
                  href={paper.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-foreground hover:text-primary hover:underline"
                >
                  {paper.title}
                </a>
                <span className="text-muted-foreground ml-1.5">
                  {paper.venue}
                </span>
              </li>
            ))}
          </ul>
        </div>

        <div className="mt-8 flex justify-center items-center space-x-4">
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
