"use client";

import React, { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { Scroll, ChevronDown, ChevronUp } from "lucide-react";
import PresidentialDebateDemo from "@/components/PresidentialDebateDemo";
import { Button } from "@/components/ui/button";
import { sendGAEvent } from "@next/third-parties/google";

export default function Home() {
  const [showDemo, setShowDemo] = useState(true);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 sm:p-8">
      <div className="max-w-4xl w-full text-center">
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
            <em>New NotebookLM Podcast!</em>{" "}
            <a
              href="https://notebooklm.google.com/notebook/ef73248b-5a43-49cd-9976-432d20f9fa4f/audio?pli=1"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 underline"
            >
              Sept 28, 2024
            </a>
            . Thanks to Shabie from our Discord community!
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
            onClick={() => {
              setShowDemo(!showDemo);
              sendGAEvent("event", "buttonClicked", { value: "demo" });
            }}
            className="btn btn-primary flex items-center"
          >
            Demo
            {showDemo ? (
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

        <div className={`demo-wrapper ${showDemo ? "show" : ""}`}>
          <div className="demo-content">
            <div className="demo-inner">
              <PresidentialDebateDemo />
            </div>
          </div>
        </div>

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
