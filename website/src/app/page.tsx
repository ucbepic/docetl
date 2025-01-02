"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import PresidentialDebateDemo from "@/components/PresidentialDebateDemo";
import { Button } from "@/components/ui/button";
import { sendGAEvent } from "@next/third-parties/google";
import { Card, CardContent } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Scroll,
  ChevronDown,
  ChevronUp,
  Play,
  FileCode,
  Github,
  BookOpen,
  FileText,
  MessageCircle,
  Gamepad2,
  Menu,
} from "lucide-react";

export default function Home() {
  const [showDemo, setShowDemo] = useState(false);
  const [showVision, setShowVision] = useState(false);

  useEffect(() => {
    if (window.innerWidth >= 640) {
      setShowDemo(true);
    }
  }, []);

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

  const newsItems = [
    {
      highlight: "Launched DocWrangler!",
      content:
        "Making prompt engineering easier than ever with DocWrangler, our new interactive playground.",
      link: "/playground",
      linkText: "Try it now",
      date: "Dec 2024",
    },
    {
      highlight: "Gave talks on DocETL!",
      content: "Check out our presentation slides.",
      link: "/docetl-50m-fall-2024.pdf",
      linkText: "View slides",
      date: "Nov 2024",
    },
    {
      highlight: "New paper on agentic query optimization!",
      content:
        "Improving LLM reliability through intelligent task decomposition.",
      link: "https://arxiv.org/abs/2410.12189",
      linkText: "Read paper",
      date: "Oct 2024",
    },
    {
      highlight: "New blog post!",
      content:
        "Learn how DocETL is transforming document processing pipelines.",
      link: "https://data-people-group.github.io/blogs/2024/09/24/docetl/",
      linkText: "Read post",
      date: "Sep 2024",
    },
  ];

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 sm:p-8">
      <div className="max-w-4xl w-full">
        <div className="text-center">
          <div className="flex items-center justify-center mb-2">
            <Scroll
              className="w-12 h-12 sm:w-16 sm:h-16 mr-2 text-primary"
              strokeWidth={1.5}
            />
            <span className="logo-text text-2xl sm:text-3xl">DocETL</span>
          </div>
          <p className="text-lg sm:text-xl mb-4 sm:mb-6">
            A system for LLM-powered data processing
          </p>

          <div className="max-w-lg mx-auto flex flex-col items-center mb-6">
            <Popover>
              <PopoverTrigger asChild>
                <div className="w-full overflow-hidden bg-primary/5 rounded-lg cursor-pointer hover:bg-primary/10 transition-colors">
                  <div className="news-ticker py-2">
                    <div className="news-ticker-content">
                      <div className="ticker-item">
                        {newsItems.map((item, i) => (
                          <React.Fragment key={i}>
                            <em>{item.highlight}</em> {item.content}{" "}
                            <span className="text-muted-foreground">
                              ({item.date})
                            </span>{" "}
                            <a
                              href={item.link}
                              className="text-blue-500 hover:underline"
                              onClick={(e) => e.stopPropagation()}
                              target={
                                item.link.startsWith("http")
                                  ? "_blank"
                                  : undefined
                              }
                              rel={
                                item.link.startsWith("http")
                                  ? "noopener noreferrer"
                                  : undefined
                              }
                            >
                              {item.linkText} →
                            </a>
                            <span className="ticker-space" />
                          </React.Fragment>
                        ))}
                      </div>
                      <div className="ticker-item">
                        <em>Launched DocWrangler!</em> Making prompt engineering
                        easier than ever with our new interactive playground{" "}
                        <a
                          href="/playground"
                          className="text-blue-500 hover:underline"
                        >
                          Dec 2024
                        </a>
                        <span className="ticker-space" />
                        <em>Gave talks on DocETL!</em> Check out our
                        presentation slides{" "}
                        <a
                          href="/docetl-50m-fall-2024.pdf"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-500 hover:underline"
                        >
                          Nov 2024
                        </a>
                        <span className="ticker-space" />
                        <em>New paper on agentic query optimization!</em>{" "}
                        Improving LLM reliability through intelligent task
                        decomposition{" "}
                        <a
                          href="https://arxiv.org/abs/2410.12189"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-500 hover:underline"
                        >
                          Oct 2024
                        </a>
                        <span className="ticker-space" />
                        <em>New blog post!</em> Learn how DocETL is transforming
                        data processing{" "}
                        <a
                          href="https://data-people-group.github.io/blogs/2024/09/24/docetl/"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-500 hover:underline"
                        >
                          Sep 2024
                        </a>
                        <span className="ticker-space" />
                      </div>
                    </div>
                  </div>
                </div>
              </PopoverTrigger>
              <PopoverContent className="w-[calc(100vw-2rem)] sm:w-[500px] max-h-[400px] overflow-y-auto p-4">
                <div className="space-y-4">
                  {newsItems.map((item, i) => (
                    <div
                      key={i}
                      className="space-y-2 border-b last:border-b-0 pb-3 last:pb-0"
                    >
                      <div className="font-medium text-base">
                        <em className="text-primary">{item.highlight}</em>{" "}
                        <span className="text-muted-foreground text-sm">
                          ({item.date})
                        </span>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {item.content}{" "}
                        <a
                          href={item.link}
                          className="text-blue-500 hover:underline inline-flex items-center gap-1"
                          target={
                            item.link.startsWith("http") ? "_blank" : undefined
                          }
                          rel={
                            item.link.startsWith("http")
                              ? "noopener noreferrer"
                              : undefined
                          }
                        >
                          {item.linkText} →
                        </a>
                      </p>
                    </div>
                  ))}
                </div>
              </PopoverContent>
            </Popover>
          </div>

          <div className="flex flex-col items-center gap-6 mb-6 sm:mb-8">
            {/* Mobile Dropdowns */}
            <div className="flex gap-4 sm:hidden w-full max-w-xl">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button className="flex-1 btn btn-primary font-bold">
                    <Menu className="mr-2 h-4 w-4" />
                    Try DocETL
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="center" className="w-48">
                  <DropdownMenuItem onClick={toggleDemo}>
                    <Play className="mr-2 h-4 w-4" />
                    Demo
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link href="/playground">
                      <Gamepad2 className="mr-2 h-4 w-4" />
                      Playground
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <a
                      href="https://github.com/ucbepic/docetl"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Github className="mr-2 h-4 w-4" />
                      GitHub
                    </a>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button className="flex-1 bg-secondary/70 hover:bg-secondary/60 text-secondary-foreground font-bold">
                    <Menu className="mr-2 h-4 w-4" />
                    Resources
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="center" className="w-48">
                  <DropdownMenuItem onClick={toggleVision}>
                    <FileCode className="mr-2 h-4 w-4" />
                    Research Projects
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <a
                      href="https://discord.gg/fHp7B2X3xx"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <MessageCircle className="mr-2 h-4 w-4" />
                      Discord
                    </a>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link
                      href="https://ucbepic.github.io/docetl/"
                      target="_blank"
                    >
                      <BookOpen className="mr-2 h-4 w-4" />
                      Docs
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link
                      href="https://arxiv.org/abs/2410.12189"
                      target="_blank"
                    >
                      <FileText className="mr-2 h-4 w-4" />
                      Paper
                    </Link>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            {/* Desktop Buttons - Same as before but hidden on mobile */}
            <div className="hidden sm:flex flex-col items-center gap-6 w-full">
              <div className="flex flex-wrap justify-center gap-4 w-full max-w-xl">
                <Button
                  onClick={toggleDemo}
                  className="flex-1 h-10 btn btn-primary flex items-center justify-center font-bold"
                >
                  <Play className="mr-2 h-4 w-4" />
                  Demo
                  {showDemo ? (
                    <ChevronUp className="ml-2 h-4 w-4" />
                  ) : (
                    <ChevronDown className="ml-2 h-4 w-4" />
                  )}
                </Button>

                <Button
                  asChild
                  className="flex-1 h-10 btn btn-primary flex items-center justify-center font-bold"
                >
                  <Link href="/playground">
                    <Gamepad2 className="mr-2 h-4 w-4" />
                    Playground
                  </Link>
                </Button>

                <Button
                  asChild
                  className="flex-1 h-10 btn btn-primary flex items-center justify-center font-bold"
                >
                  <a
                    href="https://github.com/ucbepic/docetl"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Github className="mr-2 h-4 w-4" />
                    GitHub
                  </a>
                </Button>
              </div>

              <div className="flex flex-wrap justify-center gap-4 w-full max-w-xl">
                <Button
                  onClick={toggleVision}
                  className="flex-1 h-10 bg-secondary/70 hover:bg-secondary/60 text-secondary-foreground flex items-center justify-center font-bold"
                >
                  <FileCode className="mr-2 h-4 w-4" />
                  Projects
                  {showVision ? (
                    <ChevronUp className="ml-2 h-4 w-4" />
                  ) : (
                    <ChevronDown className="ml-2 h-4 w-4" />
                  )}
                </Button>
                <Button
                  asChild
                  className="flex-1 h-10 bg-secondary/70 hover:bg-secondary/60 text-secondary-foreground flex items-center justify-center font-bold"
                >
                  <a
                    href="https://discord.gg/fHp7B2X3xx"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <MessageCircle className="mr-2 h-4 w-4" />
                    Discord
                  </a>
                </Button>
                <Button
                  asChild
                  className="flex-1 h-10 bg-secondary/70 hover:bg-secondary/60 text-secondary-foreground flex items-center justify-center font-bold"
                >
                  <Link
                    href="https://ucbepic.github.io/docetl/"
                    target="_blank"
                  >
                    <BookOpen className="mr-2 h-4 w-4" />
                    Docs
                  </Link>
                </Button>
                <Button
                  asChild
                  className="flex-1 h-10 bg-secondary/70 hover:bg-secondary/60 text-secondary-foreground flex items-center justify-center font-bold"
                >
                  <Link href="https://arxiv.org/abs/2410.12189" target="_blank">
                    <FileText className="mr-2 h-4 w-4" />
                    Paper
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </div>

        {showVision && (
          <Card className="max-w-4xl mx-auto bg-white shadow-md">
            <CardContent className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2">
                  Building the Future of AI-Powered Data Systems 🚀
                </h2>
                <p className="text-sm text-muted-foreground mb-6">
                  Traditional databases are great for structured data, but they
                  weren&apos;t built for the Gen AI era. We&apos;re rethinking
                  how data systems should work with LLMs - making them more
                  reliable, cost-effective, and actually usable in production.
                  Here&apos;s what we&apos;re working on:
                </p>

                <div className="grid gap-6">
                  <div className="border rounded-md p-4">
                    <h4 className="font-medium text-primary mb-2">
                      Agentic Query Optimizer
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      Most LLM systems just try to cut costs, but accuracy is
                      the real challenge. Our optimizer uses LLM agents to
                      automatically break down complex operations into smaller,
                      more focused tasks; kind of like having a smart teaching
                      assistant that helps structure your work. Early results
                      show this approach can significantly improve reliability.
                      We are also working on finding plans that are both cheap
                      and accurate.{" "}
                      <a
                        href="https://arxiv.org/abs/2410.12189"
                        className="text-blue-500 hover:underline inline-flex items-center"
                      >
                        Check out our paper <span className="ml-1">→</span>
                      </a>
                    </p>
                  </div>

                  <div className="border rounded-md p-4">
                    <h4 className="font-medium text-primary mb-2">
                      High-Performance Execution Engine
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      Users consistently highlight map operations as the most
                      powerful operationts, but they can get expensive fast -
                      imagine paying for an LLM call on every single document if
                      you have tens of thousands of documents. We&apos;re
                      working on techniques to dramatically reduce these costs
                      without compromising on quality. Approximate query
                      processing will have its comeback!
                    </p>
                  </div>

                  <div className="border rounded-md p-4">
                    <h4 className="font-medium text-primary mb-2">
                      Interactive Playground
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      Prompts are the primary interface between humans and
                      LLM-powered data systems, but crafting them is more art
                      than science. DocWrangler, our IDE, explores new ways to
                      make prompt engineering systematic and intuitive, with
                      interactive tools that help users express their intent
                      clearly and debug unexpected behaviors.{" "}
                      <a
                        href="/playground"
                        className="text-blue-500 hover:underline inline-flex items-center"
                      >
                        Try it yourself <span className="ml-1">→</span>
                      </a>{" "}
                      <span className="text-gray-500">
                        (Paper coming January 2025)
                      </span>
                    </p>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground mt-6">
                  We&apos;re working with universities, governments, and
                  organizations to solve real-world data challenges - especially
                  for teams without ML expertise. Want to use DocETL for your
                  project or be part of our case studies? Drop a line at{" "}
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
