"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChat } from "@ai-sdk/react";
import {
  Send,
  Loader2,
  Code2,
  Database,
  Scroll,
  Home,
  Square,
  Link as LinkIcon,
  Download,
  RefreshCcw,
} from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import { Parser } from "json2csv";

interface CodeExecution {
  id: string;
  code: string;
  status: "running" | "completed" | "error";
  output?: string;
  error?: string;
  timestamp: Date;
}

interface FoundLink {
  title: string;
  url: string;
  snippet: string;
  score?: number;
  timestamp: Date;
}

export default function ScraperPage() {
  const [userQuery, setUserQuery] = useState("");
  const [schema, setSchema] = useState("");
  const [dataset, setDataset] = useState<Record<string, unknown>[]>([]);
  const [codeExecutions, setCodeExecutions] = useState<CodeExecution[]>([]);
  const [foundLinks, setFoundLinks] = useState<FoundLink[]>([]);
  const [isStarting, setIsStarting] = useState(false);
  const [isRefiningDataset, setIsRefiningDataset] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const codeEndRef = useRef<HTMLDivElement>(null);

  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    setMessages,
    append,
    stop,
  } = useChat({
    api: "/api/scraper",
    body: {
      userQuery,
      schema: schema || undefined,
      sessionId: sessionId || undefined,
    },
    onFinish: async () => {
      // Reset isStarting when the stream finishes
      setIsStarting(false);
      // Clear refining state when agent finishes
      setIsRefiningDataset(false);
      // Automatically fetch the dataset when agent finishes
      if (sessionId) {
        try {
          const response = await fetch(
            `/api/scraper/dataset?sessionId=${sessionId}`
          );
          const result = await response.json();
          if (result.data && Array.isArray(result.data)) {
            const typedData = result.data.filter(
              (item: unknown): item is Record<string, unknown> =>
                typeof item === "object" &&
                item !== null &&
                !Array.isArray(item)
            );
            setDataset(typedData);
          }
        } catch (error) {
          console.error("Failed to fetch dataset:", error);
        }
      }
    },
    onError: () => {
      // Reset isStarting on error
      setIsStarting(false);
      // Clear refining state on error
      setIsRefiningDataset(false);
    },
  });

  // Extract tool invocations from all messages - process incrementally for streaming
  useEffect(() => {
    // Process all messages to extract tool invocations
    messages.forEach((message) => {
      if (message.toolInvocations) {
        message.toolInvocations.forEach(
          (invocation: {
            toolCallId: string;
            toolName: string;
            state?: string;
            args?: { code?: string };
            result?: {
              output?: string;
              error?: string;
              data?: unknown[];
              results?: Array<{
                title?: string;
                url?: string;
                snippet?: string;
                content?: string;
                score?: number;
              }>;
            };
          }) => {
            const toolCallId = invocation.toolCallId;

            if (invocation.toolName === "execute_code") {
              setCodeExecutions((prev) => {
                const existing = prev.find((e) => e.id === toolCallId);
                const isCompleted = invocation.state === "result";

                if (existing) {
                  // Update existing execution - allow incremental updates
                  if (isCompleted) {
                    setIsRefiningDataset(true);
                  }
                  return prev.map((e) =>
                    e.id === toolCallId
                      ? {
                          ...e,
                          status: isCompleted ? "completed" : "running",
                          output: invocation.result?.output || e.output || "",
                          error:
                            invocation.result?.error || e.error || undefined,
                          // Update code if we have partial code from streaming
                          code:
                            (invocation.args?.code as string) || e.code || "",
                        }
                      : e
                  );
                }
                // Add new execution - create immediately even if not completed
                if (isCompleted) {
                  setIsRefiningDataset(true);
                }
                return [
                  ...prev,
                  {
                    id: toolCallId,
                    code: (invocation.args?.code as string) || "",
                    status: isCompleted ? "completed" : "running",
                    output: invocation.result?.output || "",
                    error: invocation.result?.error || undefined,
                    timestamp: new Date(),
                  },
                ];
              });
            } else if (invocation.toolName === "read_json") {
              // Handle read_json results - check multiple possible result structures
              // Process regardless of state (partial or complete results)
              const result = invocation.result;
              let data: unknown[] | null = null;

              if (result) {
                // Try different possible structures
                if (Array.isArray(result.data)) {
                  data = result.data;
                } else if (Array.isArray(result)) {
                  data = result;
                } else if (typeof result === "object" && "data" in result) {
                  const resultData = (result as { data?: unknown }).data;
                  if (Array.isArray(resultData)) {
                    data = resultData;
                  }
                }
              }

              if (data && Array.isArray(data) && data.length > 0) {
                // Ensure data is properly typed
                const typedData = data.filter(
                  (item): item is Record<string, unknown> =>
                    typeof item === "object" &&
                    item !== null &&
                    !Array.isArray(item)
                );
                if (typedData.length > 0) {
                  setDataset(typedData);
                  // Dataset refreshed - clear refining state
                  setIsRefiningDataset(false);
                }
              } else if (data && Array.isArray(data) && data.length === 0) {
                // Empty array - still update to clear any stale data
                setDataset([]);
                setIsRefiningDataset(false);
              }
            } else if (invocation.toolName === "search_internet") {
              // Process results even if partial (for streaming)
              const results = invocation.result?.results;
              if (Array.isArray(results) && results.length > 0) {
                const newLinks: FoundLink[] = results.map((result) => ({
                  title: result.title || "Untitled",
                  url: result.url || "",
                  snippet: result.snippet || result.content || "",
                  score: result.score,
                  timestamp: new Date(),
                }));
                setFoundLinks((prev) => {
                  // Normalize URLs for better deduplication
                  const normalizeUrl = (url: string) => {
                    try {
                      const urlObj = new URL(url);
                      // Remove trailing slashes and normalize
                      return (
                        urlObj.origin +
                        urlObj.pathname.replace(/\/$/, "") +
                        urlObj.search +
                        urlObj.hash
                      );
                    } catch {
                      // If URL parsing fails, just normalize the string
                      return url.replace(/\/$/, "").toLowerCase();
                    }
                  };
                  const existingUrls = new Set(
                    prev.map((l) => normalizeUrl(l.url))
                  );
                  const uniqueNewLinks = newLinks.filter((link) => {
                    const normalizedUrl = normalizeUrl(link.url);
                    return !existingUrls.has(normalizedUrl);
                  });
                  return [...prev, ...uniqueNewLinks];
                });
              }
            }
          }
        );
      }
    });
  }, [messages]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    codeEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [codeExecutions]);

  // Clear refining state when agent stops loading (user's turn)
  useEffect(() => {
    if (!isLoading && !isStarting) {
      // Check if there are any running code executions
      const hasRunningExecutions = codeExecutions.some(
        (e) => e.status === "running"
      );
      if (!hasRunningExecutions) {
        setIsRefiningDataset(false);
      }
    }
  }, [isLoading, isStarting, codeExecutions]);

  const handleStartScraping = async () => {
    if (!userQuery.trim()) return;

    // Generate a new session ID for this scraping session with date prefix
    // Format: YYYYMMDD-{uuid} (e.g., 20250130-abc123-def456-...)
    const now = new Date();
    const dateStr = now.toISOString().split("T")[0].replace(/-/g, ""); // YYYYMMDD
    const uuid = crypto.randomUUID();
    const newSessionId = `${dateStr}-${uuid}`;
    setSessionId(newSessionId);

    setIsStarting(true);
    setMessages([]);
    setCodeExecutions([]);
    setDataset([]);
    setFoundLinks([]);
    setIsRefiningDataset(false);

    // Use the useChat hook's append method to send the initial message
    const initialMessage = `User Query: ${userQuery}\n\nSchema: ${
      schema || "No schema specified - infer from query"
    }`;

    // Append will trigger the API call and handle streaming
    // isStarting will be reset in onFinish callback
    await append({
      role: "user",
      content: initialMessage,
    });
  };

  // Download functions
  const downloadJSON = () => {
    if (dataset.length === 0) return;
    const jsonContent = JSON.stringify(dataset, null, 2);
    const blob = new Blob([jsonContent], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "dataset.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  const downloadCSV = () => {
    if (dataset.length === 0) return;
    try {
      const parser = new Parser();
      const csvContent = parser.parse(dataset);
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "dataset.csv";
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Error converting to CSV:", err);
    }
  };

  // Fetch dataset from backend
  const [isRefreshingDataset, setIsRefreshingDataset] = useState(false);
  const refreshDataset = async () => {
    if (!sessionId) {
      console.warn("No session ID available");
      return;
    }
    setIsRefreshingDataset(true);
    try {
      const response = await fetch(
        `/api/scraper/dataset?sessionId=${sessionId}`
      );
      const result = await response.json();
      if (result.data && Array.isArray(result.data)) {
        const typedData = result.data.filter(
          (item: unknown): item is Record<string, unknown> =>
            typeof item === "object" && item !== null && !Array.isArray(item)
        );
        setDataset(typedData);
        console.log(`Dataset refreshed: ${typedData.length} items`);
      } else {
        console.warn("No data in response:", result);
      }
    } catch (error) {
      console.error("Failed to refresh dataset:", error);
    } finally {
      setIsRefreshingDataset(false);
    }
  };

  // Get column keys from dataset
  const columnKeys = React.useMemo(() => {
    if (dataset.length === 0) return [];
    return Object.keys(dataset[0]);
  }, [dataset]);

  const hasMessages = messages.length > 0;

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header matching DocWrangler style */}
      <div className="px-8 py-4 flex justify-between items-center border-b bg-white shadow-sm relative">
        <a
          href="https://docetl.org"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-sm text-gray-600 hover:text-primary transition-colors"
        >
          <Home size={14} />
          <span>Back to DocETL</span>
        </a>
        <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center gap-2">
          <Scroll className="text-primary" size={20} />
          <h1 className="text-lg font-bold text-primary">DocScraper</h1>
        </div>
        <div className="w-[120px]"></div>
      </div>

      <ResizablePanelGroup direction="horizontal" className="flex-1">
        {/* Left Panel - Chat */}
        <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
          <div className="h-full flex flex-col border-r">
            <div className="p-4">
              <h2 className="font-semibold flex items-center gap-2">
                <Database className="h-4 w-4" />
                Agent Chat
              </h2>
            </div>
            <ScrollArea className="flex-1 p-4" scrollbarOrientation="both">
              <div className="space-y-4">
                {/* Show form when no messages */}
                {!hasMessages && (
                  <div className="space-y-4">
                    <div>
                      <Label
                        htmlFor="query"
                        className="text-sm mb-2 block font-medium"
                      >
                        What data do you want to scrape?
                      </Label>
                      <Textarea
                        id="query"
                        placeholder="e.g., Scrape travel blog posts about hidden gems in Japan from the past year, find food blog posts about best restaurants in Lisbon, or collect adventure travel stories about hiking trails in Patagonia..."
                        value={userQuery}
                        onChange={(e) => setUserQuery(e.target.value)}
                        className="min-h-[80px]"
                        disabled={isLoading || isStarting}
                      />
                    </div>
                    <div>
                      <Label
                        htmlFor="schema"
                        className="text-sm mb-2 block font-medium"
                      >
                        Attributes you want your dataset to have (optional)
                      </Label>
                      <Textarea
                        id="schema"
                        placeholder="e.g., url, scraped_text, publication_date"
                        value={schema}
                        onChange={(e) => setSchema(e.target.value)}
                        className="min-h-[60px]"
                        disabled={isLoading || isStarting}
                      />
                      <button
                        type="button"
                        onClick={() => {
                          setUserQuery(
                            "Scrape travel blog posts about hidden gem destinations in Japan that locals recommend, published within the last year"
                          );
                          setSchema("url, scraped_text, author");
                        }}
                        className="text-xs text-muted-foreground hover:text-foreground mt-1 underline"
                        disabled={isLoading || isStarting}
                      >
                        Use example: hidden gems in Japan
                      </button>
                    </div>
                    <Button
                      onClick={handleStartScraping}
                      disabled={!userQuery.trim() || isLoading || isStarting}
                      className="w-full"
                    >
                      {isStarting || isLoading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Scraping...
                        </>
                      ) : (
                        "Start Scraping"
                      )}
                    </Button>
                  </div>
                )}

                {/* Show messages when chat has started */}
                {hasMessages &&
                  messages.map((message, index) => {
                    // Show messages with content OR tool invocations
                    const hasContent =
                      message.content && message.content.trim() !== "";
                    const hasToolInvocations =
                      message.toolInvocations &&
                      message.toolInvocations.length > 0;

                    // Skip if message has neither content nor tool invocations
                    if (!hasContent && !hasToolInvocations) {
                      return null;
                    }

                    return (
                      <div
                        key={index}
                        className={cn(
                          "rounded-lg p-3",
                          message.role === "user"
                            ? "bg-primary text-primary-foreground ml-auto max-w-[80%]"
                            : "bg-muted mr-auto max-w-[80%]"
                        )}
                      >
                        {hasContent && (
                          <ReactMarkdown className="text-sm prose prose-sm max-w-none">
                            {message.content}
                          </ReactMarkdown>
                        )}
                        {hasToolInvocations && message.role === "assistant" && (
                          <div className="mt-2 space-y-1">
                            {message.toolInvocations.map(
                              (
                                invocation: {
                                  toolName?: string;
                                  state?: string;
                                  result?: { message?: string };
                                },
                                idx: number
                              ) => {
                                const toolName =
                                  invocation.toolName || "unknown";
                                const state = invocation.state || "unknown";
                                const isRunning = state === "partial";

                                // Format tool name for display
                                const formatToolName = (name: string) => {
                                  const names: Record<string, string> = {
                                    search_internet: "Search",
                                    execute_code: "Execute Code",
                                    read_json: "Read Dataset",
                                  };
                                  return names[name] || name;
                                };

                                return (
                                  <div
                                    key={idx}
                                    className="flex items-center gap-2 text-xs text-muted-foreground"
                                  >
                                    {isRunning && (
                                      <Loader2 className="h-3 w-3 animate-spin shrink-0" />
                                    )}
                                    <span className="font-medium text-foreground/70 shrink-0">
                                      {formatToolName(toolName)}
                                    </span>
                                    {invocation.result?.message && (
                                      <span className="flex-1">
                                        {invocation.result.message}
                                      </span>
                                    )}
                                  </div>
                                );
                              }
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                {isLoading && (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm">Agent is working...</span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
            </ScrollArea>
            {hasMessages && (
              <div className="p-4 border-t">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (input.trim() && !isLoading) {
                      handleSubmit(e);
                    }
                  }}
                  className="flex gap-2"
                >
                  <Input
                    value={input}
                    onChange={handleInputChange}
                    placeholder="Ask the agent..."
                    disabled={isLoading || isStarting}
                    className="flex-1"
                  />
                  {isLoading || isStarting ? (
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => {
                        stop();
                        setIsStarting(false);
                      }}
                    >
                      <Square className="h-4 w-4" />
                    </Button>
                  ) : (
                    <Button
                      type="submit"
                      disabled={!input.trim() || isLoading || isStarting}
                    >
                      <Send className="h-4 w-4" />
                    </Button>
                  )}
                </form>
              </div>
            )}
          </div>
        </ResizablePanel>

        <ResizableHandle withHandle />

        {/* Links Panel */}
        <ResizablePanel defaultSize={20} minSize={15} maxSize={30}>
          <div className="h-full flex flex-col border-r">
            <div className="p-4">
              <h2 className="font-semibold flex items-center gap-2">
                <LinkIcon className="h-4 w-4" />
                Found Links ({foundLinks.length})
              </h2>
            </div>
            <ScrollArea
              className="flex-1 p-4 [&>[data-radix-scroll-area-viewport]]:overflow-x-visible"
              scrollbarOrientation="both"
            >
              <div className="space-y-3 w-full min-w-0">
                {foundLinks.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    <LinkIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p className="text-sm">
                      Links found from searches will appear here...
                    </p>
                  </div>
                ) : (
                  foundLinks.map((link, index) => (
                    <div
                      key={index}
                      className="border rounded-lg p-3 bg-card hover:bg-muted/50 transition-colors w-full min-w-0"
                    >
                      <a
                        href={link.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block w-full min-w-0"
                      >
                        <div className="flex items-start justify-between gap-2 mb-1">
                          <h3 className="font-medium text-sm line-clamp-2 hover:text-primary transition-colors flex-1 min-w-0">
                            {link.title}
                          </h3>
                          {link.score !== undefined && (
                            <span className="text-xs text-muted-foreground whitespace-nowrap shrink-0">
                              {Math.round(link.score * 100)}%
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground line-clamp-2 mb-1">
                          {link.snippet}
                        </p>
                        <div
                          className="w-full mb-1"
                          style={{ width: "100%", overflow: "hidden" }}
                        >
                          <div
                            className="text-xs text-blue-600 whitespace-nowrap"
                            style={{
                              scrollbarWidth: "thin",
                              width: "100%",
                              overflowX: "auto",
                              overflowY: "hidden",
                              display: "block",
                              WebkitOverflowScrolling: "touch",
                            }}
                            title={link.url}
                          >
                            <span
                              style={{
                                display: "inline-block",
                                whiteSpace: "nowrap",
                              }}
                            >
                              {link.url}
                            </span>
                          </div>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {link.timestamp.toLocaleTimeString()}
                        </span>
                      </a>
                    </div>
                  ))
                )}
              </div>
            </ScrollArea>
          </div>
        </ResizablePanel>

        <ResizableHandle withHandle />

        {/* Right Panel - Code Execution and Dataset stacked vertically */}
        <ResizablePanel defaultSize={30} minSize={25}>
          <ResizablePanelGroup direction="vertical" className="h-full">
            {/* Code Execution Panel - Top Half */}
            <ResizablePanel defaultSize={50} minSize={30}>
              <div className="h-full flex flex-col">
                <div className="p-4 border-b">
                  <h2 className="font-semibold flex items-center gap-2">
                    <Code2 className="h-4 w-4" />
                    Code Execution
                  </h2>
                </div>
                <ScrollArea className="flex-1 p-4" scrollbarOrientation="both">
                  <div className="space-y-4">
                    {codeExecutions.length === 0 ? (
                      <div className="text-center text-muted-foreground py-8">
                        <Code2 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                        <p className="text-sm">
                          Code executions will appear here as the agent works...
                        </p>
                      </div>
                    ) : (
                      codeExecutions.map((execution) => (
                        <div
                          key={execution.id}
                          className="border rounded-lg p-4 bg-card"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-muted-foreground">
                              {execution.timestamp.toLocaleTimeString()}
                            </span>
                            <span
                              className={cn(
                                "text-xs px-2 py-1 rounded",
                                execution.status === "running" &&
                                  "bg-blue-100 text-blue-800",
                                execution.status === "completed" &&
                                  "bg-green-100 text-green-800",
                                execution.status === "error" &&
                                  "bg-red-100 text-red-800"
                              )}
                            >
                              {execution.status}
                            </span>
                          </div>
                          <div
                            className="text-xs bg-muted p-3 rounded mb-2 w-full overflow-x-auto overflow-y-hidden"
                            style={{ scrollbarWidth: "thin" }}
                          >
                            <pre className="whitespace-pre">
                              <code className="block min-w-max">
                                {execution.code}
                              </code>
                            </pre>
                          </div>
                          {execution.output && (
                            <div className="text-xs bg-muted/50 p-2 rounded max-h-96 overflow-y-auto overflow-x-hidden">
                              <strong>Output:</strong>
                              <pre className="mt-1 whitespace-pre-wrap break-words">
                                {execution.output}
                              </pre>
                            </div>
                          )}
                          {execution.error && (
                            <div className="text-xs bg-red-50 text-red-800 p-2 rounded max-h-96 overflow-y-auto overflow-x-hidden">
                              <strong>Error:</strong>
                              <pre className="mt-1 whitespace-pre-wrap break-words">
                                {execution.error}
                              </pre>
                            </div>
                          )}
                          {execution.status === "running" &&
                            !execution.output && (
                              <div className="text-xs text-muted-foreground italic">
                                Code is executing... output will appear here
                                when complete.
                              </div>
                            )}
                        </div>
                      ))
                    )}
                    <div ref={codeEndRef} />
                  </div>
                </ScrollArea>
              </div>
            </ResizablePanel>

            <ResizableHandle withHandle />

            {/* Dataset Panel - Bottom Half */}
            <ResizablePanel defaultSize={50} minSize={30}>
              <div className="h-full flex flex-col">
                <div className="p-4 border-b flex items-center justify-between">
                  <h2 className="font-semibold flex items-center gap-2">
                    <Database className="h-4 w-4" />
                    Dataset ({dataset.length} rows)
                    {isRefiningDataset && (
                      <Loader2 className="h-4 w-4 animate-spin text-primary ml-2" />
                    )}
                  </h2>
                  <div className="flex gap-2">
                    {sessionId && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={refreshDataset}
                        disabled={isRefreshingDataset}
                        className="h-8"
                      >
                        <RefreshCcw
                          className={cn(
                            "h-3 w-3 mr-1",
                            isRefreshingDataset && "animate-spin"
                          )}
                        />
                        Refresh
                      </Button>
                    )}
                    {dataset.length > 0 && (
                      <>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={downloadJSON}
                          className="h-8"
                        >
                          <Download className="h-3 w-3 mr-1" />
                          JSON
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={downloadCSV}
                          className="h-8"
                        >
                          <Download className="h-3 w-3 mr-1" />
                          CSV
                        </Button>
                      </>
                    )}
                  </div>
                </div>
                <div className="flex-1 overflow-auto">
                  {dataset.length === 0 ? (
                    <div className="text-center text-muted-foreground py-8 px-4">
                      <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p className="text-sm">
                        Scraped data will appear here as the agent collects
                        it...
                      </p>
                    </div>
                  ) : (
                    <div className="p-4">
                      <div className="border rounded-lg overflow-x-auto">
                        <table className="min-w-full text-sm">
                          <thead className="bg-muted/50">
                            <tr>
                              {columnKeys.map((key) => (
                                <th
                                  key={key}
                                  className="px-4 py-2 text-left font-medium text-muted-foreground border-b"
                                  style={{
                                    width: "200px",
                                    minWidth: "200px",
                                    maxWidth: "200px",
                                  }}
                                >
                                  <div className="overflow-x-auto overflow-y-hidden whitespace-nowrap">
                                    {key}
                                  </div>
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {dataset.map((row, rowIndex) => (
                              <tr
                                key={rowIndex}
                                className="border-b hover:bg-muted/30 transition-colors"
                              >
                                {columnKeys.map((key) => (
                                  <td
                                    key={key}
                                    className="px-4 py-2 border-r last:border-r-0"
                                    style={{
                                      width: "200px",
                                      minWidth: "200px",
                                      maxWidth: "200px",
                                    }}
                                  >
                                    <div
                                      className="w-full overflow-x-auto overflow-y-hidden"
                                      style={{
                                        scrollbarWidth: "thin",
                                        maxHeight: "100px",
                                      }}
                                      title={
                                        row[key] !== null &&
                                        row[key] !== undefined
                                          ? String(row[key])
                                          : ""
                                      }
                                    >
                                      <div className="whitespace-nowrap min-w-max py-1">
                                        {row[key] !== null &&
                                        row[key] !== undefined
                                          ? String(row[key])
                                          : ""}
                                      </div>
                                    </div>
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
