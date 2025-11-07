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
import { useChat } from "ai/react";
import { Send, Loader2, Code2, Database } from "lucide-react";
import ResizableDataTable, {
  ColumnType,
} from "@/components/ResizableDataTable";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";

interface CodeExecution {
  id: string;
  code: string;
  status: "running" | "completed" | "error";
  output?: string;
  error?: string;
  timestamp: Date;
}

export default function ScraperPage() {
  const [userQuery, setUserQuery] = useState("");
  const [schema, setSchema] = useState("");
  const [dataset, setDataset] = useState<Record<string, unknown>[]>([]);
  const [codeExecutions, setCodeExecutions] = useState<CodeExecution[]>([]);
  const [isStarting, setIsStarting] = useState(false);
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
  } = useChat({
    api: "/api/scraper",
    body: {
      userQuery,
      schema: schema || undefined,
    },
    onFinish: (message) => {
      // Parse tool calls and results from messages
      // Tool results are included in assistant messages
      const lastMessage = messages[messages.length - 1];
      if (lastMessage?.toolInvocations) {
        lastMessage.toolInvocations.forEach((invocation: any) => {
          if (invocation.toolName === "execute_code") {
            setCodeExecutions((prev) => {
              const existing = prev.find((e) => e.id === invocation.toolCallId);
              if (existing) {
                return prev.map((e) =>
                  e.id === invocation.toolCallId
                    ? {
                        ...e,
                        status: invocation.state === "result" ? "completed" : "running",
                        output: invocation.result?.output || "",
                        error: invocation.result?.error || undefined,
                      }
                    : e
                );
              }
              return [
                ...prev,
                {
                  id: invocation.toolCallId,
                  code: invocation.args.code as string,
                  status: invocation.state === "result" ? "completed" : "running",
                  output: invocation.result?.output || "",
                  error: invocation.result?.error || undefined,
                  timestamp: new Date(),
                },
              ];
            });
          } else if (invocation.toolName === "read_json" && invocation.result?.data) {
            const data = invocation.result.data;
            if (Array.isArray(data)) {
              setDataset(data);
            }
          }
        });
      }
    },
    experimental_onToolCall: async ({ toolCall }) => {
      // Handle tool calls when they're invoked
      if (toolCall.toolName === "execute_code") {
        const executionId = toolCall.toolCallId;
        setCodeExecutions((prev) => [
          ...prev,
          {
            id: executionId,
            code: toolCall.args.code as string,
            status: "running",
            timestamp: new Date(),
          },
        ]);
      }
      // Return empty result - the actual result will come via onFinish
      return {};
    },
  });

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    codeEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [codeExecutions]);

  const handleStartScraping = async () => {
    if (!userQuery.trim()) return;

    setIsStarting(true);
    setMessages([]);
    setCodeExecutions([]);
    setDataset([]);

    // Use the useChat hook's append method to send the initial message
    const initialMessage = `User Query: ${userQuery}\n\nSchema: ${schema || "No schema specified - infer from query"}`;
    
    // Append will trigger the API call and handle streaming
    await append({
      role: "user",
      content: initialMessage,
    });

    setIsStarting(false);
  };

  // Generate columns from dataset schema
  const columns: ColumnType<Record<string, unknown>>[] = React.useMemo(() => {
    if (dataset.length === 0) return [];

    const keys = Object.keys(dataset[0]);
    return keys.map((key) => ({
      accessorKey: key,
      header: key,
      initialWidth: 200,
    }));
  }, [dataset]);

  return (
    <div className="h-screen flex flex-col bg-background">
      <div className="border-b p-4 bg-card">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold">Web Scraper</h1>
          <div className="flex-1 flex gap-4">
            <div className="flex-1">
              <Label htmlFor="query" className="text-sm mb-2 block">
                What data do you want to scrape?
              </Label>
              <Textarea
                id="query"
                placeholder="e.g., Find all restaurants in San Francisco with ratings above 4.5 stars..."
                value={userQuery}
                onChange={(e) => setUserQuery(e.target.value)}
                className="min-h-[60px]"
                disabled={isLoading || isStarting}
              />
            </div>
            <div className="flex-1">
              <Label htmlFor="schema" className="text-sm mb-2 block">
                Schema (optional - JSON format)
              </Label>
              <Textarea
                id="schema"
                placeholder='e.g., {"name": "string", "rating": "float", "address": "string"}'
                value={schema}
                onChange={(e) => setSchema(e.target.value)}
                className="min-h-[60px] font-mono text-xs"
                disabled={isLoading || isStarting}
              />
            </div>
            <div className="flex items-end">
              <Button
                onClick={handleStartScraping}
                disabled={!userQuery.trim() || isLoading || isStarting}
                className="h-[60px]"
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
          </div>
        </div>
      </div>

      <ResizablePanelGroup direction="horizontal" className="flex-1">
        {/* Left Panel - Chat */}
        <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
          <div className="h-full flex flex-col border-r">
            <div className="p-4 border-b bg-muted/50">
              <h2 className="font-semibold flex items-center gap-2">
                <Database className="h-4 w-4" />
                Agent Chat
              </h2>
            </div>
            <ScrollArea className="flex-1 p-4">
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={cn(
                      "rounded-lg p-3",
                      message.role === "user"
                        ? "bg-primary text-primary-foreground ml-auto max-w-[80%]"
                        : "bg-muted mr-auto max-w-[80%]"
                    )}
                  >
                    <ReactMarkdown className="text-sm prose prose-sm max-w-none">
                      {message.content}
                    </ReactMarkdown>
                  </div>
                ))}
                {(isLoading || isStarting) && (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm">Agent is working...</span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
            </ScrollArea>
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
                <Button type="submit" disabled={!input.trim() || isLoading || isStarting}>
                  <Send className="h-4 w-4" />
                </Button>
              </form>
            </div>
          </div>
        </ResizablePanel>

        <ResizableHandle withHandle />

        {/* Middle Panel - Code Execution */}
        <ResizablePanel defaultSize={30} minSize={20} maxSize={40}>
          <div className="h-full flex flex-col border-r">
            <div className="p-4 border-b bg-muted/50">
              <h2 className="font-semibold flex items-center gap-2">
                <Code2 className="h-4 w-4" />
                Code Execution
              </h2>
            </div>
            <ScrollArea className="flex-1 p-4">
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
                      <pre className="text-xs bg-muted p-3 rounded overflow-x-auto mb-2">
                        <code>{execution.code}</code>
                      </pre>
                      {execution.output && (
                        <div className="text-xs bg-muted/50 p-2 rounded">
                          <strong>Output:</strong>
                          <pre className="mt-1 whitespace-pre-wrap">
                            {execution.output}
                          </pre>
                        </div>
                      )}
                      {execution.error && (
                        <div className="text-xs bg-red-50 text-red-800 p-2 rounded">
                          <strong>Error:</strong>
                          <pre className="mt-1 whitespace-pre-wrap">
                            {execution.error}
                          </pre>
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

        {/* Right Panel - Dataset Table */}
        <ResizablePanel defaultSize={45} minSize={30}>
          <div className="h-full flex flex-col">
            <div className="p-4 border-b bg-muted/50">
              <h2 className="font-semibold flex items-center gap-2">
                <Database className="h-4 w-4" />
                Dataset ({dataset.length} rows)
              </h2>
            </div>
            <div className="flex-1 overflow-auto p-4">
              {dataset.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p className="text-sm">
                    Scraped data will appear here as the agent collects it...
                  </p>
                </div>
              ) : (
                <ResizableDataTable data={dataset} columns={columns} />
              )}
            </div>
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
