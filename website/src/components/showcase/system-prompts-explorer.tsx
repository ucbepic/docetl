"use client";

import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface SystemExample {
  system: string;
  example: string;
}

interface StrategyData {
  strategy: string;
  summary: string;
  systems_and_examples: SystemExample[];
  _counts_prereduce_summarize_strategies: number;
}

export default function SystemPromptsExplorer() {
  const [isLoading, setIsLoading] = useState(true);
  const [strategies, setStrategies] = useState<StrategyData[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedSystem, setSelectedSystem] = useState<string>("all");
  const [systemsList, setSystemsList] = useState<string[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyData | null>(
    null
  );

  useEffect(() => {
    async function fetchStrategies() {
      try {
        const response = await fetch(
          "https://docetl.blob.core.windows.net/demos/analyzed_strategies.json"
        );
        if (!response.ok) {
          throw new Error("Failed to fetch data");
        }
        const data: StrategyData[] = await response.json();
        setStrategies(data);

        // Extract unique systems
        const systems = new Set<string>();
        data.forEach((strategy) => {
          strategy.systems_and_examples.forEach((example) => {
            systems.add(example.system);
          });
        });
        setSystemsList(Array.from(systems).sort());

        if (data.length > 0) {
          setSelectedStrategy(data[0]);
        }

        setIsLoading(false);
      } catch (error) {
        console.error("Error fetching strategies:", error);
        setIsLoading(false);
      }
    }

    fetchStrategies();
  }, []);

  // Filter strategies based on search term and selected system
  const filteredStrategies = strategies.filter((strategy) => {
    const matchesSearch =
      strategy.strategy.toLowerCase().includes(searchTerm.toLowerCase()) ||
      strategy.summary.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesSystem =
      selectedSystem === "all" ||
      strategy.systems_and_examples.some(
        (example) => example.system === selectedSystem
      );

    return matchesSearch && matchesSystem;
  });

  // Sort strategies by frequency of examples (descending)
  const sortedStrategies = [...filteredStrategies].sort(
    (a, b) => b.systems_and_examples.length - a.systems_and_examples.length
  );

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Only handle arrow keys when a strategy is already selected
      if (!selectedStrategy) return;

      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault(); // Prevent default scrolling

        const currentIndex = sortedStrategies.findIndex(
          (s) => s.strategy === selectedStrategy.strategy
        );

        if (currentIndex !== -1) {
          let newIndex;

          if (e.key === "ArrowDown") {
            // Move to next strategy (if not at end)
            newIndex = Math.min(currentIndex + 1, sortedStrategies.length - 1);
          } else {
            // Move to previous strategy (if not at beginning)
            newIndex = Math.max(currentIndex - 1, 0);
          }

          // Only update if index actually changed
          if (newIndex !== currentIndex) {
            setSelectedStrategy(sortedStrategies[newIndex]);

            // Find and scroll the item into view
            const element = document.getElementById(
              `strategy-item-${newIndex}`
            );
            if (element) {
              element.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
          }
        }
      }
    }

    // Add event listener
    window.addEventListener("keydown", handleKeyDown);

    // Clean up
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [selectedStrategy, sortedStrategies]); // Dependencies

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-12">
        <Loader2 className="h-8 w-8 animate-spin mr-2" />
        <span>Loading strategies data...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="flex-1">
          <Label htmlFor="search-strategies">Search Strategies</Label>
          <Input
            id="search-strategies"
            placeholder="Search by strategy name or description..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="w-full sm:w-64">
          <Label htmlFor="filter-system">Filter by System</Label>
          <Select value={selectedSystem} onValueChange={setSelectedSystem}>
            <SelectTrigger id="filter-system">
              <SelectValue placeholder="All Systems" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Systems</SelectItem>
              {systemsList.map((system) => (
                <SelectItem key={system} value={system}>
                  {system}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-1">
          <Card className="h-[650px] flex flex-col">
            <CardHeader>
              <CardTitle className="text-lg">Strategies</CardTitle>
              <CardDescription>
                {sortedStrategies.length} strategies found
              </CardDescription>
            </CardHeader>
            <ScrollArea className="flex-1 px-4 pb-4">
              <div className="space-y-2">
                {sortedStrategies.map((strategy, index) => (
                  <div
                    id={`strategy-item-${index}`}
                    key={strategy.strategy}
                    className={`p-3 rounded-md cursor-pointer hover:bg-muted transition-colors ${
                      selectedStrategy?.strategy === strategy.strategy
                        ? "bg-primary/10 font-medium"
                        : ""
                    }`}
                    onClick={() => setSelectedStrategy(strategy)}
                  >
                    <div className="flex justify-between items-start">
                      <h3 className="font-medium text-sm">
                        {strategy.strategy}
                      </h3>
                      <Badge variant="secondary" className="ml-2">
                        {strategy.systems_and_examples.length}
                      </Badge>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                      {strategy.summary.substring(0, 100)}...
                    </p>
                  </div>
                ))}
                {sortedStrategies.length === 0 && (
                  <div className="p-4 text-center text-muted-foreground">
                    No strategies found matching your filters
                  </div>
                )}
              </div>
            </ScrollArea>
          </Card>
        </div>

        <div className="md:col-span-2">
          {selectedStrategy ? (
            <Card className="h-[650px] overflow-hidden flex flex-col">
              <div className="px-6 pt-6 pb-2">
                <h2 className="text-xl font-semibold">
                  {selectedStrategy.strategy}
                </h2>
                {Array.isArray(selectedStrategy.systems_and_examples) &&
                selectedStrategy.systems_and_examples.length > 0 ? (
                  <p className="text-sm text-muted-foreground mt-1">
                    Implemented in{" "}
                    {selectedStrategy.systems_and_examples.length} system(s)
                  </p>
                ) : (
                  <p className="text-sm text-muted-foreground mt-1">
                    No implementation examples found
                  </p>
                )}
              </div>

              <div className="flex-1 overflow-hidden px-6">
                <Tabs defaultValue="summary" className="h-full flex flex-col">
                  <TabsList className="mb-4">
                    <TabsTrigger value="summary">Summary</TabsTrigger>
                    {Array.isArray(selectedStrategy.systems_and_examples) &&
                      selectedStrategy.systems_and_examples.length > 0 && (
                        <TabsTrigger value="examples">
                          Examples (
                          {selectedStrategy.systems_and_examples.length})
                        </TabsTrigger>
                      )}
                  </TabsList>

                  <TabsContent
                    value="summary"
                    className="mt-0 flex-1 overflow-y-auto data-[state=inactive]:hidden"
                  >
                    <div className="prose prose-sm max-w-none prose-headings:font-semibold prose-h1:text-xl prose-h2:text-lg prose-h3:text-base">
                      <ReactMarkdown
                        components={{
                          h1: ({ node, ...props }) => (
                            <h1
                              className="text-xl font-bold mt-6 mb-4"
                              {...props}
                            />
                          ),
                          h2: ({ node, ...props }) => (
                            <h2
                              className="text-lg font-bold mt-5 mb-3"
                              {...props}
                            />
                          ),
                          h3: ({ node, ...props }) => (
                            <h3
                              className="text-base font-semibold mt-4 mb-2"
                              {...props}
                            />
                          ),
                          ul: ({ node, ...props }) => (
                            <ul className="list-disc pl-5 my-3" {...props} />
                          ),
                          ol: ({ node, ...props }) => (
                            <ol className="list-decimal pl-5 my-3" {...props} />
                          ),
                          li: ({ node, ...props }) => (
                            <li className="mb-1" {...props} />
                          ),
                          p: ({ node, ...props }) => (
                            <p className="mb-4" {...props} />
                          ),
                        }}
                      >
                        {selectedStrategy.summary}
                      </ReactMarkdown>
                    </div>
                  </TabsContent>

                  {Array.isArray(selectedStrategy.systems_and_examples) &&
                    selectedStrategy.systems_and_examples.length > 0 && (
                      <TabsContent
                        value="examples"
                        className="mt-0 flex-1 overflow-y-auto data-[state=inactive]:hidden"
                      >
                        <div className="space-y-4 pb-6">
                          {selectedStrategy.systems_and_examples.map(
                            (example, index) => (
                              <div
                                key={index}
                                className="border rounded-lg p-4"
                              >
                                <div className="flex items-center mb-2">
                                  <Badge variant="outline" className="mr-2">
                                    {example.system}
                                  </Badge>
                                </div>
                                <div className="bg-muted p-3 rounded text-sm font-mono whitespace-pre-wrap">
                                  {example.example}
                                </div>
                              </div>
                            )
                          )}
                        </div>
                      </TabsContent>
                    )}
                </Tabs>
              </div>
            </Card>
          ) : (
            <Card className="h-[650px] flex items-center justify-center">
              <p className="text-muted-foreground">
                Select a strategy to view details
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
