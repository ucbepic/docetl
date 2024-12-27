import React, { useRef, useMemo, useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { useInfiniteQuery } from "@tanstack/react-query";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ChevronUp, ChevronDown, Search } from "lucide-react";
import { Loader2 } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronRight } from "lucide-react";
import { Database } from "lucide-react";
import { File } from "@/app/types";
import { cn } from "@/lib/utils";

interface FileChunk {
  content: string;
  totalSize: number;
  page: number;
  hasMore: boolean;
}

interface Match {
  lineIndex: number;
  startIndex: number;
  endIndex: number;
}

interface HistogramBin {
  start: number;
  end: number;
  count: number;
}

interface DatasetStats {
  documentCount: number;
  averageWords: number;
  minWords: number;
  maxWords: number;
  standardDeviation: number;
  isCalculating: boolean;
  wordCounts: number[];
  histogram: HistogramBin[];
}

const DatasetView: React.FC<{ file: File | null }> = ({ file }) => {
  const parentRef = useRef<HTMLDivElement>(null);
  const [keys, setKeys] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);
  const [matches, setMatches] = useState<Match[]>([]);
  const [datasetStats, setDatasetStats] = useState<DatasetStats>({
    documentCount: 0,
    averageWords: 0,
    minWords: 0,
    maxWords: 0,
    standardDeviation: 0,
    isCalculating: false,
    wordCounts: [],
    histogram: [],
  });

  const fetchFileContent = async ({ pageParam = 0 }): Promise<FileChunk> => {
    if (!file?.path) throw new Error("No file selected");
    const response = await fetch(
      `/api/readFilePage?path=${encodeURIComponent(
        file.path
      )}&page=${pageParam}`
    );
    if (!response.ok) throw new Error("Failed to fetch file content");
    return response.json();
  };

  const { data, fetchNextPage, hasNextPage, isFetching, isError, error } =
    // @ts-expect-error Property 'initialPageParam' is missing in type
    useInfiniteQuery<FileChunk>({
      queryKey: ["fileContent", file?.path],
      // @ts-expect-error Parameter 'pageParam' implicitly has an 'any' type
      queryFn: ({ pageParam = 0 }) => fetchFileContent({ pageParam }),
      getNextPageParam: (lastPage) =>
        lastPage.hasMore ? lastPage.page + 1 : undefined,
      enabled: !!file?.path,
    });

  const lines = useMemo(() => {
    // @ts-expect-error Property 'content' does not exist on type 'unknown'
    return data?.pages.flatMap((page) => page.content.split("\n")) ?? [];
  }, [data]);

  // Extract keys from the first valid JSON object in the data
  useMemo(() => {
    if (!lines.length) return;

    try {
      // Try to parse the entire content as JSON first
      const content = lines.join("\n");
      const parsed = JSON.parse(content);

      if (Array.isArray(parsed)) {
        // If it's an array, get keys from the first object
        if (parsed.length > 0 && typeof parsed[0] === "object") {
          setKeys(Object.keys(parsed[0]));
        }
      } else if (typeof parsed === "object" && parsed !== null) {
        // If it's a single object, get its keys
        setKeys(Object.keys(parsed));
      }
    } catch (error) {
      // Fallback to the original line-by-line approach
      let jsonString = "";
      let braceCount = 0;
      let inObject = false;

      for (const line of lines) {
        for (let i = 0; i < line.length; i++) {
          const char = line[i];
          if (char === "{") {
            if (!inObject) inObject = true;
            braceCount++;
          } else if (char === "}") {
            braceCount--;
          }

          if (inObject) {
            jsonString += char;
          }

          if (inObject && braceCount === 0) {
            try {
              const parsedObject = JSON.parse(jsonString);
              setKeys(Object.keys(parsedObject));
              return;
            } catch (error) {
              console.error("Error parsing JSON:", error);
            }
            jsonString = "";
            inObject = false;
          }
        }
      }
    }
  }, [lines]);

  // Perform search and update matches
  useEffect(() => {
    if (searchTerm.length >= 5) {
      const newMatches: Match[] = [];
      const regex = new RegExp(searchTerm, "gi");
      lines.forEach((line, lineIndex) => {
        let match;
        while ((match = regex.exec(line)) !== null) {
          newMatches.push({
            lineIndex,
            startIndex: match.index,
            endIndex: match.index + match[0].length,
          });
        }
      });
      setMatches(newMatches);
      setCurrentMatchIndex(0);
    } else {
      setMatches([]);
      setCurrentMatchIndex(0);
    }
  }, [searchTerm, lines]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // The search is already performed in the useEffect above
  };

  const navigateMatch = (direction: "next" | "prev") => {
    if (matches.length === 0) return;
    let newIndex =
      direction === "next" ? currentMatchIndex + 1 : currentMatchIndex - 1;
    if (newIndex < 0) newIndex = matches.length - 1;
    if (newIndex >= matches.length) newIndex = 0;
    setCurrentMatchIndex(newIndex);

    // Scroll to the new match
    const matchElement = document.getElementById(`match-${newIndex}`);
    if (matchElement) {
      matchElement.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  const highlightMatches = (text: string, lineIndex: number) => {
    if (!searchTerm || searchTerm.length < 5) return text;
    const parts = [];
    let lastIndex = 0;
    matches
      .filter((match) => match.lineIndex === lineIndex)
      .forEach((match) => {
        if (lastIndex < match.startIndex) {
          parts.push(text.slice(lastIndex, match.startIndex));
        }
        parts.push(
          <mark
            key={match.startIndex}
            id={`match-${matches.findIndex(
              (m) =>
                m.lineIndex === lineIndex && m.startIndex === match.startIndex
            )}`}
            className={`bg-yellow-200 ${
              currentMatchIndex ===
              matches.findIndex(
                (m) =>
                  m.lineIndex === lineIndex && m.startIndex === match.startIndex
              )
                ? "ring-2 ring-blue-500"
                : ""
            }`}
          >
            {text.slice(match.startIndex, match.endIndex)}
          </mark>
        );
        lastIndex = match.endIndex;
      });
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }
    return parts;
  };

  // Keep fetching in the background
  useEffect(() => {
    const fetchNextPageIfNeeded = () => {
      if (hasNextPage && !isFetching) {
        fetchNextPage();
      }
    };

    const intervalId = setInterval(fetchNextPageIfNeeded, 1000); // Check every second

    return () => clearInterval(intervalId);
  }, [fetchNextPage, hasNextPage, isFetching]);

  useEffect(() => {
    // Scroll to the current match when it changes
    const currentMatchElement = document.getElementById(
      `match-${currentMatchIndex}`
    );
    if (currentMatchElement) {
      currentMatchElement.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [currentMatchIndex]);

  useEffect(() => {
    if (
      !hasNextPage &&
      !isFetching &&
      data?.pages &&
      !datasetStats.isCalculating
    ) {
      setDatasetStats((prev) => ({ ...prev, isCalculating: true }));

      setTimeout(() => {
        try {
          // @ts-expect-error Property 'content' does not exist on type 'unknown'
          const allContent = data.pages.map((page) => page.content).join("");
          let documents: Record<string, unknown>[] = [];

          try {
            documents = JSON.parse(allContent) as Record<string, unknown>[];
            if (!Array.isArray(documents)) {
              throw new Error("Content is not an array of documents");
            }
          } catch (parseError) {
            console.error("Error parsing JSON:", parseError);
            setDatasetStats((prev) => ({ ...prev, isCalculating: false }));
            return;
          }

          const wordCounts = documents.map((doc) => {
            const text =
              typeof doc === "object"
                ? JSON.stringify(doc, null, 2).replace(/[{}[\]"]/g, "")
                : String(doc);
            return text
              .trim()
              .split(/\s+/)
              .filter((word) => word.length > 0).length;
          });

          if (wordCounts.length === 0) {
            setDatasetStats((prev) => ({ ...prev, isCalculating: false }));
            return;
          }

          // Modified histogram calculation
          const binCount = Math.min(
            20,
            Math.ceil(Math.sqrt(wordCounts.length))
          );
          const minCount = Math.min(...wordCounts);
          const maxCount = Math.max(...wordCounts);

          // Handle single document case
          const effectiveRange =
            maxCount === minCount ? maxCount : maxCount - minCount;
          const binSize = Math.max(
            1,
            Math.ceil(effectiveRange / (binCount || 1))
          );

          const histogram: HistogramBin[] =
            maxCount === minCount
              ? [{ start: minCount, end: maxCount, count: wordCounts.length }]
              : Array.from({ length: binCount }, (_, i) => ({
                  start: minCount + i * binSize,
                  end: minCount + (i + 1) * binSize,
                  count: 0,
                }));

          // Count documents in each bin
          if (maxCount !== minCount) {
            wordCounts.forEach((count) => {
              if (typeof count === "number") {
                const binIndex = Math.min(
                  Math.floor((count - minCount) / binSize),
                  binCount - 1
                );
                if (binIndex >= 0 && binIndex < histogram.length) {
                  histogram[binIndex].count++;
                }
              }
            });
          }

          const average =
            wordCounts.reduce((sum, count) => sum + count, 0) /
            documents.length;

          // Calculate standard deviation
          const squareDiffs = wordCounts.map((count) =>
            Math.pow(count - average, 2)
          );
          const standardDeviation = Math.round(
            Math.sqrt(
              squareDiffs.reduce((sum, diff) => sum + diff, 0) /
                documents.length
            )
          );

          const stats: DatasetStats = {
            documentCount: documents.length,
            averageWords: Math.round(average),
            minWords: wordCounts.length > 0 ? Math.min(...wordCounts) : 0,
            maxWords: wordCounts.length > 0 ? Math.max(...wordCounts) : 0,
            standardDeviation,
            isCalculating: false,
            wordCounts,
            histogram,
          };

          setDatasetStats(stats);
        } catch (error) {
          console.error("Error calculating stats:", error);
          setDatasetStats((prev) => ({ ...prev, isCalculating: false }));
        }
      }, 0);
    }
  }, [hasNextPage, isFetching, data]);

  if (!file) {
    return (
      <div className="h-full flex flex-col p-4">
        <div className="flex-1 flex flex-col items-center justify-center gap-2 text-center">
          <Database className="h-12 w-12 text-muted-foreground/50" />
          <h3 className="font-medium text-muted-foreground">
            No Dataset Selected
          </h3>
          <p className="text-sm text-muted-foreground/80">
            Please select or upload a file from the left panel to view its
            contents.
          </p>
        </div>
      </div>
    );
  }

  if (isError) return <div>Error: {error.message}</div>;

  return (
    <div className="h-full flex flex-col p-4">
      <div className="flex justify-between items-center mb-4 border-b pb-3">
        <h2 className="text-base font-bold flex items-center">
          <Database className="mr-2" size={18} />
          {file?.name}
        </h2>
      </div>

      <div className="text-xs mb-4 bg-muted/50 p-2 rounded-md">
        <span className="text-muted-foreground font-medium">
          Available Keys:{" "}
        </span>
        <div className="flex flex-wrap gap-1 mt-2">
          {keys.map((key) => (
            <Badge
              key={key}
              variant="default"
              className="transition-none hover:bg-primary hover:text-primary-foreground"
            >
              {key}
            </Badge>
          ))}
        </div>
      </div>

      <Collapsible className="mb-4">
        <CollapsibleTrigger className="flex items-center gap-2 hover:text-primary transition-colors">
          <ChevronRight className="h-4 w-4 transition-transform ui-expanded:rotate-90" />
          <p className="text-sm font-medium">Dataset Statistics</p>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-4">
          {hasNextPage || datasetStats.isCalculating ? (
            <div className="flex items-center gap-2 p-4 rounded-lg border border-border bg-card/50">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <p className="text-sm text-muted-foreground">
                Calculating dataset statistics...
              </p>
            </div>
          ) : (
            <div className="p-4 rounded-lg border border-border bg-card/50">
              <div className="flex gap-8">
                <div className="flex flex-col justify-center space-y-2 w-[20%]">
                  <div>
                    <p className="text-xs text-muted-foreground">Documents</p>
                    <p className="text-xl font-semibold">
                      {datasetStats.documentCount.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">
                      Average Words
                    </p>
                    <p className="text-xl font-semibold">
                      {datasetStats.averageWords.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Min Words</p>
                    <p className="text-xl font-semibold">
                      {datasetStats.minWords.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Max Words</p>
                    <p className="text-xl font-semibold">
                      {datasetStats.maxWords.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">
                      Std Deviation
                    </p>
                    <p className="text-xl font-semibold">
                      {datasetStats.standardDeviation.toLocaleString()}
                    </p>
                  </div>
                </div>
                {datasetStats.histogram.length > 0 && (
                  <div className="w-[80%] h-[300px]">
                    <p className="text-sm font-medium mb-3 text-muted-foreground">
                      Word Count Distribution
                    </p>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={datasetStats.histogram}>
                        <XAxis
                          dataKey="start"
                          tickFormatter={(value) => value.toLocaleString()}
                          angle={-45}
                          textAnchor="end"
                          height={60}
                          className="text-foreground text-xs"
                          stroke="currentColor"
                        />
                        <YAxis
                          className="text-foreground text-xs"
                          stroke="currentColor"
                          tickFormatter={(value) => value.toLocaleString()}
                        />
                        <Tooltip
                          formatter={(value: number) => [
                            value.toLocaleString(),
                            "Documents",
                          ]}
                          labelFormatter={(label: number) =>
                            `${label.toLocaleString()} - ${(
                              label +
                              Math.ceil(
                                (datasetStats.maxWords -
                                  datasetStats.minWords) /
                                  20
                              )
                            ).toLocaleString()} words`
                          }
                          contentStyle={{
                            backgroundColor: "hsl(var(--popover))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: "var(--radius)",
                            color: "hsl(var(--popover-foreground))",
                            padding: "8px 12px",
                            boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
                          }}
                        />
                        <Bar
                          dataKey="count"
                          fill="hsl(var(--chart-2))"
                          name="Documents"
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>

      <form onSubmit={handleSearch} className="flex items-center mb-4">
        <Input
          type="text"
          placeholder="Search (min 5 characters)..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="mr-1"
        />
        <Button
          type="submit"
          variant="outline"
          size="icon"
          disabled={searchTerm.length < 5}
        >
          <Search className="h-4 w-4" />
        </Button>
        <Button
          type="button"
          size="icon"
          variant="outline"
          onClick={() => navigateMatch("prev")}
          disabled={matches.length === 0}
          className="ml-1"
        >
          <ChevronUp className="h-4 w-4" />
        </Button>
        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={() => navigateMatch("next")}
          disabled={matches.length === 0}
          className="ml-1"
        >
          <ChevronDown className="h-4 w-4" />
        </Button>
        <span className="ml-2 text-sm text-muted-foreground">
          {matches.length > 0
            ? `${currentMatchIndex + 1} of ${matches.length} matches`
            : "No matches"}
        </span>
      </form>

      <div ref={parentRef} className="flex-grow overflow-y-auto">
        {lines.map((lineContent, index) => (
          <div key={index} className="flex hover:bg-gray-50">
            <span className="inline-block w-12 flex-shrink-0 text-muted-foreground select-none text-right pr-2 text-sm">
              {index + 1}
            </span>
            <div className="flex-grow">
              <pre className="whitespace-pre-wrap break-words font-mono text-sm">
                {highlightMatches(lineContent, index)}
              </pre>
            </div>
          </div>
        ))}
        {isFetching && (
          <div className="text-center py-4 text-sm text-muted-foreground">
            Loading more...
          </div>
        )}
      </div>
    </div>
  );
};

export default DatasetView;
