import { File } from "@/app/types";
import React, { useRef, useMemo, useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { useInfiniteQuery } from "@tanstack/react-query";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ChevronUp, ChevronDown, Search } from "lucide-react";
import BookmarkableText from "@/components/BookmarkableText";
import { Skeleton } from "@/components/ui/skeleton";
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
    // @ts-ignore
    useInfiniteQuery({
      // @ts-ignore
      queryKey: ["fileContent", file?.path],
      // @ts-ignore
      queryFn: ({ pageParam = 0 }) => fetchFileContent({ pageParam }),
      // @ts-ignore
      getNextPageParam: (lastPage) =>
        // @ts-ignore
        lastPage.hasMore ? lastPage.page + 1 : undefined,
      // @ts-ignore
      enabled: !!file?.path,
    });

  const lines = useMemo(() => {
    // @ts-ignore
    return data?.pages.flatMap((page) => page.content.split("\n")) ?? [];
  }, [data]);

  // Extract keys from the first valid JSON object in the data
  useMemo(() => {
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
            return; // Exit after finding the first valid object
          } catch (error) {
            console.error("Error parsing JSON:", error);
          }
          // Reset for next attempt
          jsonString = "";
          inObject = false;
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
          // @ts-ignore
          const allContent = data.pages.map((page) => page.content).join("");
          const documents = JSON.parse(allContent) as Record<string, unknown>[];

          const wordCounts = documents.map((doc) => {
            const text = Object.values(doc).join(" ");
            return text
              .trim()
              .split(/\s+/)
              .filter((word) => word.length > 0).length;
          });

          // Calculate histogram bins
          const binCount = Math.min(
            20,
            Math.ceil(Math.sqrt(wordCounts.length))
          );
          const minCount = Math.min(...wordCounts);
          const maxCount = Math.max(...wordCounts);
          const binSize = Math.ceil((maxCount - minCount) / binCount);

          const histogram: HistogramBin[] = Array.from(
            { length: binCount },
            (_, i) => ({
              start: minCount + i * binSize,
              end: minCount + (i + 1) * binSize,
              count: 0,
            })
          );

          wordCounts.forEach((count) => {
            const binIndex = Math.min(
              Math.floor((count - minCount) / binSize),
              binCount - 1
            );
            histogram[binIndex].count++;
          });

          const stats: DatasetStats = {
            documentCount: documents.length,
            averageWords:
              wordCounts.length > 0
                ? Math.round(
                    wordCounts.reduce((sum, count) => sum + count, 0) /
                      documents.length
                  )
                : 0,
            minWords: wordCounts.length > 0 ? Math.min(...wordCounts) : 0,
            maxWords: wordCounts.length > 0 ? Math.max(...wordCounts) : 0,
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

  if (isError) return <div>Error: {error.message}</div>;

  return (
    <div className="h-full p-4 bg-white flex flex-col">
      <h2 className="text-lg font-bold mb-2">{file?.name}</h2>
      <div className="mb-4 p-2 rounded-lg border border-border bg-card/50">
        <p className="mb-2 font-semibold text-sm text-muted-foreground uppercase tracking-wide">
          Available keys
        </p>
        <div className="flex flex-wrap gap-1">
          {keys.map((key) => (
            <Badge
              key={key}
              variant="secondary"
              className="px-2 py-0.5 text-sm font-medium"
            >
              {key}
            </Badge>
          ))}
        </div>
      </div>
      <Collapsible className="mb-4">
        <CollapsibleTrigger className="flex items-center gap-2 hover:text-blue-500 transition-colors">
          <ChevronRight className="h-4 w-4 transition-transform ui-expanded:rotate-90" />
          <p className="font-semibold">Dataset Statistics</p>
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
              <div className="flex gap-12">
                <div className="flex flex-col justify-center space-y-3 w-[20%]">
                  <div>
                    <p className="text-sm text-muted-foreground">Documents</p>
                    <p className="text-2xl font-bold">
                      {datasetStats.documentCount.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">
                      Average Words
                    </p>
                    <p className="text-2xl font-bold">
                      {datasetStats.averageWords.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Min Words</p>
                    <p className="text-2xl font-bold">
                      {datasetStats.minWords.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Max Words</p>
                    <p className="text-2xl font-bold">
                      {datasetStats.maxWords.toLocaleString()}
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
        <span className="ml-2">
          {matches.length > 0
            ? `${currentMatchIndex + 1} of ${matches.length} matches`
            : "No matches"}
        </span>
      </form>
      <BookmarkableText source="dataset">
        <div ref={parentRef} className="flex-grow overflow-y-auto">
          {lines.map((lineContent, index) => (
            <div key={index} className="flex">
              <span className="inline-block w-12 flex-shrink-0 text-gray-500 select-none text-right pr-2">
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
            <div className="text-center py-4">Loading more...</div>
          )}
        </div>
      </BookmarkableText>
    </div>
  );
};

export default DatasetView;
