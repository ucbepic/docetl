"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Loader2 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { FixedSizeList as List } from "react-window";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import ReactMarkdown from "react-markdown";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Info } from "lucide-react";

// Define the type for a single response object
interface RfiResponse {
  concrete_proposal_described: boolean;
  from_famous_entity: boolean;
  entity_name: string;
  age_bracket: string; // "< 18", "18-25", "25-54", "55+", "N/A"
  main_topic: string;
  summary: string;
  filename?: string;
  text?: string;
}

// Define the type for age bracket counts
type AgeBracketCounts = {
  [key: string]: number;
};

export default function RfiResponseExplorer() {
  const {
    data: responses = [] as RfiResponse[],
    isLoading,
    error,
  } = useQuery({
    queryKey: ["rfi-responses"],
    queryFn: async () => {
      const res = await fetch("/api/rfi-responses");
      if (!res.ok) {
        throw new Error(`Failed to fetch data: ${res.statusText}`);
      }
      return res.json() as Promise<RfiResponse[]>;
    },
    staleTime: 1000 * 60 * 5, // Cache for 5 minutes
  });

  const [filterFamous, setFilterFamous] = useState<boolean | null>(null); // null means 'all'
  const [filterConcrete, setFilterConcrete] = useState<boolean | null>(null); // null means 'all'
  const [filterAgeBracket, setFilterAgeBracket] = useState<string | null>(null); // null means 'all'
  const [searchEntityName, setSearchEntityName] = useState<string>("");
  const [searchMainTopic, setSearchMainTopic] = useState<string>("");
  const [searchSummary, setSearchSummary] = useState<string>("");

  // Pagination state
  const [currentPage, setCurrentPage] = useState<number>(1);
  const itemsPerPage = 20;

  // Add a debounced filter implementation
  useEffect(() => {
    const timer = setTimeout(() => {
      // Apply filters here
    }, 300);

    return () => clearTimeout(timer);
  }, [filterFamous, filterConcrete]);

  const filteredResponses = useMemo(() => {
    return responses.filter((response) => {
      const famousMatch =
        filterFamous === null || response.from_famous_entity === filterFamous;
      const concreteMatch =
        filterConcrete === null ||
        response.concrete_proposal_described === filterConcrete;
      const ageBracketMatch =
        filterAgeBracket === null || response.age_bracket === filterAgeBracket;
      const entityNameMatch =
        searchEntityName === "" ||
        response.entity_name
          .toLowerCase()
          .includes(searchEntityName.toLowerCase());
      const mainTopicMatch =
        searchMainTopic === "" ||
        response.main_topic
          .toLowerCase()
          .includes(searchMainTopic.toLowerCase());
      const summaryMatch =
        searchSummary === "" ||
        response.summary.toLowerCase().includes(searchSummary.toLowerCase());
      return (
        famousMatch &&
        concreteMatch &&
        ageBracketMatch &&
        entityNameMatch &&
        mainTopicMatch &&
        summaryMatch
      );
    });
  }, [
    responses,
    filterFamous,
    filterConcrete,
    filterAgeBracket,
    searchEntityName,
    searchMainTopic,
    searchSummary,
  ]);

  const ageBracketCounts = useMemo(() => {
    return filteredResponses.reduce((acc, response) => {
      const bracket = response.age_bracket || "N/A";
      acc[bracket] = (acc[bracket] || 0) + 1;
      return acc;
    }, {} as AgeBracketCounts);
  }, [filteredResponses]);

  // Get current items
  const currentItems = useMemo(() => {
    const indexOfLastItem = currentPage * itemsPerPage;
    const indexOfFirstItem = indexOfLastItem - itemsPerPage;
    return filteredResponses.slice(indexOfFirstItem, indexOfLastItem);
  }, [filteredResponses, currentPage, itemsPerPage]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <span className="ml-2">Loading responses (may take 10 seconds)...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          {error instanceof Error ? error.message : error}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Filter Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 border rounded-md">
        <div className="space-y-2">
          <Label className="font-semibold">Filter by Notable Entity</Label>
          <div className="flex flex-col space-y-1">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="famous-all"
                checked={filterFamous === null}
                onCheckedChange={() => setFilterFamous(null)}
              />
              <Label htmlFor="famous-all">All</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="famous-true"
                checked={filterFamous === true}
                onCheckedChange={(checked) =>
                  setFilterFamous(checked ? true : null)
                }
              />
              <Label htmlFor="famous-true">Notable Entity</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="famous-false"
                checked={filterFamous === false}
                onCheckedChange={(checked) =>
                  setFilterFamous(checked ? false : null)
                }
              />
              <Label htmlFor="famous-false">Other/Unknown</Label>
            </div>
          </div>
        </div>
        <div className="space-y-2">
          <Label className="font-semibold">Filter by Proposal Type</Label>
          <div className="flex flex-col space-y-1">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="concrete-all"
                checked={filterConcrete === null}
                onCheckedChange={() => setFilterConcrete(null)}
              />
              <Label htmlFor="concrete-all">All</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="concrete-true"
                checked={filterConcrete === true}
                onCheckedChange={(checked) =>
                  setFilterConcrete(checked ? true : null)
                }
              />
              <Label htmlFor="concrete-true">Concrete Proposal</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="concrete-false"
                checked={filterConcrete === false}
                onCheckedChange={(checked) =>
                  setFilterConcrete(checked ? false : null)
                }
              />
              <Label htmlFor="concrete-false">General/Vague</Label>
            </div>
          </div>
        </div>
        {/* Age Bracket Visualization - More compact for mobile */}
        <Card className="sm:col-span-1">
          <CardHeader className="pb-1 pt-2 px-3">
            <CardTitle className="text-base font-semibold">
              Age Brackets (Filtered)
            </CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            {Object.keys(ageBracketCounts).length > 0 ? (
              <ul className="text-sm space-y-1">
                {Object.entries(ageBracketCounts)
                  .sort(([keyA], [keyB]) => keyA.localeCompare(keyB))
                  .map(([bracket, count]) => (
                    <li
                      key={bracket}
                      className={`flex justify-between p-0.5 rounded cursor-pointer hover:bg-gray-100 ${
                        filterAgeBracket === bracket
                          ? "bg-primary/10 font-medium"
                          : ""
                      }`}
                      onClick={() =>
                        setFilterAgeBracket(
                          filterAgeBracket === bracket ? null : bracket
                        )
                      }
                    >
                      <span className="text-xs md:text-sm">{bracket}:</span>
                      <Badge
                        variant="secondary"
                        className="text-xs px-1.5 py-0 h-5"
                      >
                        {count}
                      </Badge>
                    </li>
                  ))}
              </ul>
            ) : (
              <p className="text-xs md:text-sm text-muted-foreground">
                No data for selected filters.
              </p>
            )}

            {filterAgeBracket && (
              <button
                onClick={() => setFilterAgeBracket(null)}
                className="mt-1 text-xs text-blue-600 hover:underline"
              >
                Clear age filter
              </button>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Results Display - Table for desktop, Cards for mobile */}
      <div className="border rounded-md">
        <p className="p-4 text-sm text-muted-foreground">
          Showing {filteredResponses.length} of {responses.length} responses
        </p>

        {/* Mobile Search Bar */}
        <div className="p-4 sm:hidden">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search entities, topics or summaries..."
              className="pl-8 w-full"
              value={searchEntityName}
              onChange={(e) => {
                setSearchEntityName(e.target.value);
                setSearchMainTopic(e.target.value);
                setSearchSummary(e.target.value);
              }}
            />
          </div>
        </div>

        {/* Desktop View: Table */}
        <div className="hidden sm:block overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="min-w-[120px]">
                  <div className="space-y-2">
                    <div className="font-medium">Entity Name</div>
                    <input
                      type="text"
                      placeholder="Search..."
                      value={searchEntityName}
                      onChange={(e) => setSearchEntityName(e.target.value)}
                      className="w-full p-1 text-sm border rounded"
                    />
                  </div>
                </TableHead>
                <TableHead className="min-w-[120px] hidden sm:table-cell">
                  <div className="space-y-2">
                    <div className="font-medium">Main Topic</div>
                    <input
                      type="text"
                      placeholder="Search..."
                      value={searchMainTopic}
                      onChange={(e) => setSearchMainTopic(e.target.value)}
                      className="w-full p-1 text-sm border rounded"
                    />
                  </div>
                </TableHead>
                <TableHead className="min-w-[200px] hidden md:table-cell">
                  <div className="space-y-2">
                    <div className="font-medium">Summary</div>
                    <input
                      type="text"
                      placeholder="Search..."
                      value={searchSummary}
                      onChange={(e) => setSearchSummary(e.target.value)}
                      className="w-full p-1 text-sm border rounded"
                    />
                  </div>
                </TableHead>
                <TableHead className="text-center w-[80px]">Notable</TableHead>
                <TableHead className="text-center w-[80px]">Concrete</TableHead>
                <TableHead className="w-[60px]">Age</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody style={{ height: "400px" }}>
              {currentItems.map((response, index) => (
                <HoverCard key={index} openDelay={0} closeDelay={0}>
                  <HoverCardTrigger asChild>
                    <TableRow className="cursor-pointer hover:bg-muted/50">
                      <TableCell className="font-medium max-w-[150px] whitespace-normal break-words">
                        {response.entity_name}
                      </TableCell>
                      <TableCell className="max-w-[200px] whitespace-normal break-words">
                        {response.main_topic}
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground max-w-[300px] whitespace-normal">
                        {response.summary}
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge
                          variant={
                            response.from_famous_entity ? "default" : "outline"
                          }
                        >
                          {response.from_famous_entity ? "Yes" : "No"}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge
                          variant={
                            response.concrete_proposal_described
                              ? "default"
                              : "outline"
                          }
                        >
                          {response.concrete_proposal_described ? "Yes" : "No"}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary">
                          {response.age_bracket}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  </HoverCardTrigger>
                  <HoverCardContent className="w-80">
                    <div className="space-y-2">
                      {response.filename && (
                        <div>
                          <h4 className="text-sm font-semibold">
                            Source File:
                          </h4>
                          <p className="text-sm">{response.filename}</p>
                        </div>
                      )}
                      {response.text && (
                        <div>
                          <h4 className="text-sm font-semibold">
                            Original Text:
                          </h4>
                          <div className="text-sm max-h-80 overflow-y-auto markdown-content">
                            <ReactMarkdown>{response.text}</ReactMarkdown>
                          </div>
                        </div>
                      )}
                      {!response.filename && !response.text && (
                        <p className="text-sm text-muted-foreground italic">
                          Additional details not available
                        </p>
                      )}
                    </div>
                  </HoverCardContent>
                </HoverCard>
              ))}
            </TableBody>
          </Table>
        </div>

        {/* Mobile View: Simplified list-based */}
        <div className="sm:hidden space-y-2 px-2 pb-4">
          {currentItems.map((response, index) => (
            <div
              key={index}
              className="p-3 border border-border rounded-md bg-card/50"
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-sm">
                    {response.entity_name}
                  </h3>
                  <Badge variant="secondary" className="text-xs">
                    {response.age_bracket}
                  </Badge>
                </div>

                <div className="text-xs">
                  <span className="font-medium">Topic:</span>{" "}
                  {response.main_topic}
                </div>

                <div className="text-xs text-muted-foreground">
                  {response.summary}
                </div>

                <div className="flex flex-wrap gap-2 pt-1">
                  <Badge
                    variant={
                      response.from_famous_entity ? "default" : "outline"
                    }
                    className="text-xs"
                  >
                    {response.from_famous_entity ? "Notable" : "Other"}
                  </Badge>
                  <Badge
                    variant={
                      response.concrete_proposal_described
                        ? "default"
                        : "outline"
                    }
                    className="text-xs"
                  >
                    {response.concrete_proposal_described
                      ? "Concrete"
                      : "General"}
                  </Badge>
                </div>

                <Dialog>
                  <DialogTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-xs flex items-center mt-1 px-2 h-7"
                    >
                      <Info className="h-3.5 w-3.5 mr-1" />
                      View Details
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                      <DialogTitle>{response.entity_name}</DialogTitle>
                    </DialogHeader>
                    <div className="space-y-3 max-h-[70vh] overflow-y-auto">
                      {response.filename && (
                        <div>
                          <h4 className="text-sm font-semibold">
                            Source File:
                          </h4>
                          <p className="text-sm">{response.filename}</p>
                        </div>
                      )}
                      {response.text && (
                        <div>
                          <h4 className="text-sm font-semibold">
                            Original Text:
                          </h4>
                          <div className="text-sm markdown-content">
                            <ReactMarkdown>{response.text}</ReactMarkdown>
                          </div>
                        </div>
                      )}
                      {!response.filename && !response.text && (
                        <p className="text-sm text-muted-foreground italic">
                          Additional details not available
                        </p>
                      )}
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Pagination Controls */}
      <div className="flex flex-wrap justify-center gap-2 mt-4">
        <button
          onClick={() => setCurrentPage(currentPage - 1)}
          disabled={currentPage === 1}
          className="px-3 py-2 bg-gray-200 text-gray-500 rounded-md text-sm"
        >
          Previous
        </button>
        <div className="px-3 py-2 bg-gray-200 text-gray-500 flex items-center text-sm">
          <span className="hidden sm:inline">Page</span>{" "}
          <input
            type="number"
            min={1}
            max={Math.ceil(filteredResponses.length / itemsPerPage)}
            value={currentPage}
            onChange={(e) => {
              const value = parseInt(e.target.value);
              if (
                !isNaN(value) &&
                value >= 1 &&
                value <= Math.ceil(filteredResponses.length / itemsPerPage)
              ) {
                setCurrentPage(value);
              }
            }}
            onBlur={(e) => {
              const value = parseInt(e.target.value);
              if (
                isNaN(value) ||
                value < 1 ||
                value > Math.ceil(filteredResponses.length / itemsPerPage)
              ) {
                // Reset to current page if invalid
                e.target.value = currentPage.toString();
              }
            }}
            className="w-10 mx-1 text-center bg-white rounded border border-gray-300"
          />{" "}
          <span className="hidden sm:inline">of</span>{" "}
          <span className="sm:hidden">/</span>{" "}
          {Math.ceil(filteredResponses.length / itemsPerPage)}
        </div>
        <button
          onClick={() => setCurrentPage(currentPage + 1)}
          disabled={
            currentPage === Math.ceil(filteredResponses.length / itemsPerPage)
          }
          className="px-3 py-2 bg-gray-200 text-gray-500 rounded-md text-sm"
        >
          Next
        </button>
      </div>
    </div>
  );
}
