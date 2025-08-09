"use client";

import React, { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Calendar, Users, Quote, TrendingUp } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { showcaseStyles, getBadgeStyle, getCardClassName } from "@/lib/showcase-styles";
import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HealthQuote {
  quote: string;
  speaker: string;
  role: string;
  topic: string;
  tone: string;
  interaction: string;
  response_type: string;
  quotability_score: number;
}

interface Hearing {
  title: string;
  date: string;
  committee: string;
  role_shares: { [key: string]: number };
  top_topics: string[];
  evasion_rate: number;
  most_quoted: HealthQuote[];
  quotes: HealthQuote[];
  summary?: string;
  html_link?: string;
}

export default function HealthQuotesExplorer() {
  const {
    data: hearings = [] as Hearing[],
    isLoading,
    error,
  } = useQuery({
    queryKey: ["health-quotes"],
    queryFn: async () => {
      const res = await fetch("/api/health-quotes");
      if (!res.ok) throw new Error(res.statusText);
      return res.json() as Promise<Hearing[]>;
    },
    staleTime: 1000 * 60 * 5,
  });

  const [selectedHearing, setSelectedHearing] = useState<Hearing | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [yearFilter, setYearFilter] = useState<string>("all");
  const [topicFilter, setTopicFilter] = useState<string>("all");
  const [committeeFilter, setCommitteeFilter] = useState<string>("all");

  // Extract unique years, topics, and committees
  const { years, topics, committees } = useMemo(() => {
    const yearSet = new Set<string>();
    const topicSet = new Set<string>();
    const committeeSet = new Set<string>();
    
    hearings.forEach((hearing) => {
      const year = hearing.date.split("-")[0];
      yearSet.add(year);
      committeeSet.add(hearing.committee);
      hearing.top_topics.forEach((topic) => topicSet.add(topic));
    });
    
    return {
      years: Array.from(yearSet).sort().reverse(),
      topics: Array.from(topicSet).sort(),
      committees: Array.from(committeeSet).sort(),
    };
  }, [hearings]);

  // Filter hearings
  const filteredHearings = useMemo(() => {
    return hearings.filter((hearing) => {
      const matchesSearch = searchQuery === "" || 
        hearing.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        hearing.top_topics.some(t => t.toLowerCase().includes(searchQuery.toLowerCase()));
      
      const matchesYear = yearFilter === "all" || hearing.date.startsWith(yearFilter);
      const matchesTopic = topicFilter === "all" || hearing.top_topics.includes(topicFilter);
      const matchesCommittee = committeeFilter === "all" || hearing.committee === committeeFilter;
      
      return matchesSearch && matchesYear && matchesTopic && matchesCommittee;
    });
  }, [hearings, searchQuery, yearFilter, topicFilter, committeeFilter]);

  // Calculate statistics
  const stats = useMemo(() => {
    const totalQuotes = hearings.reduce((sum, h) => sum + h.quotes.length, 0);
    const avgEvasionRate = hearings.reduce((sum, h) => sum + h.evasion_rate, 0) / hearings.length;
    const topSpeakers = new Map<string, number>();
    
    hearings.forEach((hearing) => {
      hearing.quotes.forEach((quote) => {
        topSpeakers.set(quote.speaker, (topSpeakers.get(quote.speaker) || 0) + 1);
      });
    });
    
    const sortedSpeakers = Array.from(topSpeakers.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    
    return { totalQuotes, avgEvasionRate, topSpeakers: sortedSpeakers };
  }, [hearings]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-6 text-gray-600">
        <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading congressional health hearings...
      </div>
    );
  }

  if (error) {
    return (
      <p className="text-red-600">Failed to load data: {String(error)}</p>
    );
  }

  return (
    <div className="space-y-6">
      {/* Filters */}
      <Card className={showcaseStyles.card.base}>
        <CardHeader>
          <CardTitle className={showcaseStyles.text.title}>Search & Filter</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="relative">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search hearings by title or topic..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 bg-white border-gray-200 text-gray-900 placeholder:text-gray-400"
            />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Select value={yearFilter} onValueChange={setYearFilter}>
              <SelectTrigger className="bg-white border-gray-200 text-gray-900">
                <SelectValue placeholder="Filter by year" />
              </SelectTrigger>
              <SelectContent className="bg-white">
                <SelectItem value="all">All Years</SelectItem>
                {years.map((year) => (
                  <SelectItem key={year} value={year}>{year}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Select value={topicFilter} onValueChange={setTopicFilter}>
              <SelectTrigger className="bg-white border-gray-200 text-gray-900">
                <SelectValue placeholder="Filter by topic" />
              </SelectTrigger>
              <SelectContent className="bg-white">
                <SelectItem value="all">All Topics</SelectItem>
                {topics.map((topic) => (
                  <SelectItem key={topic} value={topic}>{topic}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Select value={committeeFilter} onValueChange={setCommitteeFilter}>
              <SelectTrigger className="bg-white border-gray-200 text-gray-900">
                <SelectValue placeholder="Filter by committee" />
              </SelectTrigger>
              <SelectContent className="bg-white">
                <SelectItem value="all">All Committees</SelectItem>
                {committees.map((committee) => (
                  <SelectItem key={committee} value={committee}>{committee}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Hearings List */}
        <Card className={`${showcaseStyles.card.base} lg:col-span-1 overflow-hidden`}>
          <CardHeader>
            <CardTitle className={showcaseStyles.text.title}>Hearings ({filteredHearings.length})</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="max-h-[600px] overflow-y-auto overflow-x-hidden">
              {filteredHearings.map((hearing, idx) => (
                <div
                  key={idx}
                  className={`p-4 border-b border-gray-100 cursor-pointer transition-colors ${
                    selectedHearing?.title === hearing.title 
                      ? "bg-blue-50 border-l-4 border-l-blue-500" 
                      : "hover:bg-gray-50"
                  }`}
                  onClick={() => setSelectedHearing(hearing)}
                >
                  <div className="space-y-2">
                    <h3 className="font-medium text-sm text-gray-900 line-clamp-2 break-words pr-2">{hearing.title}</h3>
                    <div className="flex items-center gap-2 text-xs text-gray-600">
                      <Calendar className="h-3 w-3" />
                      {hearing.date}
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {hearing.top_topics.slice(0, 3).map((topic, i) => (
                        <Badge key={i} variant="secondary" className="text-xs bg-gray-100 text-gray-700">
                          {topic}
                        </Badge>
                      ))}
                    </div>
                    {hearing.evasion_rate > 0 && (
                      <div className="text-xs text-orange-600 font-medium">
                        Evasion rate: {(hearing.evasion_rate * 100).toFixed(0)}%
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Selected Hearing Details */}
        <Card className={`${showcaseStyles.card.base} lg:col-span-2`}>
          <CardHeader>
            <div className="flex justify-between items-start">
              <CardTitle className={showcaseStyles.text.title}>
                {selectedHearing ? "Hearing Analysis" : "Select a Hearing"}
              </CardTitle>
              {selectedHearing?.html_link && (
                <a
                  href={selectedHearing.html_link}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Button variant="outline" size="sm" className="bg-white">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    View Transcript
                  </Button>
                </a>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {selectedHearing ? (
              <Tabs defaultValue="quotes" className="space-y-4">
                <TabsList className="bg-gray-100">
                  <TabsTrigger value="quotes" className="data-[state=active]:bg-white">Top Quotes</TabsTrigger>
                  <TabsTrigger value="speakers" className="data-[state=active]:bg-white">Speaker Analysis</TabsTrigger>
                  <TabsTrigger value="topics" className="data-[state=active]:bg-white">Topics</TabsTrigger>
                </TabsList>
                
                <TabsContent value="quotes" className="space-y-4">
                  <div className="space-y-4">
                    {selectedHearing.most_quoted.map((quote, idx) => (
                      <Card key={idx} className="bg-gray-50 border-gray-200">
                        <CardContent className="pt-4">
                          <div className="flex items-start gap-2 mb-2">
                            <Quote className="h-4 w-4 mt-1 text-gray-400 flex-shrink-0" />
                            <p className="text-sm italic text-gray-800">"{quote.quote}"</p>
                          </div>
                          <div className="flex items-center justify-between mt-3">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-sm text-gray-900">{quote.speaker}</span>
                              <Badge className={`border ${getBadgeStyle('role', quote.role)}`}>
                                {quote.role}
                              </Badge>
                              <Badge className={`border ${getBadgeStyle('tone', quote.tone)}`}>
                                {quote.tone}
                              </Badge>
                            </div>
                            <div className="flex items-center gap-1">
                              <TrendingUp className="h-3 w-3 text-gray-500" />
                              <span className="text-xs text-gray-600">
                                Impact: {(quote.quotability_score * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                          <div className="mt-2">
                            <Badge variant="outline" className="text-xs border-gray-300 text-gray-700">
                              {quote.topic}
                            </Badge>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </TabsContent>
                
                <TabsContent value="speakers" className="space-y-4">
                  <div className="space-y-4">
                    <h3 className="font-medium text-gray-900">Speaker Distribution</h3>
                    {Object.entries(selectedHearing.role_shares).map(([role, share]) => (
                      <div key={role} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-700">{role}</span>
                          <span className="text-gray-900 font-medium">{(share * 100).toFixed(0)}%</span>
                        </div>
                        <Progress value={share * 100} className="h-2 bg-gray-100" />
                      </div>
                    ))}
                  </div>
                </TabsContent>
                
                <TabsContent value="topics" className="space-y-4">
                  <div className="space-y-2">
                    <h3 className="font-medium text-gray-900">Key Topics Discussed</h3>
                    <div className="flex flex-wrap gap-2">
                      {selectedHearing.top_topics.map((topic, idx) => (
                        <Badge key={idx} variant="secondary" className="bg-gray-100 text-gray-700">
                          {topic}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div className="mt-4 space-y-3">
                    <div>
                      <h3 className="font-medium mb-2 text-gray-900">Committee</h3>
                      <p className="text-sm text-gray-600">{selectedHearing.committee}</p>
                    </div>
                    
                    {selectedHearing.summary && (
                      <div>
                        <h3 className="font-medium mb-2 text-gray-900">Summary</h3>
                        <p className="text-sm text-gray-600">{selectedHearing.summary}</p>
                      </div>
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <Users className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Select a hearing to view detailed analysis</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Top Speakers */}
      <Card className={showcaseStyles.card.base}>
        <CardHeader>
          <CardTitle className={showcaseStyles.text.title}>Most Quoted Speakers</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {stats.topSpeakers.map(([speaker, count]) => (
              <div key={speaker} className="flex justify-between items-center">
                <span className="text-sm text-gray-700">{speaker}</span>
                <Badge variant="secondary" className="bg-gray-100 text-gray-700">
                  {count} quotes
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}