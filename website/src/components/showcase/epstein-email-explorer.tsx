"use client";

import React, { useState, useMemo, useCallback, useRef, useEffect } from "react";
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
import { Loader2, AlertTriangle, Users, Calendar as CalendarIcon, Search, X, Filter, ChevronDown, ChevronUp, Plus, Mail, UserCircle, Download, Check, FileJson } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Participant {
  name: string;
  email: string;
}

interface InferredPerson {
  name: string;
  reference: string;
  reasoning: string;
}

interface Email {
  document_id: string;
  email_text: string;
  source_file: string;
  is_email: boolean;
  participants: Participant[];
  date: string;
  time: string;
  subject: string;
  has_attachments: boolean;
  attachment_names: string[];
  people_mentioned: string[];
  organizations: string[];
  locations: string[];
  phone_numbers: string[];
  urls: string[];
  notable_figures: string[];
  primary_topic: string;
  topics: string[];
  inferred_people: InferredPerson[];
  summary: string;
  key_quotes: string[];
  tone: string;
  potential_crimes: string;
  evidence_strength: string;
  crime_types: string[];
  mentions_victims: boolean;
  victim_names: string[];
  cover_up: string;
}

interface AttributeFilter {
  id: string;
  attribute: string;
  value: string;
}

interface PersonDetailModalProps {
  person: string;
  emails: Email[];
  open: boolean;
  onClose: () => void;
  onFilterByPerson: (person: string, attribute: string) => void;
}

function PersonDetailModal({ person, emails, open, onClose, onFilterByPerson }: PersonDetailModalProps) {
  const participatedIn = emails.filter((e) =>
    e.participants.some((p) => p.name.toLowerCase().includes(person.toLowerCase()))
  );
  const mentionedIn = emails.filter((e) =>
    e.people_mentioned.some((p) => p.toLowerCase().includes(person.toLowerCase())) ||
    e.notable_figures.some((p) => p.toLowerCase().includes(person.toLowerCase()))
  );

  const handleFilterAsParticipant = () => {
    onFilterByPerson(person, "participant");
    onClose();
  };

  const handleFilterAsMentioned = () => {
    onFilterByPerson(person, "people_mentioned");
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>{person}</DialogTitle>
          <DialogDescription>
            Found in {participatedIn.length} email(s) as participant, {mentionedIn.length} email(s) as mentioned
          </DialogDescription>
        </DialogHeader>
        <div className="mb-4 flex gap-2">
          {participatedIn.length > 0 && (
            <Button onClick={handleFilterAsParticipant} variant="outline" size="sm">
              <Filter className="h-4 w-4 mr-2" />
              Filter as Participant
            </Button>
          )}
          {mentionedIn.length > 0 && (
            <Button onClick={handleFilterAsMentioned} variant="outline" size="sm">
              <Filter className="h-4 w-4 mr-2" />
              Filter as Mentioned
            </Button>
          )}
        </div>
        <ScrollArea className="h-[60vh]">
          <div className="space-y-4">
            {participatedIn.length > 0 && (
              <div>
                <h3 className="font-semibold mb-2">Participated In ({participatedIn.length})</h3>
                <div className="space-y-2">
                  {participatedIn.slice(0, 10).map((email) => (
                    <div key={email.document_id} className="text-sm border rounded p-2">
                      <div className="font-medium">{email.subject || "(No subject)"}</div>
                      <div className="text-gray-500">{email.date}</div>
                      <div className="text-xs text-gray-600 mt-1">{email.summary.slice(0, 150)}...</div>
                    </div>
                  ))}
                  {participatedIn.length > 10 && (
                    <div className="text-sm text-gray-500">...and {participatedIn.length - 10} more</div>
                  )}
                </div>
              </div>
            )}

            {mentionedIn.length > 0 && (
              <div>
                <h3 className="font-semibold mb-2">Mentioned In ({mentionedIn.length})</h3>
                <div className="space-y-2">
                  {mentionedIn.slice(0, 10).map((email) => (
                    <div key={email.document_id} className="text-sm border rounded p-2">
                      <div className="font-medium">{email.subject || "(No subject)"}</div>
                      <div className="text-gray-500">{email.date}</div>
                      <div className="text-xs text-gray-600 mt-1">{email.summary.slice(0, 150)}...</div>
                    </div>
                  ))}
                  {mentionedIn.length > 10 && (
                    <div className="text-sm text-gray-500">...and {mentionedIn.length - 10} more</div>
                  )}
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}

interface EmailDetailViewProps {
  email: Email;
  onPersonClick: (person: string) => void;
}

const EmailDetailView = React.memo(({ email, onPersonClick }: EmailDetailViewProps) => {
  const [showFullText, setShowFullText] = useState(false);

  const renderClickablePeople = (text: string, people: string[]) => {
    if (!people || people.length === 0) return text;

    let result = text;
    const allPeople = [
      ...email.participants.map(p => p.name),
      ...email.people_mentioned,
      ...email.notable_figures
    ].filter(Boolean);

    // Sort by length descending to match longer names first
    const sortedPeople = Array.from(new Set(allPeople)).sort((a, b) => b.length - a.length);

    sortedPeople.forEach(person => {
      const regex = new RegExp(`\\b${person.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi');
      result = result.replace(regex, `<span class="cursor-pointer text-blue-600 hover:underline" data-person="${person}">$&</span>`);
    });

    return result;
  };

  const handleTextClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    const person = target.getAttribute('data-person');
    if (person) {
      onPersonClick(person);
    }
  };

  const emailText = email.email_text || "";
  const truncatedText = showFullText ? emailText : emailText.slice(0, 2000);
  const allPeople = [
    ...email.participants.map(p => p.name),
    ...email.people_mentioned,
    ...email.notable_figures
  ].filter(Boolean);
  const highlightedText = renderClickablePeople(truncatedText, allPeople);

  return (
    <div className="flex flex-col lg:flex-row gap-4 flex-1">
      {/* Left: Email Text */}
      <div className="flex-1 lg:w-3/5 border border-gray-200 rounded-md p-4 overflow-y-auto max-h-[70vh] bg-white">
        <div className="mb-4">
          {email.source_file && (
            <div className="text-xs text-gray-500 font-mono mb-2 bg-gray-50 p-2 rounded border border-gray-200">
              {email.source_file}
            </div>
          )}
          <h3 className="font-semibold text-lg mb-2">{email.subject || "(No subject)"}</h3>
          <div className="text-sm text-gray-600 space-y-1">
            <div><strong>Date:</strong> {email.date} {email.time}</div>
            <div><strong>From/To:</strong> {email.participants.map(p => p.name).join(", ")}</div>
          </div>
        </div>
        <hr className="mb-4 border-gray-200" />
        <div
          className="whitespace-pre-wrap text-sm"
          onClick={handleTextClick}
          dangerouslySetInnerHTML={{ __html: highlightedText }}
        />
        {emailText.length > 2000 && (
          <Button
            variant="link"
            size="sm"
            onClick={() => setShowFullText(!showFullText)}
            className="mt-2"
          >
            {showFullText ? "Show less" : `Show more (${emailText.length - 2000} more characters)`}
          </Button>
        )}
      </div>

      {/* Right: DocETL-generated Metadata */}
      <div className="lg:w-2/5 border border-gray-200 rounded-md p-4 overflow-y-auto max-h-[70vh] bg-white">
        <h3 className="font-semibold text-base mb-4 pb-2 border-b border-gray-200">
          DocETL-generated Metadata
        </h3>
        <div className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2 flex items-center gap-2">
              <Users className="h-4 w-4" /> Participants
            </h4>
            <div className="space-y-1">
              {email.participants.map((p, i) => (
                <div
                  key={i}
                  className="text-sm cursor-pointer text-blue-600 hover:underline"
                  onClick={() => onPersonClick(p.name)}
                >
                  {p.name} {p.email && `<${p.email}>`}
                </div>
              ))}
            </div>
          </div>

          {email.people_mentioned.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">People Mentioned</h4>
              <div className="flex flex-wrap gap-1">
                {email.people_mentioned.map((person, i) => (
                  <Badge
                    key={i}
                    variant="secondary"
                    className="cursor-pointer hover:bg-gray-300"
                    onClick={() => onPersonClick(person)}
                  >
                    {person}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {email.notable_figures.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Notable Figures</h4>
              <div className="flex flex-wrap gap-1">
                {email.notable_figures.map((person, i) => (
                  <Badge
                    key={i}
                    variant="outline"
                    className="cursor-pointer hover:bg-gray-100"
                    onClick={() => onPersonClick(person)}
                  >
                    {person}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {email.organizations.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Organizations</h4>
              <div className="text-sm text-gray-700">
                {email.organizations.join(", ")}
              </div>
            </div>
          )}

          {email.locations.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Locations</h4>
              <div className="text-sm text-gray-700">
                {email.locations.join(", ")}
              </div>
            </div>
          )}

          {(email.has_attachments || email.attachment_names.length > 0) && (
            <div>
              <h4 className="font-semibold mb-2">Attachments</h4>
              <div className="text-sm text-gray-700">
                {email.attachment_names.length > 0 ? email.attachment_names.join(", ") : "Yes"}
              </div>
            </div>
          )}

          <div>
            <h4 className="font-semibold mb-2">Summary</h4>
            <div className="text-sm text-gray-700">{email.summary}</div>
          </div>

          {email.key_quotes.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Key Quotes</h4>
              <div className="space-y-2">
                {email.key_quotes.map((quote, i) => (
                  <div key={i} className="text-sm italic border-l-2 border-gray-300 pl-2">
                    {quote}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <strong>Tone:</strong> <Badge variant="secondary">{email.tone}</Badge>
            </div>
            <div>
              <strong>Topic:</strong> <Badge variant="secondary">{email.primary_topic}</Badge>
            </div>
          </div>

          {email.potential_crimes && email.potential_crimes.trim() && (
            <div className="bg-red-50 border border-red-200 rounded p-3">
              <h4 className="font-semibold mb-2 text-red-900 flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" /> Potential Concerns
              </h4>
              <div className="text-sm text-red-800">{email.potential_crimes}</div>
              {email.evidence_strength && email.evidence_strength !== "none" && (
                <div className="text-xs mt-2">
                  <strong>Evidence Strength:</strong> {email.evidence_strength}
                </div>
              )}
              {email.crime_types && email.crime_types.length > 0 && (
                <div className="text-xs mt-2">
                  <strong>Crime Types:</strong> {email.crime_types.join(", ")}
                </div>
              )}
            </div>
          )}

          {email.cover_up && email.cover_up.trim() && (
            <div className="bg-orange-50 border border-orange-200 rounded p-3">
              <h4 className="font-semibold mb-2 text-orange-900">Cover-up Indicators</h4>
              <div className="text-sm text-orange-800">{email.cover_up}</div>
            </div>
          )}

          {email.mentions_victims && (
            <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
              <h4 className="font-semibold mb-2 text-yellow-900">Mentions Victims</h4>
              {email.victim_names.length > 0 ? (
                <div className="space-y-1">
                  {email.victim_names.map((name, i) => (
                    <div key={i} className="text-sm font-medium text-yellow-900">
                      • {name}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-yellow-800 italic">
                  Victims mentioned but not named
                </div>
              )}
            </div>
          )}

          {email.topics.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">All Topics</h4>
              <div className="flex flex-wrap gap-1">
                {email.topics.map((topic, i) => (
                  <Badge key={i} variant="outline">{topic}</Badge>
                ))}
              </div>
            </div>
          )}

          {email.phone_numbers && email.phone_numbers.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Phone Numbers</h4>
              <div className="text-sm text-gray-700">
                {email.phone_numbers.join(", ")}
              </div>
            </div>
          )}

          {email.urls && email.urls.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">URLs</h4>
              <div className="text-sm text-gray-700 break-all">
                {email.urls.map((url, i) => {
                  // Ensure URL has protocol, otherwise add https://
                  const urlWithProtocol = url.match(/^https?:\/\//)
                    ? url
                    : `https://${url}`;
                  return (
                    <div key={i}>
                      <a href={urlWithProtocol} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                        {url}
                      </a>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {email.inferred_people && email.inferred_people.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Inferred People</h4>
              <div className="space-y-2">
                {email.inferred_people.map((person, i) => (
                  <div key={i} className="text-sm border-l-2 border-blue-300 pl-2 bg-blue-50 p-2 rounded">
                    <div className="font-medium">{person.name}</div>
                    <div className="text-xs text-gray-600 mt-1">
                      <strong>Reference:</strong> {person.reference}
                    </div>
                    <div className="text-xs text-gray-600 mt-1">
                      <strong>Reasoning:</strong> {person.reasoning}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

interface StatsDashboardProps {
  emails: Email[];
}

const StatsDashboard = React.memo(({ emails }: StatsDashboardProps) => {
  const stats = useMemo(() => {
    const total = emails.length;
    const withCrimes = emails.filter(e => e.potential_crimes && e.potential_crimes.trim()).length;
    const withVictims = emails.filter(e => e.mentions_victims).length;

    const toneCounts: Record<string, number> = {};
    emails.forEach(e => {
      toneCounts[e.tone] = (toneCounts[e.tone] || 0) + 1;
    });

    return { total, withCrimes, withVictims, toneCounts };
  }, [emails]);

  const maxToneCount = Math.max(...Object.values(stats.toneCounts));

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Total Emails</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold">{stats.total}</div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Flagged with Concerns</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold text-red-600">{stats.withCrimes}</div>
          <div className="text-xs text-gray-500">{((stats.withCrimes / stats.total) * 100).toFixed(1)}% of total</div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Mention Victims</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-bold text-yellow-600">{stats.withVictims}</div>
          <div className="text-xs text-gray-500">{((stats.withVictims / stats.total) * 100).toFixed(1)}% of total</div>
        </CardContent>
      </Card>

      <Card className="md:col-span-3">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Tone Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {Object.entries(stats.toneCounts).sort((a, b) => b[1] - a[1]).map(([tone, count]) => (
              <div key={tone} className="flex items-center gap-2">
                <div className="w-20 text-sm capitalize">{tone}</div>
                <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                  <div
                    className="bg-blue-500 h-full flex items-center justify-end pr-2 text-white text-xs font-medium"
                    style={{ width: `${(count / maxToneCount) * 100}%` }}
                  >
                    {count}
                  </div>
                </div>
                <div className="w-12 text-sm text-gray-600 text-right">
                  {((count / stats.total) * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
});

const ATTRIBUTE_OPTIONS = [
  { value: "participant", label: "Participant" },
  { value: "people_mentioned", label: "People Mentioned" },
  { value: "notable_figures", label: "Notable Figures" },
  { value: "organization", label: "Organization" },
  { value: "location", label: "Location" },
  { value: "subject", label: "Subject" },
  { value: "email_text", label: "Email Content" },
  { value: "tone", label: "Tone" },
  { value: "primary_topic", label: "Primary Topic" },
  { value: "topics", label: "Topics" },
  { value: "date_range", label: "Date Range" },
];

interface PersonStats {
  name: string;
  asParticipant: number;
  asMentioned: number;
  total: number;
  emails: Email[];
}

export default function EpsteinEmailExplorer() {
  const [selectedEmail, setSelectedEmail] = useState<Email | null>(null);
  const [personModalOpen, setPersonModalOpen] = useState(false);
  const [selectedPerson, setSelectedPerson] = useState<string>("");
  const [attributeFilters, setAttributeFilters] = useState<AttributeFilter[]>([]);
  const [newFilterAttribute, setNewFilterAttribute] = useState("");
  const [newFilterValue, setNewFilterValue] = useState("");
  const [inputFilterValue, setInputFilterValue] = useState(""); // Local input state
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [showCrimesOnly, setShowCrimesOnly] = useState(false);
  const [showVictimsOnly, setShowVictimsOnly] = useState(false);
  const [filtersExpanded, setFiltersExpanded] = useState(true);
  const [emails, setEmails] = useState<Email[]>([]);
  const [totalEmails, setTotalEmails] = useState(0);
  const [isLoadingInitial, setIsLoadingInitial] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dateFrom, setDateFrom] = useState<Date | undefined>(undefined);
  const [dateTo, setDateTo] = useState<Date | undefined>(undefined);
  const [activeTab, setActiveTab] = useState<"emails" | "people">("emails");
  const [showMobileWarning, setShowMobileWarning] = useState(false);
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

  // Debounced input handler
  const handleFilterInputChange = useCallback((value: string) => {
    setInputFilterValue(value);

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      setNewFilterValue(value);
    }, 300);
  }, []);

  // Sync newFilterValue to inputFilterValue when attribute changes
  useEffect(() => {
    setInputFilterValue(newFilterValue);
  }, [newFilterAttribute]);

  // Check screen width and show mobile warning
  useEffect(() => {
    const checkScreenWidth = () => {
      const isNarrow = window.innerWidth < 1024; // 1024px is the lg breakpoint
      const hasSeenWarning = localStorage.getItem('epstein-explorer-mobile-warning-dismissed') === 'true';

      if (isNarrow && !hasSeenWarning) {
        setShowMobileWarning(true);
      }
    };

    checkScreenWidth();
    window.addEventListener('resize', checkScreenWidth);
    return () => window.removeEventListener('resize', checkScreenWidth);
  }, []);

  const handleDismissMobileWarning = () => {
    localStorage.setItem('epstein-explorer-mobile-warning-dismissed', 'true');
    setShowMobileWarning(false);
  };

  const handleDownloadInsights = () => {
    try {
      setIsDownloading(true);

      // Use the already-loaded emails data
      const compactEmails = emails.map((email) => ({
        subject: email.subject || '',
        date: email.date || '',
        participants: email.participants?.map((p) => p.name).filter(Boolean) || [],
        people_mentioned: email.people_mentioned || [],
        notable_figures: email.notable_figures || [],
        organizations: email.organizations || [],
        locations: email.locations || [],
        summary: email.summary || '',
        primary_topic: email.primary_topic || '',
        topics: email.topics || [],
        tone: email.tone || '',
        potential_crimes: email.potential_crimes || '',
        crime_types: email.crime_types || [],
        mentions_victims: email.mentions_victims || false,
        victim_names: email.victim_names || [],
        cover_up: email.cover_up || '',
      }));

      const jsonContent = JSON.stringify(compactEmails, null, 2);

      // Create blob and download
      const blob = new Blob([jsonContent], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'epstein_emails_insights.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setIsDownloading(false);
      setShowDownloadModal(true);
    } catch (err) {
      console.error('Failed to download insights:', err);
      setIsDownloading(false);
      alert('Failed to download insights. Please try again.');
    }
  };

  // Load and index emails incrementally
  React.useEffect(() => {
    const loadEmails = async () => {
      try {
        setIsLoadingInitial(true);

        // Fetch via API route (which proxies to Azure)
        const res = await fetch("/api/epstein-emails");

        if (!res.ok) {
          throw new Error(`Failed to fetch: ${res.status} ${res.statusText}`);
        }

        const allEmails = await res.json() as Email[];
        setTotalEmails(allEmails.length);

        // Process in chunks of 100 to avoid blocking the UI
        const CHUNK_SIZE = 100;
        let processedEmails: Email[] = [];

        for (let i = 0; i < allEmails.length; i += CHUNK_SIZE) {
          const chunk = allEmails.slice(i, i + CHUNK_SIZE);
          processedEmails = [...processedEmails, ...chunk];

          // Update state and yield to browser
          setEmails([...processedEmails]);

          // Allow UI to update between chunks
          await new Promise(resolve => setTimeout(resolve, 0));
        }

        setIsLoadingInitial(false);
      } catch (err) {
        console.error("Error loading emails:", err);
        setError(err instanceof Error ? err.message : String(err));
        setIsLoadingInitial(false);
      }
    };

    loadEmails();
  }, []);

  const addFilter = () => {
    if (newFilterAttribute && newFilterValue) {
      setAttributeFilters([
        ...attributeFilters,
        {
          id: Math.random().toString(),
          attribute: newFilterAttribute,
          value: newFilterValue,
        },
      ]);
      setNewFilterValue("");
      setInputFilterValue("");
    }
  };

  const removeFilter = (id: string) => {
    setAttributeFilters(attributeFilters.filter(f => f.id !== id));
  };

  const handleFilterByPerson = (person: string, attribute: string) => {
    // Add a filter for this person with the specified attribute
    setAttributeFilters([
      ...attributeFilters,
      {
        id: Math.random().toString(),
        attribute: attribute,
        value: person,
      },
    ]);
    setActiveTab("emails"); // Switch to emails tab to show filtered results
  };

  const filteredEmails = useMemo(() => {
    let result = emails;

    // Date filter
    if (dateFrom) {
      const fromStr = dateFrom.toISOString().split('T')[0];
      result = result.filter(e => e.date >= fromStr);
    }
    if (dateTo) {
      const toStr = dateTo.toISOString().split('T')[0];
      result = result.filter(e => e.date <= toStr);
    }

    // Apply attribute filters
    attributeFilters.forEach(filter => {
      const value = filter.value.toLowerCase();

      result = result.filter(e => {
        switch (filter.attribute) {
          case "participant":
            return e.participants.some(p => p.name.toLowerCase().includes(value));
          case "people_mentioned":
            return e.people_mentioned.some(p => p.toLowerCase().includes(value));
          case "notable_figures":
            return e.notable_figures.some(p => p.toLowerCase().includes(value));
          case "organization":
            return e.organizations.some(o => o.toLowerCase().includes(value));
          case "location":
            return e.locations.some(l => l.toLowerCase().includes(value));
          case "subject":
            return e.subject.toLowerCase().includes(value);
          case "email_text":
            return e.email_text.toLowerCase().includes(value);
          case "tone":
            return e.tone.toLowerCase() === value;
          case "primary_topic":
            return e.primary_topic.toLowerCase() === value;
          case "topics":
            return e.topics.some(t => t.toLowerCase().includes(value));
          default:
            return true;
        }
      });
    });

    // Crimes filter
    if (showCrimesOnly) {
      result = result.filter(e => e.potential_crimes && e.potential_crimes.trim());
    }

    // Victims filter
    if (showVictimsOnly) {
      result = result.filter(e => e.mentions_victims);
    }

    // Sort: emails with potential crimes first
    return result.sort((a, b) => {
      const aHasCrimes = a.potential_crimes && a.potential_crimes.trim() ? 1 : 0;
      const bHasCrimes = b.potential_crimes && b.potential_crimes.trim() ? 1 : 0;
      if (bHasCrimes !== aHasCrimes) return bHasCrimes - aHasCrimes;

      // Then by date (newest first)
      return b.date.localeCompare(a.date);
    });
  }, [emails, attributeFilters, showCrimesOnly, showVictimsOnly, dateFrom, dateTo]);

  // Lightweight count of unique people (always computed for tab label)
  const peopleCount = useMemo(() => {
    const uniquePeople = new Set<string>();
    filteredEmails.forEach(email => {
      email.participants.forEach(p => p.name && uniquePeople.add(p.name));
      email.people_mentioned.forEach(name => name && uniquePeople.add(name));
    });
    return uniquePeople.size;
  }, [filteredEmails]);

  const peopleStats = useMemo(() => {
    // Only compute detailed stats when on people tab to avoid expensive calculation during tab switches
    if (activeTab !== "people") {
      return [];
    }

    const stats = new Map<string, PersonStats>();

    filteredEmails.forEach(email => {
      // Track participants
      email.participants.forEach(p => {
        if (!p.name) return;
        if (!stats.has(p.name)) {
          stats.set(p.name, {
            name: p.name,
            asParticipant: 0,
            asMentioned: 0,
            total: 0,
            emails: [],
          });
        }
        const stat = stats.get(p.name)!;
        stat.asParticipant++;
      });

      // Track mentioned
      email.people_mentioned.forEach(name => {
        if (!name) return;
        if (!stats.has(name)) {
          stats.set(name, {
            name,
            asParticipant: 0,
            asMentioned: 0,
            total: 0,
            emails: [],
          });
        }
        const stat = stats.get(name)!;
        stat.asMentioned++;
      });
    });

    // Calculate total as asParticipant + asMentioned and collect unique emails
    stats.forEach((stat) => {
      stat.total = stat.asParticipant + stat.asMentioned;
      // Collect unique emails for this person
      const emailSet = new Set<string>();
      filteredEmails.forEach(email => {
        const isParticipant = email.participants.some(p => p.name === stat.name);
        const isMentioned = email.people_mentioned.some(p => p === stat.name);
        if (isParticipant || isMentioned) {
          if (!emailSet.has(email.document_id)) {
            emailSet.add(email.document_id);
            stat.emails.push(email);
          }
        }
      });
    });

    return Array.from(stats.values()).sort((a, b) => b.total - a.total);
  }, [filteredEmails, activeTab]);

  const allTones = useMemo(() => {
    return Array.from(new Set(emails.map(e => e.tone))).sort();
  }, [emails]);

  const allTopics = useMemo(() => {
    return Array.from(new Set(emails.map(e => e.primary_topic))).sort();
  }, [emails]);

  const handlePersonClick = useCallback((person: string) => {
    setSelectedPerson(person);
    setPersonModalOpen(true);
  }, []);

  const loadingProgress = totalEmails > 0 ? Math.round((emails.length / totalEmails) * 100) : 0;

  if (isLoadingInitial && emails.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-muted-foreground">
        <Loader2 className="h-8 w-8 animate-spin mb-4" />
        <div className="text-lg font-medium">Downloading email dataset...</div>
        <div className="text-sm mt-2">Fetching 2,322 emails (17.7MB)</div>
        <div className="text-xs mt-1 text-gray-500">This takes 20-30 seconds on first visit. Subsequent visits are instant.</div>
      </div>
    );
  }

  if (error) {
    return (
      <p className="text-red-600">Failed to load emails: {String(error)}</p>
    );
  }

  return (
    <div className="space-y-6">
      {isLoadingInitial && emails.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <div className="flex items-center gap-3 mb-2">
            <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
            <div className="text-sm text-blue-800 font-medium">
              Loading emails: {emails.length} of {totalEmails} ({loadingProgress}%)
            </div>
          </div>
          <div className="w-full bg-blue-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${loadingProgress}%` }}
            />
          </div>
        </div>
      )}
      {/* Download Insights Button */}
      <div className="flex justify-end mb-4">
        <Button
          onClick={handleDownloadInsights}
          disabled={isLoadingInitial || emails.length === 0 || isDownloading}
          variant="outline"
          size="sm"
          className="bg-white"
        >
          {isDownloading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Preparing download...
            </>
          ) : (
            <>
              <Download className="h-4 w-4 mr-2" />
              Download Insights for AI Chat
            </>
          )}
        </Button>
      </div>

      <StatsDashboard emails={filteredEmails} />

      {/* Filters */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base flex items-center gap-2">
              <Filter className="h-4 w-4" /> Filters
            </CardTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setFiltersExpanded(!filtersExpanded)}
            >
              {filtersExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </Button>
          </div>
        </CardHeader>
        {filtersExpanded && (
          <CardContent className="space-y-4">
            {/* Add new filter */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-2">
              <div className="md:col-span-4">
                <Label htmlFor="attribute">Attribute</Label>
                <Select value={newFilterAttribute} onValueChange={setNewFilterAttribute}>
                  <SelectTrigger id="attribute">
                    <SelectValue placeholder="Select attribute..." />
                  </SelectTrigger>
                  <SelectContent>
                    {ATTRIBUTE_OPTIONS.map(opt => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="md:col-span-6">
                <Label htmlFor="value">
                  {newFilterAttribute === "tone" || newFilterAttribute === "primary_topic" ? "Value" : "Search Value"}
                </Label>
                {newFilterAttribute === "tone" ? (
                  <Select value={newFilterValue} onValueChange={setNewFilterValue}>
                    <SelectTrigger id="value">
                      <SelectValue placeholder="Select tone..." />
                    </SelectTrigger>
                    <SelectContent>
                      {allTones.map(tone => (
                        <SelectItem key={tone} value={tone} className="capitalize">
                          {tone}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : newFilterAttribute === "primary_topic" ? (
                  <Select value={newFilterValue} onValueChange={setNewFilterValue}>
                    <SelectTrigger id="value">
                      <SelectValue placeholder="Select topic..." />
                    </SelectTrigger>
                    <SelectContent>
                      {allTopics.map(topic => (
                        <SelectItem key={topic} value={topic}>
                          {topic}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id="value"
                    placeholder={newFilterAttribute ? `Search ${ATTRIBUTE_OPTIONS.find(o => o.value === newFilterAttribute)?.label.toLowerCase()}...` : "Select an attribute first"}
                    value={inputFilterValue}
                    onChange={(e) => handleFilterInputChange(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        // Clear debounce and immediately set value
                        if (debounceTimerRef.current) {
                          clearTimeout(debounceTimerRef.current);
                        }
                        setNewFilterValue(inputFilterValue);
                        setTimeout(() => addFilter(), 0);
                      }
                    }}
                    disabled={!newFilterAttribute}
                  />
                )}
              </div>

              <div className="md:col-span-2 flex items-end">
                <Button
                  onClick={addFilter}
                  disabled={!newFilterAttribute || !newFilterValue}
                  className="w-full"
                >
                  <Plus className="h-4 w-4 mr-2" /> Add
                </Button>
              </div>
            </div>

            {/* Active filters */}
            {attributeFilters.length > 0 && (
              <div>
                <Label className="mb-2 block">Active Filters:</Label>
                <div className="flex flex-wrap gap-2">
                  {attributeFilters.map(filter => (
                    <Badge key={filter.id} variant="secondary" className="pl-3 pr-1 py-1">
                      <span className="mr-2">
                        <strong>{ATTRIBUTE_OPTIONS.find(o => o.value === filter.attribute)?.label}:</strong> {filter.value}
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-4 w-4 p-0 hover:bg-transparent"
                        onClick={() => removeFilter(filter.id)}
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Date filters */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t">
              <div>
                <Label>From Date</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      className={cn(
                        "w-full justify-start text-left font-normal",
                        !dateFrom && "text-muted-foreground"
                      )}
                    >
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {dateFrom ? dateFrom.toLocaleDateString() : "Pick a date"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0" align="start">
                    <Calendar
                      mode="single"
                      selected={dateFrom}
                      onSelect={setDateFrom}
                      captionLayout="dropdown"
                      fromYear={2000}
                      toYear={2026}
                      initialFocus
                    />
                  </PopoverContent>
                </Popover>
                {dateFrom && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="mt-1"
                    onClick={() => setDateFrom(undefined)}
                  >
                    Clear
                  </Button>
                )}
              </div>
              <div>
                <Label>To Date</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      className={cn(
                        "w-full justify-start text-left font-normal",
                        !dateTo && "text-muted-foreground"
                      )}
                    >
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {dateTo ? dateTo.toLocaleDateString() : "Pick a date"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0" align="start">
                    <Calendar
                      mode="single"
                      selected={dateTo}
                      onSelect={setDateTo}
                      captionLayout="dropdown"
                      fromYear={2000}
                      toYear={2026}
                      initialFocus
                    />
                  </PopoverContent>
                </Popover>
                {dateTo && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="mt-1"
                    onClick={() => setDateTo(undefined)}
                  >
                    Clear
                  </Button>
                )}
              </div>
            </div>

            {/* Quick filters */}
            <div className="flex gap-4 pt-2 border-t">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="crimes"
                  checked={showCrimesOnly}
                  onCheckedChange={(checked) => setShowCrimesOnly(checked as boolean)}
                />
                <Label htmlFor="crimes" className="cursor-pointer">
                  Show only emails with potential concerns
                </Label>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="victims"
                  checked={showVictimsOnly}
                  onCheckedChange={(checked) => setShowVictimsOnly(checked as boolean)}
                />
                <Label htmlFor="victims" className="cursor-pointer">
                  Show only emails mentioning victims
                </Label>
              </div>
            </div>

            <div className="text-sm text-gray-600">
              Showing {filteredEmails.length} of {emails.length} emails
            </div>
          </CardContent>
        )}
      </Card>

      {/* Tabs for Emails vs People */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as "emails" | "people")}>
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="emails">
            <Mail className="h-4 w-4 mr-2" />
            Emails ({filteredEmails.length})
          </TabsTrigger>
          <TabsTrigger value="people">
            <UserCircle className="h-4 w-4 mr-2" />
            People ({peopleCount})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="emails" className="mt-4">
          {/* Email List and Detail */}
          <div className="flex flex-col lg:flex-row gap-4">
        {/* Email List */}
        <div className="lg:w-1/3 border rounded-md overflow-hidden">
          <ScrollArea className="h-[70vh]">
            <Table>
              <TableHeader className="sticky top-0 bg-white z-10">
                <TableRow>
                  <TableHead>Email</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredEmails.map((email) => (
                  <TableRow
                    key={email.document_id}
                    className={`cursor-pointer hover:bg-muted/50 ${
                      selectedEmail?.document_id === email.document_id ? "bg-primary/10" : ""
                    }`}
                    onClick={() => setSelectedEmail(email)}
                  >
                    <TableCell>
                      <div className="space-y-1">
                        <div className="font-medium text-sm line-clamp-2">
                          {email.subject || "(No subject)"}
                        </div>
                        <div className="text-xs text-gray-500 flex items-center gap-1">
                          <CalendarIcon className="h-3 w-3" />
                          {email.date}
                        </div>
                        <div className="text-xs text-gray-600">
                          {email.participants.map(p => p.name).join(", ")}
                        </div>
                        <div className="flex gap-1 flex-wrap">
                          {email.potential_crimes && email.potential_crimes.trim() && (
                            <Badge variant="destructive" className="text-xs">
                              <AlertTriangle className="h-3 w-3 mr-1" /> Flagged
                            </Badge>
                          )}
                          {email.mentions_victims && (
                            <Badge
                              variant="secondary"
                              className="text-xs bg-yellow-100 text-yellow-800"
                              title={email.victim_names.length > 0 ? email.victim_names.join(", ") : "Victims mentioned"}
                            >
                              Victims{email.victim_names.length > 0 && ` (${email.victim_names.length})`}
                            </Badge>
                          )}
                          <Badge variant="outline" className="text-xs capitalize">
                            {email.tone}
                          </Badge>
                        </div>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </ScrollArea>
        </div>

            {/* Email Detail */}
            {selectedEmail ? (
              <EmailDetailView email={selectedEmail} onPersonClick={handlePersonClick} />
            ) : (
              <div className="flex-1 border border-gray-200 rounded-md p-8 text-center text-gray-500 bg-white">
                <p>Select an email from the list to view its details</p>
              </div>
            )}
          </div>
        </TabsContent>

        <TabsContent value="people" className="mt-4">
          {/* People List */}
          <div className="border rounded-md overflow-hidden">
            <ScrollArea className="h-[70vh]">
              <Table>
                <TableHeader className="sticky top-0 bg-white z-10">
                  <TableRow>
                    <TableHead>Person</TableHead>
                    <TableHead className="text-right">As Participant</TableHead>
                    <TableHead className="text-right">Mentioned</TableHead>
                    <TableHead className="text-right">Total</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {peopleStats.map((person, idx) => (
                    <TableRow key={idx} className="hover:bg-muted/50">
                      <TableCell className="font-medium">{person.name}</TableCell>
                      <TableCell className="text-right">
                        <Badge variant="secondary">{person.asParticipant}</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="secondary">{person.asMentioned}</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Badge variant="default">{person.total}</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex gap-2 justify-end">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handlePersonClick(person.name)}
                          >
                            View Details
                          </Button>
                          {person.asParticipant > 0 && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                handleFilterByPerson(person.name, "participant");
                                setActiveTab("emails");
                              }}
                              title="Filter emails where this person is a participant"
                            >
                              <Filter className="h-3 w-3 mr-1" />
                              As Participant
                            </Button>
                          )}
                          {person.asMentioned > 0 && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                handleFilterByPerson(person.name, "people_mentioned");
                                setActiveTab("emails");
                              }}
                              title="Filter emails where this person is mentioned"
                            >
                              <Filter className="h-3 w-3 mr-1" />
                              As Mentioned
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          </div>
        </TabsContent>
      </Tabs>

      {/* Pipeline Artifacts */}
      <div className="mt-8 pt-6 border-t border-gray-200">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Pipeline Artifacts</h3>
        <div className="flex flex-wrap gap-3">
          <a
            href="https://docetlcloudbank.blob.core.windows.net/demos/emails_dataset.json"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Button variant="outline" size="sm" className="bg-white">
              <Download className="h-4 w-4 mr-2" />
              Download Pipeline Input
            </Button>
          </a>
          <a
            href="https://docetlcloudbank.blob.core.windows.net/demos/emails_with_metadata.json"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Button variant="outline" size="sm" className="bg-white">
              <FileJson className="h-4 w-4 mr-2" />
              Download Pipeline Output
            </Button>
          </a>
          <a
            href="/demos/epstein_email_pipeline.yaml"
            download
          >
            <Button variant="outline" size="sm" className="bg-white">
              <FileJson className="h-4 w-4 mr-2" />
              Download Pipeline YAML
            </Button>
          </a>
        </div>
      </div>

      {/* Person Detail Modal */}
      <PersonDetailModal
        person={selectedPerson}
        emails={emails}
        open={personModalOpen}
        onClose={() => setPersonModalOpen(false)}
        onFilterByPerson={handleFilterByPerson}
      />

      {/* Mobile Warning Modal */}
      <Dialog open={showMobileWarning} onOpenChange={setShowMobileWarning}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-600" />
              Best Viewed on Desktop
            </DialogTitle>
            <DialogDescription className="space-y-3 pt-2">
              <p>
                This email explorer is designed for desktop or tablet screens.
                For the best experience exploring 2,322+ emails with advanced filters,
                we recommend using a computer with a wider screen.
              </p>
              <p className="text-sm">
                On mobile, some features may be difficult to use or hidden.
              </p>
            </DialogDescription>
          </DialogHeader>
          <div className="flex gap-2 mt-4">
            <Button
              onClick={handleDismissMobileWarning}
              variant="default"
              className="flex-1"
            >
              Got it, continue anyway
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Download for AI Chat Modal */}
      <Dialog open={showDownloadModal} onOpenChange={setShowDownloadModal}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Check className="h-5 w-5 text-green-600" />
              File Downloaded!
            </DialogTitle>
            <DialogDescription className="space-y-4 pt-3">
              <p className="text-base">
                <strong>epstein_emails_insights.txt</strong> has been downloaded to your computer.
              </p>
              <p className="text-sm text-gray-600">
                This file contains DocETL-generated metadata for 2,322 emails (subjects, participants, summaries, topics, flagged concerns, etc.)
              </p>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="font-semibold text-blue-900 mb-2">How to chat with AI about this data:</p>
                <ol className="list-decimal list-inside space-y-3 text-sm text-blue-800">
                  <li>
                    <strong>Open ChatGPT, Claude, or Gemini</strong>
                    <ul className="list-disc list-inside ml-4 mt-1 text-xs space-y-1">
                      <li>ChatGPT: Click the <strong>📎 attach file</strong> icon</li>
                      <li>Claude: Click <strong>Add content</strong> or drag and drop</li>
                      <li>Gemini: Click the <strong>Add file</strong> button</li>
                    </ul>
                  </li>
                  <li><strong>Upload</strong> the downloaded <code className="bg-white px-1 rounded">epstein_emails_insights.txt</code> file</li>
                  <li>
                    <strong>Ask questions</strong> like:
                    <ul className="list-disc list-inside ml-4 mt-1 space-y-1">
                      <li>&quot;Which emails mention [person/organization]?&quot;</li>
                      <li>&quot;What are the most common topics?&quot;</li>
                      <li>&quot;Summarize emails flagged with concerns&quot;</li>
                      <li>&quot;Find patterns in communication over time&quot;</li>
                    </ul>
                  </li>
                </ol>
              </div>
              <p className="text-sm text-gray-600">
                The AI can analyze the structured metadata to find patterns, answer questions, and help with investigative research.
              </p>
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-2 mt-4">
            <Button onClick={() => setShowDownloadModal(false)}>
              Got it!
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
