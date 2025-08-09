"use client";

import React, { useState } from "react";
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
import { Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Contract {
  name: string;
  contents: string;
  extracted_red_flags: string[];
  _rank?: number; // ranking value from pipeline
  // Fallback for datasets that use a different key
  red_flags?: string[];
}

export default function LeaseContractExplorer() {
  const {
    data: contracts = [] as Contract[],
    isLoading,
    error,
  } = useQuery({
    queryKey: ["lease-contracts"],
    queryFn: async () => {
      const res = await fetch("/api/lease-contracts");
      if (!res.ok) throw new Error(res.statusText);
      return res.json() as Promise<Contract[]>;
    },
    staleTime: 1000 * 60 * 5,
  });

  const [selected, setSelected] = useState<Contract | null>(null);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-6 text-muted-foreground">
        <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading contracts...
      </div>
    );
  }

  if (error) {
    return (
      <p className="text-red-600">Failed to load contracts: {String(error)}</p>
    );
  }

  // Ensure we always have the red flag list populated
  const normalized = contracts.map((c) => ({
    ...c,
    extracted_red_flags:
      c.extracted_red_flags || c.red_flags || ([] as string[]),
  }));

  const sorted = [...normalized].sort((a, b) => {
    const rankA = a._rank ?? Number.MAX_SAFE_INTEGER;
    const rankB = b._rank ?? Number.MAX_SAFE_INTEGER;
    return rankA - rankB; // ascending (lower rank = more severe)
  });

  return (
    <div className="flex flex-col md:flex-row gap-4">
      {/* Left Pane: Contract List */}
      <div className="md:w-1/4 lg:w-1/5 border rounded-md overflow-y-auto max-h-[75vh]">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Contract</TableHead>
              <TableHead className="text-right"># Flags</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sorted.map((contract, idx) => (
              <TableRow
                key={idx}
                className={`cursor-pointer hover:bg-muted/50 ${
                  selected?.name === contract.name ? "bg-primary/10" : ""
                }`}
                onClick={() => setSelected(contract)}
              >
                <TableCell className="font-medium whitespace-normal break-words max-w-[180px]">
                  {(() => {
                    const match = contract.name.match(/lease_.+/i);
                    return match ? match[0] : contract.name;
                  })()}
                </TableCell>
                <TableCell className="text-right">
                  <Badge variant="secondary">
                    {contract.extracted_red_flags.length}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* Middle + Right Panes */}
      {selected ? (
        <ContractViewer contract={selected} />
      ) : (
        <div className="flex-1 border border-gray-200 rounded-md p-4 text-gray-500 bg-white">
          Select a contract on the left to view its contents.
        </div>
      )}
    </div>
  );
}

function ContractViewer({ contract }: { contract: Contract }) {
  const docRef = React.useRef<HTMLDivElement>(null);

  const highlighted = React.useMemo(() => {
    const flags = contract.extracted_red_flags || contract.red_flags || [];
    const escapeRegex = (s: string) => s.replace(/[-/$*+?.()|[\]{}]/g, "\\$&");

    let html = contract.contents;
    const docLower = contract.contents.toLowerCase();

    flags.forEach((flag, idx) => {
      // Build best matching phrase: longest seq of >=5 words from flag that appears in doc
      const flagTokens = flag.toLowerCase().match(/\b\w+\b/g) ?? [];
      let matchPhrase: string | null = null;

      for (let len = flagTokens.length; len >= 5 && !matchPhrase; len--) {
        for (let start = 0; start <= flagTokens.length - len; start++) {
          const phraseTokens = flagTokens.slice(start, start + len);
          const phrase = phraseTokens.join(" ");
          if (docLower.includes(phrase)) {
            matchPhrase = phrase;
            break;
          }
        }
      }

      if (matchPhrase) {
        const idxPos = docLower.indexOf(matchPhrase);
        if (idxPos !== -1) {
          // Determine snippet bounds ±5 words in original document
          const getSnippetBounds = (
            text: string,
            startIdx: number,
            matchLen: number,
            numWords: number
          ) => {
            let s = startIdx;
            let count = 0;
            while (s > 0 && count < numWords) {
              s--;
              if (/\s/.test(text[s])) count++;
            }
            let e = startIdx + matchLen;
            count = 0;
            while (e < text.length && count < numWords) {
              if (/\s/.test(text[e])) count++;
              e++;
            }
            return { s, e };
          };

          const { s, e } = getSnippetBounds(
            contract.contents,
            idxPos,
            matchPhrase.length,
            5
          );
          const snippet = contract.contents.slice(s, e);
          const escapedSnippet = escapeRegex(snippet);
          html = html.replace(
            new RegExp(escapeRegex(snippet)),
            `<mark id="flag-${idx}" class="bg-red-200">${escapedSnippet}</mark>`
          );
          return; // proceed to next flag
        }
      }

      // fallback highlight entire flag text
      const escapedFlag = escapeRegex(flag);
      html = html.replace(
        new RegExp(escapedFlag, "i"),
        `<mark id="flag-${idx}" class="bg-red-200">$&</mark>`
      );
    });

    return html;
  }, [contract]);

  const handleScrollTo = (idx: number) => {
    const el = docRef.current?.querySelector(
      `#flag-${idx}`
    ) as HTMLElement | null;
    if (el && docRef.current) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  return (
    <div className="flex flex-col md:flex-row gap-4 flex-1">
      {/* Middle: Document Text */}
      <div
        ref={docRef}
        className="flex-1 md:w-5/12 border border-gray-200 rounded-md p-3 overflow-y-auto max-h-[75vh] whitespace-pre-wrap text-sm bg-white text-gray-800"
        dangerouslySetInnerHTML={{ __html: highlighted }}
      />

      {/* Right: Red Flags List */}
      <div className="md:w-1/3 lg:w-1/4 border border-gray-200 rounded-md p-2 overflow-y-auto max-h-[75vh] bg-white">
        <Card className="shadow-none border-0">
          <CardHeader className="pb-2">
            <CardTitle className="text-base text-gray-900">
              Red Flags ({contract.extracted_red_flags.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1 text-sm">
              {contract.extracted_red_flags.map((flag, i) => (
                <div
                  key={i}
                  className="cursor-pointer hover:text-blue-600 break-words text-gray-700"
                  onClick={() => handleScrollTo(i)}
                >
                  {flag}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
