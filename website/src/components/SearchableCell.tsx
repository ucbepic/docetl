import React, { useState, useEffect, useRef } from "react";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { MarkdownCell } from "@/components/MarkdownCell";

interface SearchableCellProps {
  content: string;
  isResizing: boolean;
  children?: (searchTerm: string) => React.ReactNode;
}

export const SearchableCell = React.memo(
  ({ content, isResizing, children }: SearchableCellProps) => {
    const [searchTerm, setSearchTerm] = useState("");
    const [highlightedContent, setHighlightedContent] = useState(content);
    const [currentMatchIndex, setCurrentMatchIndex] = useState(0);
    const [matchCount, setMatchCount] = useState(0);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
      if (!searchTerm) {
        setHighlightedContent(content);
        setMatchCount(0);
        setCurrentMatchIndex(0);
        return;
      }

      try {
        const regex = new RegExp(`(${searchTerm})`, "gi");
        const matches = content.match(regex);
        const matchesCount = matches ? matches.length : 0;
        setMatchCount(matchesCount);

        if (matchesCount > 0) {
          const highlighted = content.replace(
            regex,
            (match) => `<mark class="search-match">${match}</mark>`
          );
          setHighlightedContent(highlighted);

          setTimeout(() => {
            if (containerRef.current) {
              const marks =
                containerRef.current.getElementsByClassName("search-match");
              if (marks.length > 0 && currentMatchIndex < marks.length) {
                marks[currentMatchIndex].scrollIntoView({
                  behavior: "smooth",
                  block: "center",
                });
              }
            }
          }, 100);
        } else {
          setHighlightedContent(content);
        }
      } catch {
        setHighlightedContent(content);
        setMatchCount(0);
      }
    }, [searchTerm, content, currentMatchIndex]);

    const navigateMatches = (direction: "next" | "prev") => {
      if (matchCount === 0) return;

      if (direction === "next") {
        setCurrentMatchIndex((prev) => (prev + 1) % matchCount);
      } else {
        setCurrentMatchIndex((prev) => (prev - 1 + matchCount) % matchCount);
      }
    };

    useEffect(() => {
      const styleId = "search-match-style";
      let styleElement = document.getElementById(styleId) as HTMLStyleElement;

      if (!styleElement) {
        styleElement = document.createElement("style");
        styleElement.id = styleId;
        document.head.appendChild(styleElement);
      }

      styleElement.textContent = `
        mark.search-match {
          background-color: hsl(var(--primary) / 0.2);
          color: inherit;
          padding: 0;
          border-radius: 2px;
        }
        mark.search-match:nth-of-type(${currentMatchIndex + 1}) {
          background-color: hsl(var(--primary) / 0.5);
        }
      `;

      return () => {
        if (styleElement && styleElement.parentNode) {
          styleElement.parentNode.removeChild(styleElement);
        }
      };
    }, [currentMatchIndex]);

    return (
      <div className="relative">
        <div className="sticky top-0 z-10 bg-background/80 backdrop-blur-sm rounded-md">
          <div className="flex items-center gap-2 p-1">
            <Search className="h-3 w-3 text-muted-foreground" />
            <Input
              placeholder="Search in cell..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentMatchIndex(0);
              }}
              className="h-6 text-xs border-none shadow-none focus-visible:ring-0"
            />
            {matchCount > 0 && (
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => navigateMatches("prev")}
                  disabled={currentMatchIndex === 0}
                  className="h-5 w-5 p-0"
                >
                  <ChevronLeft className="h-3 w-3" />
                </Button>
                <span className="text-xs text-muted-foreground whitespace-nowrap">
                  {currentMatchIndex + 1}/{matchCount}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => navigateMatches("next")}
                  disabled={currentMatchIndex === matchCount - 1}
                  className="h-5 w-5 p-0"
                >
                  <ChevronRight className="h-3 w-3" />
                </Button>
              </div>
            )}
          </div>
        </div>
        <div ref={containerRef}>
          {children ? (
            searchTerm ? (
              <div
                dangerouslySetInnerHTML={{ __html: highlightedContent }}
                className="prose prose-sm max-w-none"
              />
            ) : (
              children(searchTerm)
            )
          ) : isResizing ? (
            <div>{content}</div>
          ) : searchTerm ? (
            <div
              dangerouslySetInnerHTML={{ __html: highlightedContent }}
              className="prose prose-sm max-w-none"
            />
          ) : (
            <MarkdownCell content={content} />
          )}
        </div>
      </div>
    );
  }
);

SearchableCell.displayName = "SearchableCell";
