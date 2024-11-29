import React, { useState, useEffect, useCallback } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { SearchableCell } from "@/components/SearchableCell";
import { PrettyJSON } from "@/components/PrettyJSON";
import { RowNavigator } from "@/components/RowNavigator";
import { ChevronDown, Eye, ChevronLeft, ChevronRight } from "lucide-react";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { useBookmarkContext } from "@/contexts/BookmarkContext";
import { Textarea } from "@/components/ui/textarea";
import { UserNote } from "@/app/types";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/components/ui/tooltip";

interface ObservabilityIndicatorProps {
  row: Record<string, unknown>;
  currentOperation: string;
}

const ObservabilityIndicator = React.memo(
  ({ row, currentOperation }: ObservabilityIndicatorProps) => {
    const observabilityEntries = Object.entries(row).filter(
      ([key]) => key === `_observability_${currentOperation}`
    );

    if (observabilityEntries.length === 0) return null;

    return (
      <HoverCard>
        <HoverCardTrigger asChild>
          <div className="cursor-help">
            <Eye className="h-4 w-4 text-muted-foreground hover:text-primary" />
          </div>
        </HoverCardTrigger>
        <HoverCardContent
          className="w-[800px] max-h-[600px] overflow-auto"
          side="right"
          align="start"
        >
          <div className="space-y-4">
            <h3 className="text-lg font-semibold border-b pb-2">
              LLM Call(s) for {currentOperation}
            </h3>
            <div className="space-y-2">
              {observabilityEntries.map(([key, value]) => (
                <div key={key} className="flex flex-col gap-1">
                  <div className="text-sm text-muted-foreground">
                    {typeof value === "object"
                      ? JSON.stringify(value, null, 2)
                      : String(value)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </HoverCardContent>
      </HoverCard>
    );
  }
);
ObservabilityIndicator.displayName = "ObservabilityIndicator";

export interface ColumnDialogProps<T extends Record<string, unknown>> {
  isOpen: boolean;
  onClose: () => void;
  columnId: string;
  columnHeader: string;
  data: T[];
  currentIndex: number;
  onNavigate: (direction: "prev" | "next") => void;
  onJumpToRow: (index: number) => void;
  currentOperation: string;
}

export function ColumnDialog<T extends Record<string, unknown>>({
  isOpen,
  onClose,
  columnId,
  columnHeader,
  data,
  currentIndex,
  onNavigate,
  onJumpToRow,
  currentOperation,
}: ColumnDialogProps<T>) {
  const [splitView, setSplitView] = useState(false);
  const [compareIndex, setCompareIndex] = useState<number | null>(null);
  const [expandedFields, setExpandedFields] = useState<string[]>([]);
  const [isAllExpanded, setIsAllExpanded] = useState(false);
  const [feedbackColor, setFeedbackColor] = useState("#FF0000");
  const [showPreviousNotes, setShowPreviousNotes] = useState(false);

  const currentRow = data[currentIndex];
  const compareRow = compareIndex !== null ? data[compareIndex] : null;
  const currentValue = currentRow[columnId];
  const compareValue = compareRow?.[columnId];

  const otherFields = Object.entries(currentRow)
    .filter(([key]) => key !== columnId && !key.startsWith("_"))
    .sort((a, b) => a[0].localeCompare(b[0]));

  const toggleField = (fieldKey: string) => {
    setExpandedFields((prev) =>
      prev.includes(fieldKey)
        ? prev.filter((k) => k !== fieldKey)
        : [...prev, fieldKey]
    );
  };

  const toggleAllFields = () => {
    if (isAllExpanded) {
      setExpandedFields([]);
    } else {
      setExpandedFields(otherFields.map(([key]) => key));
    }
    setIsAllExpanded(!isAllExpanded);
  };

  const renderContent = (value: unknown) => {
    if (value === null || value === undefined) {
      return <span className="text-muted-foreground">No value</span>;
    }

    if (typeof value === "object") {
      return (
        <SearchableCell
          content={JSON.stringify(value, null, 2)}
          isResizing={false}
        >
          {(searchTerm) => (searchTerm ? null : <PrettyJSON data={value} />)}
        </SearchableCell>
      );
    }

    if (typeof value === "string") {
      return <SearchableCell content={value} isResizing={false} />;
    }

    return String(value);
  };

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLInputElement
      ) {
        return; // Don't handle shortcuts when typing in inputs
      }

      if (e.key === "ArrowLeft") {
        onNavigate("prev");
      } else if (e.key === "ArrowRight") {
        onNavigate("next");
      }
    },
    [onNavigate]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const renderRowContent = (row: T | null, value: unknown) => {
    if (!row) return null;
    const { addBookmark, getNotesForRowAndColumn } = useBookmarkContext();

    const handleSubmitFeedback = (feedbackText: string) => {
      if (!feedbackText.trim()) return;

      const filteredRowContent = Object.fromEntries(
        Object.entries(row).filter(([key]) => !key.startsWith("_observability"))
      );

      const feedback: UserNote[] = [
        {
          id: Date.now().toString(),
          note: feedbackText,
          metadata: {
            columnId,
            rowIndex: currentIndex,
            mainColumnValue: row[columnId],
            rowContent: filteredRowContent,
            operationName: currentOperation,
          },
        },
      ];

      addBookmark(feedbackColor, feedback);
    };

    const otherFields = Object.entries(row)
      .filter(([key]) => key !== columnId && !key.startsWith("_"))
      .sort((a, b) => a[0].localeCompare(b[0]));

    const existingNotes = getNotesForRowAndColumn(currentIndex, columnId);

    return (
      <ResizablePanelGroup direction="horizontal" className="h-full">
        <ResizablePanel defaultSize={75} minSize={30}>
          <div className="h-full flex">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex-none flex items-stretch w-8">
                    <Button
                      variant="ghost"
                      onClick={() => onNavigate("prev")}
                      className="h-full w-8 rounded-none hover:bg-muted/20 flex items-center justify-center bg-muted/5"
                      disabled={currentIndex === 0}
                      aria-label="Previous example (Left arrow key)"
                    >
                      <ChevronLeft className="h-12 w-12 absolute" />
                    </Button>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="right">
                  <p>Previous example</p>
                  <p className="text-xs text-muted-foreground">
                    Left arrow key
                  </p>
                </TooltipContent>
              </Tooltip>

              <div className="flex-1 overflow-auto">
                <div className="px-4 py-2">{renderContent(value)}</div>
              </div>

              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex-none flex items-stretch w-8">
                    <Button
                      variant="ghost"
                      onClick={() => onNavigate("next")}
                      className="h-full w-8 rounded-none hover:bg-muted/20 flex items-center justify-center bg-muted/5"
                      disabled={currentIndex === data.length - 1}
                      aria-label="Next example (Right arrow key)"
                    >
                      <ChevronRight className="h-12 w-12 absolute" />
                    </Button>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="left">
                  <p>Next example</p>
                  <p className="text-xs text-muted-foreground">
                    Right arrow key
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel defaultSize={25} minSize={20}>
          <ResizablePanelGroup direction="vertical">
            <ResizablePanel defaultSize={60} minSize={30}>
              <div className="h-full overflow-auto bg-muted/10">
                <div className="sticky top-0 bg-background/95 backdrop-blur-sm z-10 p-2 border-b">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-sm">Other Keys</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={toggleAllFields}
                      className="h-7 px-2"
                    >
                      <ChevronDown
                        className={`h-4 w-4 transition-transform ${
                          isAllExpanded ? "rotate-180" : ""
                        }`}
                      />
                      <span className="ml-1 text-xs">
                        {isAllExpanded ? "Collapse" : "Expand"}
                      </span>
                    </Button>
                  </div>
                </div>
                <div className="divide-y divide-border">
                  {otherFields.map(([key, value]) => (
                    <div key={key} className="bg-background/50">
                      <button
                        onClick={() => toggleField(key)}
                        className="w-full text-left p-3 hover:bg-muted/30 flex items-center justify-between"
                      >
                        <span className="font-medium text-sm truncate">
                          {key}
                        </span>
                        <ChevronDown
                          className={`h-4 w-4 flex-shrink-0 transition-transform ${
                            expandedFields.includes(key) ? "rotate-180" : ""
                          }`}
                        />
                      </button>
                      {expandedFields.includes(key) && (
                        <div className="p-3 bg-muted/10">
                          {renderContent(value)}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={40} minSize={20}>
              <div className="h-full bg-muted/10 flex flex-col">
                <div className="flex-none p-2 border-b">
                  <h3 className="text-sm font-medium">Feedback</h3>
                  <p className="text-xs text-muted-foreground">
                    Share your thoughts on this output
                  </p>
                </div>
                <div className="flex-1 overflow-auto p-2">
                  {existingNotes.length > 0 && (
                    <div className="mb-2">
                      <button
                        onClick={() => setShowPreviousNotes(!showPreviousNotes)}
                        className="flex items-center justify-between w-full text-sm font-medium text-muted-foreground hover:text-foreground mb-2"
                      >
                        <span>Previous Notes ({existingNotes.length})</span>
                        <ChevronDown
                          className={`h-4 w-4 transition-transform ${
                            showPreviousNotes ? "rotate-180" : ""
                          }`}
                        />
                      </button>
                      {showPreviousNotes && (
                        <div className="space-y-2 mb-2">
                          {existingNotes.map((note) => (
                            <div
                              key={note.id}
                              className="p-2 rounded-lg bg-muted/20 text-sm"
                            >
                              <span className="italic">"{note.note}"</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  <div className="flex-none">
                    <Textarea
                      placeholder="e.g., I don't like how short these outputs are"
                      className="min-h-[80px] bg-transparent resize-none"
                    />
                  </div>
                  <div className="flex-none mt-2">
                    <Select
                      value={feedbackColor}
                      onValueChange={setFeedbackColor}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue>
                          <div className="flex items-center">
                            <div
                              className="w-4 h-4 rounded-full mr-2"
                              style={{ backgroundColor: feedbackColor }}
                            />
                            {feedbackColor === "#FF0000"
                              ? "Red"
                              : feedbackColor === "#00FF00"
                              ? "Green"
                              : feedbackColor === "#0000FF"
                              ? "Blue"
                              : feedbackColor === "#FFFF00"
                              ? "Yellow"
                              : feedbackColor === "#FF00FF"
                              ? "Magenta"
                              : "Cyan"}
                          </div>
                        </SelectValue>
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="#FF0000">
                          <div className="flex items-center">
                            <div className="w-4 h-4 rounded-full bg-[#FF0000] mr-2" />
                            Red
                          </div>
                        </SelectItem>
                        <SelectItem value="#00FF00">
                          <div className="flex items-center">
                            <div className="w-4 h-4 rounded-full bg-[#00FF00] mr-2" />
                            Green
                          </div>
                        </SelectItem>
                        <SelectItem value="#0000FF">
                          <div className="flex items-center">
                            <div className="w-4 h-4 rounded-full bg-[#0000FF] mr-2" />
                            Blue
                          </div>
                        </SelectItem>
                        <SelectItem value="#FFFF00">
                          <div className="flex items-center">
                            <div className="w-4 h-4 rounded-full bg-[#FFFF00] mr-2" />
                            Yellow
                          </div>
                        </SelectItem>
                        <SelectItem value="#FF00FF">
                          <div className="flex items-center">
                            <div className="w-4 h-4 rounded-full bg-[#FF00FF] mr-2" />
                            Magenta
                          </div>
                        </SelectItem>
                        <SelectItem value="#00FFFF">
                          <div className="flex items-center">
                            <div className="w-4 h-4 rounded-full bg-[#00FFFF] mr-2" />
                            Cyan
                          </div>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex-none mt-2">
                    <Button
                      className="w-full"
                      onClick={(e) => {
                        const textarea = (e.target as HTMLElement)
                          .closest(".flex-col")
                          ?.querySelector("textarea");
                        if (textarea) {
                          handleSubmitFeedback(textarea.value);
                          textarea.value = "";
                        }
                      }}
                    >
                      Submit Feedback
                    </Button>
                  </div>
                </div>
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        </ResizablePanel>
      </ResizablePanelGroup>
    );
  };

  return (
    <Dialog open={isOpen} onOpenChange={() => onClose()}>
      <DialogContent className="max-w-[95vw] max-h-[95vh] w-full h-full flex flex-col p-0 bg-background rounded-lg overflow-hidden">
        <div className="flex flex-col h-full">
          <div className="flex-none flex items-center justify-between px-6 py-3 border-b bg-background">
            <div className="flex items-center gap-4 flex-1">
              <h2 className="text-lg font-semibold">{columnHeader}</h2>
              <div className="flex items-center gap-4">
                <ObservabilityIndicator
                  row={currentRow}
                  currentOperation={currentOperation}
                />
                <RowNavigator
                  currentRow={currentIndex}
                  totalRows={data.length}
                  onNavigate={onNavigate}
                  onJumpToRow={onJumpToRow}
                />
                <span className="text-sm text-muted-foreground">
                  Use ← → arrow keys to navigate
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setSplitView(!splitView);
                    if (!splitView && compareIndex === null) {
                      setCompareIndex(
                        Math.min(currentIndex + 1, data.length - 1)
                      );
                    }
                  }}
                >
                  {splitView ? "Single View" : "Split View"}
                </Button>
              </div>
            </div>
          </div>

          <div className="flex-1 min-h-0">
            {splitView ? (
              <ResizablePanelGroup direction="vertical" className="h-full">
                <ResizablePanel defaultSize={50} minSize={20}>
                  {renderRowContent(currentRow, currentValue)}
                </ResizablePanel>
                <ResizableHandle withHandle />
                <ResizablePanel defaultSize={50} minSize={20}>
                  <div className="flex flex-col h-full">
                    <div className="flex-none flex items-center px-6 py-1.5 border-b bg-muted/30">
                      <div className="flex items-center gap-4">
                        <ObservabilityIndicator
                          row={compareRow ?? {}}
                          currentOperation={currentOperation}
                        />
                        <RowNavigator
                          currentRow={compareIndex ?? 0}
                          totalRows={data.length}
                          onNavigate={(direction) => {
                            setCompareIndex((prev) =>
                              direction === "next"
                                ? Math.min((prev ?? 0) + 1, data.length - 1)
                                : Math.max((prev ?? 0) - 1, 0)
                            );
                          }}
                          onJumpToRow={(index) => setCompareIndex(index)}
                          label="Compare Row"
                          compact={true}
                        />
                      </div>
                    </div>
                    <div className="flex-1 min-h-0">
                      {renderRowContent(compareRow, compareValue)}
                    </div>
                  </div>
                </ResizablePanel>
              </ResizablePanelGroup>
            ) : (
              <div className="h-full">
                {renderRowContent(currentRow, currentValue)}
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
