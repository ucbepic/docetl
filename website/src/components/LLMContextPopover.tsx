"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { usePipelineContext } from "@/contexts/PipelineContext";
import { Loader2 } from "lucide-react";

export const LLMContextPopover: React.FC = () => {
  const { serializeState, highLevelGoal } = usePipelineContext();
  const [contextData, setContextData] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const loadTimeoutRef = useRef<NodeJS.Timeout>();

  const loadContext = async () => {
    setIsLoading(true);
    try {
      const data = await serializeState();
      setContextData(data);
    } catch (error) {
      console.error("Failed to load context:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Update context when high-level goal changes and popover is open
  useEffect(() => {
    if (isOpen) {
      // Clear any pending timeout
      if (loadTimeoutRef.current) {
        clearTimeout(loadTimeoutRef.current);
      }
      // Set a new timeout to load context
      loadTimeoutRef.current = setTimeout(() => {
        loadContext();
      }, 500);
    }
    return () => {
      if (loadTimeoutRef.current) {
        clearTimeout(loadTimeoutRef.current);
      }
    };
  }, [highLevelGoal, isOpen]);

  const handlePopoverOpen = async (open: boolean) => {
    setIsOpen(open);
    if (open && !contextData) {
      await loadContext();
    }
  };

  return (
    <Popover onOpenChange={handlePopoverOpen}>
      <PopoverTrigger asChild>
        <button className="text-xs text-blue-500 hover:underline">
          Show LLM Context
        </button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[600px] bg-background border shadow-none"
        align="end"
        side="left"
      >
        <ScrollArea className="h-[400px] w-full rounded-md p-1">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="h-4 w-4 animate-spin" />
            </div>
          ) : (
            <pre className="text-sm whitespace-pre-wrap">{contextData}</pre>
          )}
        </ScrollArea>
      </PopoverContent>
    </Popover>
  );
};
