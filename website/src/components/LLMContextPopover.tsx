"use client";

import React, { useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { usePipelineContext } from "@/contexts/PipelineContext";
import { Loader2 } from "lucide-react";

export const LLMContextPopover: React.FC = () => {
  const { serializeState } = usePipelineContext();
  const [contextData, setContextData] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const handlePopoverOpen = async (open: boolean) => {
    if (open && !contextData) {
      setIsLoading(true);
      try {
        const data = await serializeState();
        setContextData(data);
      } catch (error) {
        console.error("Failed to load context:", error);
      } finally {
        setIsLoading(false);
      }
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
