import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { PopoverContent } from "@/components/ui/popover";

interface AIEditPopoverProps {
  onSubmit: (instruction: string) => void;
}

export const AIEditPopover: React.FC<AIEditPopoverProps> = React.memo(
  ({ onSubmit }) => {
    const [instruction, setInstruction] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
      e.preventDefault();
      if (!instruction.trim()) return;

      setIsLoading(true);
      try {
        await onSubmit(instruction);
        setInstruction("");
      } finally {
        setIsLoading(false);
      }
    };

    return (
      <PopoverContent>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-2">
            <p className="text-sm text-muted-foreground">
              Describe how you want to modify this operation.
            </p>
            <div className="grid gap-2">
              <Textarea
                placeholder="e.g. Make the prompt more concise"
                value={instruction}
                onChange={(e) => setInstruction(e.target.value)}
                disabled={isLoading}
              />
              <Button type="submit" disabled={!instruction.trim() || isLoading}>
                {isLoading ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white"></div>
                ) : (
                  "Apply"
                )}
              </Button>
            </div>
          </div>
        </form>
      </PopoverContent>
    );
  }
);

AIEditPopover.displayName = "AIEditPopover";
