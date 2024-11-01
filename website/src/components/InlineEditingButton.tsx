"use client";

import React from "react";
import { Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";

interface InlineEditingButtonProps {
  selectedText: string;
  onSubmit: (instruction: string) => void;
}

const InlineEditingButton: React.FC<InlineEditingButtonProps> = ({
  selectedText,
  onSubmit,
}) => {
  const [instruction, setInstruction] = React.useState("");
  const [isOpen, setIsOpen] = React.useState(false);

  const handleSubmit = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (instruction.trim()) {
      onSubmit(instruction);
      setInstruction("");
      setIsOpen(false);
    }
  };

  const handleButtonClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsOpen(true);
  };

  return (
    <Popover modal={true} open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild onClick={handleButtonClick}>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 absolute bg-background shadow-sm border"
        >
          <Wand2 className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80" sideOffset={5}>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSubmit(e as any);
          }}
        >
          <div className="space-y-4">
            <div className="text-sm text-muted-foreground">
              Selected text: {selectedText}
            </div>
            <Textarea
              placeholder="Enter your instruction for editing..."
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
            />
            <Button type="submit" className="w-full">
              Submit
            </Button>
          </div>
        </form>
      </PopoverContent>
    </Popover>
  );
};
export default InlineEditingButton;
