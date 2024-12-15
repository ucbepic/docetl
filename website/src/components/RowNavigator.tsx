import React, { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface RowNavigatorProps {
  currentRow: number;
  totalRows: number;
  onNavigate: (direction: "prev" | "next") => void;
  onJumpToRow: (index: number) => void;
  disabled?: boolean;
  label?: string;
  compact?: boolean;
}

export const RowNavigator = React.memo(
  ({
    currentRow,
    totalRows,
    onNavigate,
    onJumpToRow,
    disabled = false,
    label = "Row",
    compact = false,
  }: RowNavigatorProps) => {
    const [isEditing, setIsEditing] = useState(false);
    const [inputValue, setInputValue] = useState("");
    const inputRef = useRef<HTMLInputElement>(null);

    const handleSubmit = () => {
      const rowNum = parseInt(inputValue);
      if (!isNaN(rowNum) && rowNum >= 1 && rowNum <= totalRows) {
        onJumpToRow(rowNum - 1);
        setIsEditing(false);
      }
      setInputValue("");
    };

    return (
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onNavigate("prev")}
          disabled={currentRow === 0 || disabled}
          className={`${compact ? "h-6 w-6" : "h-8 w-8"} p-0`}
        >
          <ChevronLeft className={`${compact ? "h-3 w-3" : "h-4 w-4"}`} />
        </Button>

        <div className="flex items-center gap-1.5">
          <span
            className={`${
              compact ? "text-xs" : "text-sm"
            } text-muted-foreground`}
          >
            {label}
          </span>
          {isEditing ? (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSubmit();
              }}
              className="relative"
            >
              <Input
                ref={inputRef}
                type="number"
                min={1}
                max={totalRows}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onBlur={() => {
                  if (inputValue) handleSubmit();
                  else setIsEditing(false);
                }}
                className={`${
                  compact ? "w-16 h-6 text-xs" : "w-20 h-8 text-sm"
                } pr-8`}
                autoFocus
              />
              <span className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
                ↵
              </span>
            </form>
          ) : (
            <Button
              variant="ghost"
              size="sm"
              className={`${
                compact
                  ? "h-6 min-w-[3rem] px-1.5 text-xs"
                  : "h-8 min-w-[4rem] px-2 text-sm"
              } font-mono`}
              onClick={() => {
                setIsEditing(true);
                setTimeout(() => inputRef.current?.focus(), 0);
              }}
              disabled={disabled}
            >
              {currentRow + 1}
            </Button>
          )}
          <span
            className={`${
              compact ? "text-xs" : "text-sm"
            } text-muted-foreground`}
          >
            of {totalRows}
          </span>
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={() => onNavigate("next")}
          disabled={currentRow === totalRows - 1 || disabled}
          className={`${compact ? "h-6 w-6" : "h-8 w-8"} p-0`}
        >
          <ChevronRight className={`${compact ? "h-3 w-3" : "h-4 w-4"}`} />
        </Button>
      </div>
    );
  }
);

RowNavigator.displayName = "RowNavigator";
