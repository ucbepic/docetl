"use client";

import React, { useRef, useState, useEffect } from "react";
import { ResizableBox } from "react-resizable";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChat } from "ai/react";
import { cn } from "@/lib/utils";
import { Loader2, Scroll } from "lucide-react";
import "react-resizable/css/styles.css";

interface AIChatPanelProps {
  onClose: () => void;
}

interface Message {
  role: "assistant" | "user";
  content: string;
}

const AIChatPanel: React.FC<AIChatPanelProps> = ({ onClose }) => {
  const [position, setPosition] = useState({
    x: window.innerWidth - 400,
    y: 80,
  });
  const isDragging = useRef(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const { messages, input, handleInputChange, handleSubmit, isLoading } =
    useChat({
      keepLastMessageOnError: true,
    });

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).classList.contains("drag-handle")) {
      isDragging.current = true;
      dragOffset.current = {
        x: e.clientX - position.x,
        y: e.clientY - position.y,
      };
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging.current) {
      setPosition({
        x: Math.max(
          0,
          Math.min(window.innerWidth - 400, e.clientX - dragOffset.current.x)
        ),
        y: Math.max(
          0,
          Math.min(window.innerHeight - 600, e.clientY - dragOffset.current.y)
        ),
      });
    }
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  useEffect(() => {
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div
      style={{
        position: "fixed",
        top: position.y,
        left: position.x,
        zIndex: 9999,
      }}
    >
      <ResizableBox
        width={400}
        height={500}
        minConstraints={[300, 400]}
        maxConstraints={[800, 800]}
        resizeHandles={["sw", "se"]}
        className="bg-white rounded-lg shadow-lg border overflow-hidden"
      >
        <div
          className="h-6 bg-gray-100 drag-handle flex justify-between items-center px-2 cursor-move"
          onMouseDown={handleMouseDown}
        >
          <span className="text-sm text-primary font-medium">
            <Scroll size={14} />
          </span>
          <Button
            variant="ghost"
            size="sm"
            className="h-5 w-5 p-0"
            onClick={onClose}
          >
            <X size={14} />
          </Button>
        </div>
        <div className="flex flex-col h-[calc(100%-24px)]">
          <ScrollArea ref={scrollAreaRef} className="flex-1 p-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={cn(
                  "mb-2 flex",
                  message.role === "assistant" ? "justify-start" : "justify-end"
                )}
              >
                <div
                  className={cn(
                    "rounded-lg px-3 py-2 max-w-[80%]",
                    message.role === "assistant"
                      ? "bg-gray-100 text-gray-900"
                      : "bg-primary text-white"
                  )}
                >
                  {message.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-center">
                <Loader2 className="h-4 w-4 animate-spin" />
              </div>
            )}
          </ScrollArea>
          <form
            onSubmit={handleSubmit}
            className="border-t p-4 flex gap-2 items-center"
          >
            <Input
              value={input}
              onChange={handleInputChange}
              placeholder="Ask a question..."
              className="flex-1"
            />
            <Button type="submit" size="sm" disabled={isLoading}>
              Send
            </Button>
          </form>
        </div>
      </ResizableBox>
    </div>
  );
};

export default AIChatPanel;
