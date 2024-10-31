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
import { LLMContextPopover } from "@/components/LLMContextPopover";
import { usePipelineContext } from "@/contexts/PipelineContext";
import ReactMarkdown from "react-markdown";

interface AIChatPanelProps {
  onClose: () => void;
}

const AIChatPanel: React.FC<AIChatPanelProps> = ({ onClose }) => {
  const [position, setPosition] = useState({
    x: window.innerWidth - 400,
    y: 80,
  });
  const isDragging = useRef(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const {
    messages,
    setMessages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
  } = useChat({
    api: "/api/chat",
    initialMessages: [],
  });
  const { serializeState } = usePipelineContext();

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

  const handleMessageSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const pipelineState = await serializeState();

    setMessages([
      {
        id: String(Date.now()),
        role: "system",
        content: `You are the DocETL assistant, helping users build and refine data analysis pipelines. DocETL enables users to create sophisticated data processing workflows that combine the power of LLMs with traditional data operations.

Each pipeline processes documents (or key-value pairs) through a sequence of operations. LLM operations like 'map' (process individual documents), 'reduce' (analyze multiple documents together), 'resolve' (entity resolution), and 'filter' (conditionally retain documents) can be combined with utility operations like 'unnest', 'split', 'gather', and 'sample' to create powerful analysis flows.
Every LLM operation has a prompt and an output schema, which determine the keys to be added to the documents as the operation runs. Prompts are jinja2 templates, and output schemas are JSON schemas. For 'map' and 'filter' operations, you can reference the current document as 'input' and access its keys as '{{ input.keyname }}'. For 'reduce' operations, you can reference the group of input documents as 'inputs' and loop over them with '{% for doc in inputs %}...{% endfor %}'. For 'resolve' operations, there are two prompts: the comparison_prompt compares two documents (referenced as '{{ input1 }}' and '{{ input2 }}') to determine if they match, while the resolution_prompt takes a group of matching documents (referenced as 'inputs') and canonicalizes them into a single output following the operation's schema.
You should help users optimize their pipelines and overcome any challenges they encounter. Ask questions to better understand their goals, suggest improvements, help debug issues, or explore new approaches to achieve their analysis objectives. Only ask one question at a time, and keep your responses concise.
Also, don't give lots of suggestions. Only give one or two at a time. For each suggestion, be very detailed--if you say "use the 'map' operation", then you should also say what the prompt and output schema should be.

You don't have the ability to write to the pipeline state. You can only give suggestions about what the user should do next.
Before jumping to a new operation, verify that the user is satisfied with the current operation's outputs. Ask them specific questions about the outputs related to the task, to get them to iterate on the current operation if needed. For example, if all the outputs look the same, maybe they want to be more specific in their operation prompt. The users will typically be biased towards accepting the current operation's outputs, so be very specific in your questions to get them to iterate on the current operation. You may have to propose iterations yourself. Always look for inadequacies in the outputs, and surface them to the user to see what they think.
Remember, always be specific! Never be vague or general. Never suggest new operations unless the user is completely satisfied with the current operation's outputs.
Your answers will be in markdown format. Use bold, italics, and lists to draw attention to important points.

Here's their current pipeline state:
${pipelineState}`,
      },
      ...messages.filter((m) => m.role !== "system"),
    ]);

    handleSubmit(e);
  };

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
        className="bg-white rounded-lg shadow-lg border overflow-hidden text-s"
      >
        <div
          className="h-6 bg-gray-100 drag-handle flex justify-between items-center px-2 cursor-move"
          onMouseDown={handleMouseDown}
        >
          <span className="text-s text-primary font-medium flex items-center gap-2">
            <Scroll size={12} />
            <LLMContextPopover />
          </span>
          <Button
            variant="ghost"
            size="sm"
            className="h-4 w-4 p-0"
            onClick={onClose}
          >
            <X size={12} />
          </Button>
        </div>
        <div className="flex flex-col h-[calc(100%-24px)]">
          <ScrollArea ref={scrollAreaRef} className="flex-1 p-4">
            {messages
              .filter((message) => message.role !== "system")
              .map((message, index) => (
                <div
                  key={index}
                  className={cn(
                    "mb-2 flex",
                    message.role === "assistant"
                      ? "justify-start"
                      : "justify-end"
                  )}
                >
                  <div
                    className={cn(
                      "rounded-lg px-3 py-2 max-w-[75%] break-words prose prose-xs [&_pre]:whitespace-pre-wrap [&_code]:whitespace-pre-wrap",
                      message.role === "assistant"
                        ? "bg-gray-100 text-gray-900"
                        : "bg-primary text-white prose-invert"
                    )}
                    style={{
                      overflowWrap: "break-word",
                      wordWrap: "break-word",
                      hyphens: "auto",
                    }}
                  >
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                  </div>
                </div>
              ))}
            {isLoading && (
              <div className="flex justify-center">
                <Loader2 className="h-3 w-3 animate-spin" />
              </div>
            )}
          </ScrollArea>
          <form
            onSubmit={handleMessageSubmit}
            className="border-t p-3 flex gap-2 items-center"
          >
            <Input
              value={input}
              onChange={handleInputChange}
              placeholder="Ask a question..."
              className="flex-1 text-s h-7"
            />
            <Button
              type="submit"
              size="sm"
              disabled={isLoading}
              className="h-7 text-s text-white"
            >
              Send
            </Button>
          </form>
        </div>
      </ResizableBox>
    </div>
  );
};

export default AIChatPanel;
