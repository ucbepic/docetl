"use client";

import React, { useRef, useState, useEffect } from "react";
import { ResizableBox } from "react-resizable";
import { Eraser, RefreshCw, X, Copy } from "lucide-react";
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

const DEFAULT_SUGGESTIONS = [
  "Go over current outputs",
  "Help me refine my current operation prompt",
  "Am I doing this right?",
  "Help me with jinja2 templating",
];

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
    id: "persistent-chat",
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
        content: `You are the DocETL assistant, helping users build and refine data analysis pipelines. You are an expert at data analysis.

Core Capabilities:
- DocETL enables users to create sophisticated data processing workflows combining LLMs with traditional data operations
- Each pipeline processes documents through a sequence of operations
- Operations can be LLM-based (map, reduce, resolve, filter) or utility-based (unnest, split, gather, sample)

Operation Details:
- Every LLM operation has:
  - A prompt (jinja2 template)
  - An output schema (JSON schema)
- Operation-specific templating:
  - Map/Filter: Access current doc with '{{ input.keyname }}'
  - Reduce: Loop through docs with '{% for doc in inputs %}...{% endfor %}'
  - Resolve: Compare docs with '{{ input1 }}/{{ input2 }}' and canonicalize with 'inputs'

Your Role:
- Help users optimize pipelines and overcome challenges
- Ask focused questions to understand goals (one at a time)
- Keep responses concise
- Provide 1-2 detailed suggestions at a time
- Cannot directly modify pipeline state - only provide guidance

Best Practices:
- Verify satisfaction with current operation before suggesting new ones
- Ask specific questions about outputs to encourage iteration
- Look for and surface potential inadequacies in outputs
- Use markdown formatting (bold, italics, lists) for clarity. Only action items or suggestions should be bolded.
- Be specific, never vague or general
- Be concise, don't repeat yourself

Here's their current pipeline state:
${pipelineState}`,
      },
      ...messages.filter((m) => m.role !== "system"),
    ]);

    handleSubmit(e);
  };

  const handleClearMessages = () => {
    setMessages([]);
  };

  const CodeBlock = ({ children }: { children: string }) => {
    const handleCopy = () => {
      navigator.clipboard.writeText(children);
    };

    return (
      <div className="relative group">
        <Button
          variant="ghost"
          className="absolute right-2 top-2 h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
          onClick={handleCopy}
        >
          <Copy size={16} />
        </Button>
        <pre>
          <code>{children}</code>
        </pre>
      </div>
    );
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
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="h-4 w-4 p-0"
              onClick={handleClearMessages}
              title="Clear messages"
            >
              <RefreshCw size={12} />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-4 w-4 p-0"
              onClick={onClose}
            >
              <X size={12} />
            </Button>
          </div>
        </div>
        <div className="flex flex-col h-[calc(100%-24px)]">
          <ScrollArea ref={scrollAreaRef} className="flex-1 p-4">
            {messages.filter((message) => message.role !== "system").length ===
            0 ? (
              <div className="flex flex-col gap-2">
                {DEFAULT_SUGGESTIONS.map((suggestion, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    className="text-xs justify-start h-auto py-2 px-3"
                    onClick={() => {
                      handleInputChange({
                        target: { value: suggestion },
                      } as any);
                      handleMessageSubmit({ preventDefault: () => {} } as any);
                    }}
                  >
                    {suggestion}
                  </Button>
                ))}
              </div>
            ) : (
              messages
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
                      <ReactMarkdown
                        components={{
                          code: ({
                            inline,
                            children,
                          }: {
                            inline?: boolean;
                            children: React.ReactNode;
                          }) => {
                            if (inline) {
                              return <code>{children}</code>;
                            }
                            return <CodeBlock>{String(children)}</CodeBlock>;
                          },
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                ))
            )}
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
