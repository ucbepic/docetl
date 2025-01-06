"use client";

import React, {
  useRef,
  useState,
  useEffect,
  useMemo,
  useCallback,
} from "react";
import { ResizableBox } from "react-resizable";
import { RefreshCw, X, Copy, AlertCircle } from "lucide-react";
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
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import { debounce } from "lodash";
import { toast } from "@/hooks/use-toast";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface AIChatPanelProps {
  onClose: () => void;
}

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  id: string;
}

const DEFAULT_SUGGESTIONS = [
  "Go over current outputs",
  "Help me refine my current operation prompt",
  "Am I doing this right?",
  "Help me with jinja2 templating",
];

const AIChatPanel: React.FC<AIChatPanelProps> = ({ onClose }) => {
  const {
    serializeState,
    highLevelGoal,
    setHighLevelGoal,
    apiKeys,
    namespace,
  } = usePipelineContext();
  const [position, setPosition] = useState({
    x: window.innerWidth - 600,
    y: 80,
  });
  const isDragging = useRef(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [usePersonalOpenAI, setUsePersonalOpenAI] = useState(false);

  const openAiKey = useMemo(() => {
    const key = apiKeys.find((key) => key.name === "OPENAI_API_KEY")?.value;
    console.log("Chat Panel: OpenAI key present:", !!key);
    return key;
  }, [apiKeys]);

  const chatHeaders = useMemo(() => {
    const headers: Record<string, string> = {};
    if (usePersonalOpenAI) {
      headers["x-use-openai"] = "true";
      if (openAiKey) {
        headers["x-openai-key"] = openAiKey;
      }
    }
    headers["x-namespace"] = namespace;
    return headers;
  }, [openAiKey, usePersonalOpenAI, namespace]);

  const {
    messages,
    setMessages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    error: chatError,
  } = useChat({
    api: "/api/chat",
    initialMessages: [],
    id: "persistent-chat",
    headers: {
      ...chatHeaders,
      "x-source": "ai_chat",
    },
    onError: (error) => {
      console.error("Chat error:", error);
      setError(error.message);
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const [localGoal, setLocalGoal] = useState(highLevelGoal);

  const hasOpenAIKey = useMemo(() => {
    if (!usePersonalOpenAI) return true;
    return apiKeys.some((key) => key.name === "OPENAI_API_KEY");
  }, [apiKeys, usePersonalOpenAI]);

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
          Math.min(window.innerWidth - 600, e.clientX - dragOffset.current.x)
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

    console.log("📝 Submitting message with API key present:", !!openAiKey);

    setError(null);

    if (!hasOpenAIKey && !usePersonalOpenAI) {
      toast({
        title: "OpenAI API Key Required",
        description: "Please add your OpenAI API key in Edit > Edit API Keys",
        variant: "destructive",
      });
      return;
    }

    const pipelineState = await serializeState();

    setMessages([
      {
        id: String(Date.now()),
        role: "system",
        content: `You are the DocWrangler assistant, helping users build and refine data analysis pipelines. You are an expert at data analysis.

Core Capabilities:
- DocWrangler is an interface that enables users to create sophisticated data processing workflows with LLM calls, like crowdsourcing pipelines. It uses the DocETL DSL and query engine.
- Each pipeline processes documents through a sequence of operations
- DocETL Operations can be LLM-based (map, reduce, resolve, filter) or utility-based (unnest, split, gather, sample) or code-based (python for map, reduce, and filter)

Operation Details:
- Every LLM operation has:
  - A prompt (jinja2 template)
  - An output schema (JSON schema)
- Operation-specific templating:
  - Map/Filter: Access current doc with '{{ input.keyname }}'
  - Reduce: Loop through docs with '{% for doc in inputs %}...{% endfor %}'
  - Resolve: Compare docs with '{{ input1 }}/{{ input2 }}' and canonicalize with '{{ inputs }}'
- Code-based operations:
  - Map: Define a transform function (def transform(doc: dict) -> dict), where the returned dict will have key-value pairs that will be added to the output document
  - Filter: Define a transform function (def transform(doc: dict) -> bool), where the function should return true if the document should be included in the output
  - Reduce: Define a transform function (def transform(docs: list[dict]) -> dict), where the returned dict will have key-value pairs that will *be* the output document (unless "pass_through" is set to true, then the first original doc for every group will also be returned)
  - Only do imports of common libraries, inside the function definition
  - Only suggest code-based operations if the task is one that is easily expressed in code, and LLMs or crowd workers are incapable of doing it correctly (e.g., word count, simple regex, etc.)

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

When Reviewing Outputs:
- All the output fields have been converted to strings, even if they were originally numbers, arrays, or other types. So NEVER COMMENT ON TYPES.
- Actively analyze outputs for discrepancies in structure across the outputs, edge cases, and quality issues.
- For discrepancies, describe how to standardize them.
- Identify where outputs may not fully satisfy the intended goals
- Never simply restate or summarize outputs - provide critical analysis
- Provide 1 suggestion at a time

Remember, you are only helping the user discover their analysis goal, and only suggest improvements that LLMs or crowd workers are capable of.

Here's their current pipeline state:
${pipelineState}

Remember, all the output fields have been converted to strings, even if they were originally numbers, arrays, or other types. So NEVER COMMENT ON TYPES. Steer the user towards their high-level goal, if specified.`,
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

  const debouncedSetHighLevelGoal = useMemo(
    () => debounce((value: string) => setHighLevelGoal(value), 1000),
    [setHighLevelGoal]
  );

  useEffect(() => {
    return () => {
      debouncedSetHighLevelGoal.cancel();
    };
  }, [debouncedSetHighLevelGoal]);

  const handleGoalUpdate = useCallback(
    (newGoal: string) => {
      setLocalGoal(newGoal);
      debouncedSetHighLevelGoal(newGoal);
    },
    [debouncedSetHighLevelGoal]
  );

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
        width={600}
        height={500}
        minConstraints={[400, 400]}
        maxConstraints={[1000, 800]}
        resizeHandles={["sw", "se"]}
        className="bg-white rounded-lg shadow-lg border overflow-hidden text-s"
      >
        <div
          className="h-8 bg-gray-100 drag-handle flex justify-between items-center px-3 cursor-move select-none"
          onMouseDown={handleMouseDown}
        >
          <span className="text-s text-primary font-medium flex items-center gap-3">
            <span className="pointer-events-none">
              <Scroll size={14} />
            </span>
            <LLMContextPopover />
          </span>
          <div className="flex items-center gap-2 pointer-events-auto">
            <div className="flex items-center gap-1.5 border-r pr-2">
              <Switch
                id="use-personal-openai"
                checked={usePersonalOpenAI}
                onCheckedChange={setUsePersonalOpenAI}
                className="h-4 w-7 data-[state=checked]:bg-primary"
              />
              <Label
                htmlFor="use-personal-openai"
                className="text-[10px] text-muted-foreground whitespace-nowrap"
              >
                Personal OpenAI Key
              </Label>
            </div>
            <Popover>
              <PopoverTrigger asChild>
                <span className="text-s text-primary font-medium flex items-center gap-2 cursor-pointer">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-4 px-2 text-xs"
                  >
                    {highLevelGoal ? "Edit Analysis Goal" : "Set Analysis Goal"}
                  </Button>
                </span>
              </PopoverTrigger>
              <PopoverContent className="w-80 z-[10000]" side="top" align="end">
                <div className="space-y-2">
                  <h4 className="font-medium text-sm">Pipeline Goal</h4>
                  <Textarea
                    placeholder="Describe the high-level goal of your pipeline..."
                    className="min-h-[100px]"
                    value={localGoal}
                    onChange={(e) => handleGoalUpdate(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    This helps the assistant provide more relevant suggestions.
                  </p>
                </div>
              </PopoverContent>
            </Popover>
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
            {error && (
              <div className="bg-destructive/10 text-destructive rounded-md p-3 mb-2 text-xs">
                <div className="flex gap-2">
                  <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium">Error</p>
                    <p className="mt-1">{error}</p>
                  </div>
                </div>
              </div>
            )}
            {messages.filter((message) => message.role !== "system").length ===
            0 ? (
              <div className="flex flex-col gap-2">
                {usePersonalOpenAI && !hasOpenAIKey && (
                  <div className="bg-destructive/10 text-destructive rounded-md p-3 mb-2 text-xs">
                    <div className="flex gap-2">
                      <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="font-medium">OpenAI API Key Required</p>
                        <p className="mt-1">
                          To use your personal OpenAI account, please add your
                          OpenAI API key in Edit {">"} Edit API Keys.
                        </p>
                      </div>
                    </div>
                  </div>
                )}
                {DEFAULT_SUGGESTIONS.map((suggestion, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    className="text-xs justify-start h-auto py-2 px-3"
                    onClick={() => {
                      handleInputChange({
                        target: { value: suggestion },
                      } as React.ChangeEvent<HTMLInputElement>);
                      handleMessageSubmit({
                        preventDefault: () => {},
                      } as React.FormEvent);
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
              placeholder={
                error
                  ? "Try again..."
                  : !hasOpenAIKey && !usePersonalOpenAI
                  ? "Add OpenAI API key to continue..."
                  : "Ask a question..."
              }
              className="flex-1 text-s h-7"
              disabled={!hasOpenAIKey && !usePersonalOpenAI}
            />
            <Button
              type="submit"
              size="sm"
              disabled={isLoading || (!hasOpenAIKey && !usePersonalOpenAI)}
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
