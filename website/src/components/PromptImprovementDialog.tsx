import React, { useState, useCallback, useEffect, useRef } from "react";
import { useChat } from "ai/react";
import { Operation } from "@/app/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, ArrowLeft } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { usePipelineContext } from "@/contexts/PipelineContext";
import { useBookmarkContext } from "@/contexts/BookmarkContext";
import { diffLines } from "diff";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

type Step = "select" | "analyze";

interface PromptImprovementDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  currentOperation: Operation;
  onSave: (
    newPrompt:
      | string
      | { comparison_prompt: string; resolution_prompt: string },
    schemaChanges?: Array<[string, string]>
  ) => void;
}

interface Revision {
  messages: {
    id: string;
    role: "system" | "user" | "assistant";
    content: string;
  }[];
  prompt: string | null;
  timestamp: number;
}

interface FeedbackConnection {
  fromRevision: number;
  toRevision: number;
  feedback: string;
}

function extractTagContent(text: string, tag: string): string | null {
  const regex = new RegExp(`<${tag}>(.*?)</${tag}>`, "s");
  const match = text.match(regex);
  return match ? match[1].trim() : null;
}

type PromptType = "comparison" | "resolve";

function DiffView({
  oldText,
  newText,
  type,
}: {
  oldText: string;
  newText: string;
  type?: PromptType;
}) {
  const diff = diffLines(oldText, newText);

  return (
    <div>
      {type && (
        <div className="text-sm font-medium mb-2">
          {type === "comparison" ? "Comparison Prompt:" : "Resolution Prompt:"}
        </div>
      )}
      <div className="font-mono text-sm whitespace-pre-wrap">
        {diff.map((part, index) => (
          <span
            key={index}
            className={
              part.added
                ? "bg-green-100 dark:bg-green-900/30"
                : part.removed
                ? "bg-red-100 dark:bg-red-900/30"
                : ""
            }
          >
            {part.value}
          </span>
        ))}
      </div>
    </div>
  );
}

function removePromptAndSchemaTags(text: string): string {
  return text
    .replace(/<prompt>[\s\S]*?<\/prompt>/g, "")
    .replace(/<schema>[\s\S]*?<\/schema>/g, "")
    .replace(/<comparison_prompt>[\s\S]*?<\/comparison_prompt>/g, "")
    .replace(/<resolution_prompt>[\s\S]*?<\/resolution_prompt>/g, "");
}

const getSystemContent = (
  pipelineState: string,
  selectedOperation: Operation
) => `You are a prompt engineering expert. Analyze the current operation's prompt${
  selectedOperation.type === "resolve" ? "s" : ""
} and suggest improvements based on the pipeline state.

Current pipeline state:
${pipelineState}

Focus on the operation named "${selectedOperation.name}" ${
  selectedOperation.type === "resolve"
    ? `with comparison prompt:
${selectedOperation.otherKwargs?.comparison_prompt || ""}

and resolution prompt:
${selectedOperation.otherKwargs?.resolution_prompt || ""}`
    : `with prompt:
${selectedOperation.prompt}`
}

${
  selectedOperation.output?.schema
    ? `
Current output schema keys:
${selectedOperation.output.schema.map((item) => `- ${item.key}`).join("\n")}
`
    : ""
}

IMPORTANT: 
1. ${
  selectedOperation.type === "resolve"
    ? "You must ALWAYS include complete revised prompts wrapped in <comparison_prompt></comparison_prompt> AND <resolution_prompt></resolution_prompt> tags in your response"
    : "You must ALWAYS include a complete revised prompt wrapped in <prompt></prompt> tags in your response"
}, even if you're just responding to feedback.

2. CRITICAL: Never modify or remove any placeholders wrapped in {{ }} brackets - these are essential template variables that must remain exactly as they are.

3. Only suggest schema key changes if absolutely necessary - when the current keys are misleading, incorrect, or ambiguous. If the schema keys are fine, don't suggest changes. Include changes in <schema> tags as a list of "oldkey,newkey" pairs, one per line. Example:
<schema>
misleading_key,accurate_key
ambiguous_name,specific_name
</schema>

When responding:
1. Briefly acknowledge/analyze any feedback (1-2 sentences)
2. ALWAYS provide ${
  selectedOperation.type === "resolve"
    ? "complete revised prompts wrapped in <comparison_prompt></comparison_prompt> AND <resolution_prompt></resolution_prompt> tags"
    : "a complete revised prompt wrapped in <prompt></prompt> tags"
}
3. The prompt${
  selectedOperation.type === "resolve" ? "s" : ""
} should include all previous improvements plus any new changes
4. Make prompts specific and concise:
   - For subjective terms like "detailed" or "comprehensive", provide examples or metrics (e.g. "include 3-5 key points per section")
   - For qualitative instructions like "long output", specify length (e.g. "200-300 words") based on my feedback or provide examples
   - When using adjectives, include a reference point (e.g. "technical like API documentation" vs "simple like a blog post")
   - NEVER modify any {{placeholder}} variables - keep them exactly as they are
5. IMPORTANT: When writing prompts that ask for specific types of analysis or output, ALWAYS include 1-2 brief examples of the expected output format. For instance:

Instead of:
"Extract the key points about the company's financial performance"

Write:
"Extract the key points about the company's financial performance. Format each point like this:
- Revenue: [Key metric] (with % change from previous period)
Example: 'Revenue: $5.2M in Q2 2023 (+15% YoY)'
- Profit margins: [Percentage] with brief context
Example: 'Profit margin: 23% (improved due to cost optimization)'"

Instead of:
"Compare the technical specifications"

Write:
"Compare the technical specifications. For each component, provide a structured comparison like this:
- Processing Power:
  * Product A: 2.4GHz quad-core
  * Product B: 3.1GHz octa-core
  * Key difference: B offers 2.5x more processing threads"

Your suggested prompt should follow this pattern of including concrete examples that demonstrate the expected format and level of detail.`;

function extractSchemaChanges(text: string): Array<[string, string]> {
  const schemaContent = extractTagContent(text, "schema");
  if (!schemaContent) return [];

  return schemaContent
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => {
      const [oldKey, newKey] = line.split(",").map((k) => k.trim());
      return [oldKey, newKey] as [string, string];
    });
}

// Move the debounce function to the top, before any component definitions
const debounce = <T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): ((...args: Parameters<T>) => void) => {
  let timeoutId: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
};

function AutosizeTextarea({
  value,
  onChange,
  onBlur,
  type,
}: {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onBlur?: () => void;
  type?: PromptType;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Debounced resize function
  const resize = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, []);

  // Debounced resize with useEffect
  useEffect(() => {
    const timeoutId = setTimeout(resize, 10);
    return () => clearTimeout(timeoutId);
  }, [value, resize]);

  return (
    <div className="relative">
      {type && (
        <div className="text-sm font-medium mb-2">
          {type === "comparison" ? "Comparison Prompt:" : "Resolution Prompt:"}
        </div>
      )}
      <Textarea
        ref={textareaRef}
        value={value}
        onChange={onChange}
        onBlur={onBlur}
        className="font-mono text-sm resize-none"
        style={{ minHeight: "100px" }}
        autoFocus={!type || type === "comparison"}
      />
    </div>
  );
}

function buildRevisionTree(
  revisions: Revision[],
  connections: FeedbackConnection[]
) {
  type TreeNode = {
    index: number;
    revision: Revision;
    children: TreeNode[];
    feedback?: string;
  };

  const nodes: TreeNode[] = revisions.map((revision, index) => ({
    index,
    revision,
    children: [],
  }));

  // Build the tree structure
  connections.forEach((conn) => {
    const parentNode = nodes[conn.fromRevision];
    const childNode = nodes[conn.toRevision];
    if (parentNode && childNode) {
      childNode.feedback = conn.feedback;
      parentNode.children.push(childNode);
    }
  });

  // Return only root nodes (nodes without parents)
  return nodes.filter(
    (node) => !connections.some((conn) => conn.toRevision === node.index)
  );
}

function RevisionTreeNode({
  node,
  depth = 0,
  selectedIndex,
  onSelect,
}: {
  node: ReturnType<typeof buildRevisionTree>[0];
  depth?: number;
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}) {
  return (
    <div className="flex flex-col">
      <div
        className="flex items-center relative"
        style={{ paddingLeft: `${depth * 20}px` }}
      >
        {depth > 0 && (
          <div
            className="absolute left-0 top-1/2 w-4 h-px border-t-2 border-muted-foreground/20"
            style={{ left: `${(depth - 1) * 20 + 10}px` }}
          />
        )}
        <button
          onClick={() => onSelect(node.index)}
          className={`flex items-center gap-2 w-full text-left py-1.5 px-2 text-sm hover:bg-muted rounded-sm transition-colors ${
            selectedIndex === node.index ? "bg-muted" : ""
          }`}
        >
          <div className="w-1.5 h-1.5 rounded-full bg-primary shrink-0" />
          <div className="min-w-0">
            <div className="font-medium truncate">
              {node.index === 0
                ? "Initial version"
                : node.feedback || `Revision ${node.index}`}
            </div>
            <div className="text-xs text-muted-foreground">
              {new Date(node.revision.timestamp).toLocaleTimeString()}
            </div>
          </div>
        </button>
      </div>
      {node.children.length > 0 && (
        <div className="relative">
          <div
            className="absolute left-0 top-0 bottom-4 border-l-2 border-muted-foreground/20"
            style={{ left: `${depth * 20 + 10}px` }}
          />
          <div className="flex flex-col">
            {node.children.map((child, i) => (
              <RevisionTreeNode
                key={i}
                node={child}
                depth={depth + 1}
                selectedIndex={selectedIndex}
                onSelect={onSelect}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Add new component for schema changes diff view
function SchemaChangesDiff({ changes }: { changes: Array<[string, string]> }) {
  if (changes.length === 0) return null;

  return (
    <div className="border rounded-md p-4">
      <h3 className="font-medium mb-2">Schema Key Changes:</h3>
      <div className="font-mono text-sm space-y-1">
        {changes.map(([oldKey, newKey], index) => (
          <div key={index}>
            <span className="line-through text-red-500">{oldKey}</span>
            <span className="mx-2">→</span>
            <span className="text-green-500">{newKey}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Add helper function to extract both prompts for resolve operations
function extractPrompts(text: string): {
  comparisonPrompt?: string;
  resolvePrompt?: string;
  prompt?: string;
} {
  const comparisonPrompt = extractTagContent(text, "comparison_prompt");
  const resolvePrompt = extractTagContent(text, "resolution_prompt");
  const prompt = extractTagContent(text, "prompt");

  return {
    comparisonPrompt,
    resolvePrompt,
    prompt,
  };
}

// Update the state to use a single editedPrompt
type EditedPrompt =
  | string
  | { comparison_prompt: string; resolution_prompt: string };

export function PromptImprovementDialog({
  open,
  onOpenChange,
  currentOperation,
  onSave,
}: PromptImprovementDialogProps) {
  const { operations, serializeState } = usePipelineContext();
  const { bookmarks, removeBookmark } = useBookmarkContext();
  const [step, setStep] = useState<Step>("select");
  const [selectedOperationId, setSelectedOperationId] = useState<string>(
    currentOperation.id
  );
  const [editedPrompt, setEditedPrompt] = useState<EditedPrompt | null>(null);
  const [feedbackText, setFeedbackText] = useState("");
  const [popoverOpen, setPopoverOpen] = useState(false);
  const [isDirectEditing, setIsDirectEditing] = useState(false);
  const [revisions, setRevisions] = useState<Revision[]>([]);
  const [connections, setConnections] = useState<FeedbackConnection[]>([]);
  const [selectedRevisionIndex, setSelectedRevisionIndex] = useState<
    number | null
  >(null);
  const [expectingNewRevision, setExpectingNewRevision] = useState(false);
  const [chatKey, setChatKey] = useState(0);
  const [localFeedbackText, setLocalFeedbackText] = useState("");
  const [showSaveConfirm, setShowSaveConfirm] = useState(false);

  const selectedOperation = operations.find(
    (op) => op.id === selectedOperationId
  );

  const relevantBookmarks = selectedOperation
    ? bookmarks.flatMap((bookmark) =>
        bookmark.notes.filter(
          (note) => note.metadata?.operationName === selectedOperation.name
        )
      )
    : [];

  const { messages, isLoading, append, setMessages } = useChat({
    api: "/api/chat",
    id: `prompt-improvement-${chatKey}`,
    onFinish: () => {
      // Optional: handle completion
    },
  });

  // Update the effect that handles new messages
  useEffect(() => {
    if (!isLoading && messages.length > 0 && expectingNewRevision) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.role === "assistant") {
        if (selectedOperation?.type === "resolve") {
          const extractedPrompts = extractPrompts(lastMessage.content);
          if (
            extractedPrompts.comparisonPrompt &&
            extractedPrompts.resolvePrompt
          ) {
            // Skip if the prompts are the same as what we have in editedPrompt
            if (
              typeof editedPrompt === "object" &&
              editedPrompt?.comparison_prompt ===
                extractedPrompts.comparisonPrompt &&
              editedPrompt?.resolution_prompt === extractedPrompts.resolvePrompt
            ) {
              return;
            }

            const newPrompt = {
              comparison_prompt: extractedPrompts.comparisonPrompt,
              resolution_prompt: extractedPrompts.resolvePrompt,
            };
            setEditedPrompt(newPrompt);

            // Create new revision
            const newRevision = {
              messages: JSON.parse(JSON.stringify(messages)),
              prompt: newPrompt,
              timestamp: Date.now(),
            };

            // Add revision and update selected index
            setRevisions((prev) => {
              const newRevisions = [...prev, newRevision] as Revision[];
              setSelectedRevisionIndex(newRevisions.length - 1);
              return newRevisions;
            });
          }
        } else {
          const extractedPrompt = extractTagContent(
            lastMessage.content,
            "prompt"
          );
          if (extractedPrompt) {
            // Skip if the prompt is the same as what we have in editedPrompt
            if (
              typeof editedPrompt === "string" &&
              editedPrompt === extractedPrompt
            ) {
              return;
            }

            setEditedPrompt(extractedPrompt);

            // Create new revision
            const newRevision = {
              messages: JSON.parse(JSON.stringify(messages)),
              prompt: extractedPrompt,
              timestamp: Date.now(),
            };

            // Add revision and update selected index
            setRevisions((prev) => {
              const newRevisions = [...prev, newRevision];
              setSelectedRevisionIndex(newRevisions.length - 1);
              return newRevisions;
            });
          }
        }

        // If this isn't the first revision, create a connection
        if (revisions.length > 0) {
          const parentIndex = selectedRevisionIndex ?? revisions.length - 1;
          setConnections((prev) => [
            ...prev,
            {
              fromRevision: parentIndex,
              toRevision: revisions.length,
              feedback: feedbackText,
            },
          ]);
        }

        setExpectingNewRevision(false);
      }
    }
  }, [messages, isLoading, expectingNewRevision, selectedOperation?.type]);

  useEffect(() => {
    if (selectedRevisionIndex !== null && revisions[selectedRevisionIndex]) {
      const revision = revisions[selectedRevisionIndex];
      // Set messages to a deep copy of the revision's messages
      setMessages(JSON.parse(JSON.stringify(revision.messages)));

      // Update the prompt
      if (revision.prompt) {
        setEditedPrompt(revision.prompt);
      }
    }
  }, [selectedRevisionIndex, revisions]);

  const handleImprove = useCallback(async () => {
    setExpectingNewRevision(true); // Set flag before getting first response
    setEditedPrompt(null);
    setStep("analyze");
    setRevisions([]);
    setConnections([]);
    setSelectedRevisionIndex(null);

    const selectedOperation = operations.find(
      (op) => op.id === selectedOperationId
    );
    if (!selectedOperation) return;

    const bookmarksSection =
      relevantBookmarks.length > 0
        ? `\nMake sure to reflect my Feedback for this operation:\n${relevantBookmarks
            .map((note) => `- ${note.note}`)
            .join("\n")}`
        : "\nNo feedback found for this operation.";

    const pipelineState = await serializeState();
    const systemContent = getSystemContent(pipelineState, selectedOperation);

    // First set the system message
    setMessages([
      {
        role: "system",
        content: systemContent,
        id: "system-1",
      },
    ]);

    // Then append the user message with the appropriate prompt(s)
    await append({
      role: "user",
      content: `Please analyze and improve my prompt${
        selectedOperation.type === "resolve" ? "s" : ""
      }:${
        selectedOperation.type === "resolve"
          ? `\nComparison prompt:\n${
              selectedOperation.otherKwargs?.comparison_prompt || ""
            }\n\nResolution prompt:\n${
              selectedOperation.otherKwargs?.resolution_prompt || ""
            }`
          : `\n${selectedOperation.prompt}`
      }${bookmarksSection}`,
      id: "user-1",
    });

    // Create first revision after we get the response
    // This will happen in the useEffect that watches messages
  }, [
    selectedOperationId,
    operations,
    serializeState,
    append,
    setMessages,
    relevantBookmarks,
  ]);

  const handleBack = () => {
    setStep("select");
    // Clear messages when going back
    setMessages([]);
  };

  const handleClose = () => {
    onOpenChange(false);
    // Reset state when dialog closes
    setTimeout(() => {
      setStep("select");
      setMessages([]);
    }, 200);
  };

  const handleFeedbackSubmit = useCallback(async () => {
    if (!localFeedbackText.trim()) return;

    setExpectingNewRevision(true);

    const sourceRevisionIndex = selectedRevisionIndex ?? revisions.length - 1;
    const sourceRevision = revisions[sourceRevisionIndex];

    const newMessages = [...sourceRevision.messages];

    const feedbackMessage = {
      role: "user" as const,
      content: `Consider this feedback and provide ${
        selectedOperation?.type === "resolve"
          ? "updated prompts"
          : "an updated prompt"
      }: ${localFeedbackText}

Remember to ${
        selectedOperation?.type === "resolve"
          ? "include the complete revised prompts wrapped in <comparison_prompt></comparison_prompt> AND <resolution_prompt></resolution_prompt> tags"
          : "include the complete revised prompt wrapped in <prompt></prompt> tags"
      }, incorporating all previous improvements plus any new changes based on this feedback.`,
      id: `user-feedback-${newMessages.length}`,
    };

    setMessages([...newMessages]);
    await append(feedbackMessage);

    setLocalFeedbackText("");
    setFeedbackText("");
    setPopoverOpen(false);
  }, [
    localFeedbackText,
    selectedRevisionIndex,
    revisions,
    messages,
    append,
    selectedOperation,
  ]);

  // Update the direct edit handlers
  const handleDirectEditStart = () => {
    setIsDirectEditing(true);
  };

  const handleDirectEditComplete = () => {
    setIsDirectEditing(false);
  };

  const handleRevisionSelect = (index: number) => {
    setSelectedRevisionIndex(index);
    const revision = revisions[index];

    // Increment the chat key to force a new instance
    setChatKey((prev) => prev + 1);

    // In the next tick, set up the new chat state
    setTimeout(() => {
      setMessages(JSON.parse(JSON.stringify(revision.messages)));
      setEditedPrompt(revision.prompt || "");
    }, 0);

    // Reset direct edit state if it was active
    if (isDirectEditing) {
      setIsDirectEditing(false);
    }
  };

  // Update the save handler
  const handleSave = () => {
    if (editedPrompt && selectedOperation) {
      setShowSaveConfirm(true);
    }
  };

  // Now debouncedSetFeedback can use the debounce function since it's defined above
  const debouncedSetFeedback = useCallback(
    debounce((value: string) => {
      setFeedbackText(value);
    }, 100),
    []
  );

  // Add this new function to handle the actual save
  const handleConfirmedSave = (shouldClearNotes: boolean) => {
    if (editedPrompt && selectedOperation) {
      const lastMessage = messages[messages.length - 1];
      const schemaChanges = lastMessage
        ? extractSchemaChanges(lastMessage.content)
        : [];

      if (shouldClearNotes) {
        // Clear notes related to this operation
        relevantBookmarks.forEach((note) => {
          if (note.metadata?.operationName === selectedOperation.name) {
            const bookmark = bookmarks.find((b) =>
              b.notes.some((n) => n.id === note.id)
            );
            if (bookmark) {
              removeBookmark(bookmark.id);
            }
          }
        });
      }

      onSave(editedPrompt, schemaChanges);
      setMessages([]);
      setRevisions([]);
      onOpenChange(false);
      setStep("select");
      setShowSaveConfirm(false);
    }
  };

  return (
    <>
      <Dialog open={open} onOpenChange={handleClose}>
        <DialogContent className="max-w-7xl h-[90vh] flex flex-col gap-4">
          <DialogHeader className="flex-none">
            <div className="flex items-center gap-4 mb-2">
              {step === "analyze" && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleBack}
                  className="h-6 w-6"
                >
                  <ArrowLeft className="h-4 w-4" />
                </Button>
              )}
              <DialogTitle>Rewrite Prompt</DialogTitle>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <div
                className={`h-2 flex-1 rounded-full ${
                  step === "select" ? "bg-primary" : "bg-muted"
                }`}
              />
              <div
                className={`h-2 flex-1 rounded-full ${
                  step === "analyze" ? "bg-primary" : "bg-muted"
                }`}
              />
            </div>
            <DialogDescription className="mt-2">
              {step === "select"
                ? "Select an operation to improve its prompt"
                : "DocETL is analyzing and suggesting improvements"}
            </DialogDescription>
          </DialogHeader>

          <ScrollArea className="flex-1 w-full h-[calc(80vh-8rem)] overflow-y-auto">
            {step === "select" ? (
              <div className="flex flex-col gap-4 pr-4 pb-4">
                <Select
                  value={selectedOperationId}
                  onValueChange={setSelectedOperationId}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select operation" />
                  </SelectTrigger>
                  <SelectContent>
                    {operations
                      .filter((op) => op.llmType === "LLM")
                      .map((op) => (
                        <SelectItem key={op.id} value={op.id}>
                          {op.name}
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>

                {selectedOperation && (
                  <>
                    <div className="text-sm">
                      <div className="font-medium mb-2">
                        Current Prompt
                        {selectedOperation.type === "resolve" ? "s" : ""}:
                      </div>
                      {selectedOperation.type === "resolve" ? (
                        <div className="space-y-4">
                          <div>
                            <div className="font-medium text-sm mb-1">
                              Comparison Prompt:
                            </div>
                            <pre className="bg-muted p-2 rounded-md whitespace-pre-wrap">
                              {selectedOperation.otherKwargs
                                ?.comparison_prompt || ""}
                            </pre>
                          </div>
                          <div>
                            <div className="font-medium text-sm mb-1">
                              Resolution Prompt:
                            </div>
                            <pre className="bg-muted p-2 rounded-md whitespace-pre-wrap">
                              {selectedOperation.otherKwargs
                                ?.resolution_prompt || ""}
                            </pre>
                          </div>
                        </div>
                      ) : (
                        <pre className="bg-muted p-2 rounded-md whitespace-pre-wrap">
                          {selectedOperation.prompt}
                        </pre>
                      )}
                    </div>

                    <div className="text-sm">
                      <div className="font-medium mb-2">Feedback:</div>
                      <div className="bg-muted p-2 rounded-md">
                        {relevantBookmarks.length > 0 ? (
                          <ul className="list-disc list-inside space-y-1">
                            {relevantBookmarks.map((note, index) => (
                              <li key={index}>{note.note}</li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-muted-foreground">
                            No feedback or bookmarks found for this operation.
                          </p>
                        )}
                      </div>
                    </div>
                  </>
                )}

                <Button
                  onClick={handleImprove}
                  disabled={isLoading || !selectedOperation}
                  className="mt-4"
                >
                  Continue to Analysis
                </Button>
              </div>
            ) : (
              <div className="flex flex-col gap-4 pr-4 pb-4">
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-8">
                    <Loader2 className="h-8 w-8 animate-spin mb-4" />
                    <p className="text-sm text-muted-foreground">
                      Starting analysis...
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="border rounded-md p-4">
                      {messages
                        .filter((m) => m.role === "assistant")
                        .slice(-1)
                        .map((message, index) => (
                          <div
                            key={`${selectedRevisionIndex}-${index}`}
                            className="prose prose-sm max-w-none relative"
                          >
                            <ReactMarkdown>
                              {isLoading
                                ? message.content
                                : removePromptAndSchemaTags(message.content)}
                            </ReactMarkdown>
                            {isLoading && (
                              <div className="absolute -right-2 -bottom-2">
                                <Loader2 className="h-4 w-4 animate-spin" />
                              </div>
                            )}
                          </div>
                        ))}
                    </div>

                    {!isLoading && editedPrompt && selectedOperation && (
                      <>
                        <div className="border rounded-md p-4">
                          <h3 className="font-medium mb-2">Prompt Changes:</h3>
                          {isDirectEditing ? (
                            selectedOperation.type === "resolve" ? (
                              <div className="space-y-4">
                                <AutosizeTextarea
                                  value={
                                    typeof editedPrompt === "object"
                                      ? editedPrompt.comparison_prompt
                                      : ""
                                  }
                                  onChange={(e) =>
                                    setEditedPrompt((prev) =>
                                      typeof prev === "object"
                                        ? {
                                            ...prev,
                                            comparison_prompt: e.target.value,
                                          }
                                        : prev
                                    )
                                  }
                                  type="comparison"
                                />
                                <AutosizeTextarea
                                  value={
                                    typeof editedPrompt === "object"
                                      ? editedPrompt.resolution_prompt
                                      : ""
                                  }
                                  onChange={(e) =>
                                    setEditedPrompt((prev) =>
                                      typeof prev === "object"
                                        ? {
                                            ...prev,
                                            resolution_prompt: e.target.value,
                                          }
                                        : prev
                                    )
                                  }
                                  type="resolve"
                                />
                              </div>
                            ) : (
                              <AutosizeTextarea
                                value={
                                  typeof editedPrompt === "string"
                                    ? editedPrompt
                                    : ""
                                }
                                onChange={(e) =>
                                  setEditedPrompt(e.target.value)
                                }
                              />
                            )
                          ) : selectedOperation.type === "resolve" &&
                            typeof editedPrompt === "object" ? (
                            <div className="space-y-4">
                              <DiffView
                                oldText={
                                  selectedOperation.otherKwargs
                                    ?.comparison_prompt || ""
                                }
                                newText={editedPrompt.comparison_prompt}
                                type="comparison"
                              />
                              <DiffView
                                oldText={
                                  selectedOperation.otherKwargs
                                    ?.resolution_prompt || ""
                                }
                                newText={editedPrompt.resolution_prompt}
                                type="resolve"
                              />
                            </div>
                          ) : (
                            <DiffView
                              oldText={selectedOperation.prompt || ""}
                              newText={
                                typeof editedPrompt === "string"
                                  ? editedPrompt
                                  : ""
                              }
                            />
                          )}
                        </div>

                        {messages.length > 0 && (
                          <SchemaChangesDiff
                            changes={extractSchemaChanges(
                              messages[messages.length - 1].content
                            )}
                          />
                        )}

                        <div className="flex gap-2 justify-end">
                          <Button
                            variant="secondary"
                            onClick={() => {
                              if (isDirectEditing) {
                                // When switching to diff view, save the changes first
                                handleDirectEditComplete();
                              } else {
                                handleDirectEditStart();
                              }
                            }}
                          >
                            {isDirectEditing ? "See diff" : "Directly edit"}
                          </Button>

                          <Popover
                            open={popoverOpen}
                            onOpenChange={setPopoverOpen}
                            modal
                          >
                            <PopoverTrigger asChild>
                              <Button variant="secondary">Add feedback</Button>
                            </PopoverTrigger>
                            <PopoverContent
                              className="w-[500px] max-h-[80vh]"
                              side="left"
                              align="start"
                            >
                              <div className="flex flex-col gap-2 h-full">
                                <div className="flex-1 overflow-y-auto border-b max-h-[500px]">
                                  <div className="font-medium text-sm mb-1">
                                    Revision History:
                                  </div>
                                  <div>
                                    {buildRevisionTree(
                                      revisions,
                                      connections
                                    ).map((node, index) => (
                                      <RevisionTreeNode
                                        key={index}
                                        node={node}
                                        selectedIndex={selectedRevisionIndex}
                                        onSelect={handleRevisionSelect}
                                      />
                                    ))}
                                  </div>
                                </div>
                                <div className="flex-none pt-2">
                                  <Textarea
                                    placeholder="What would you like to improve?"
                                    value={localFeedbackText}
                                    onChange={(e) => {
                                      setLocalFeedbackText(e.target.value);
                                      debouncedSetFeedback(e.target.value);
                                    }}
                                    className="min-h-[100px] mb-2"
                                  />
                                  <div className="flex justify-end">
                                    <Button
                                      onClick={handleFeedbackSubmit}
                                      disabled={!localFeedbackText.trim()}
                                    >
                                      Submit Feedback
                                    </Button>
                                  </div>
                                </div>
                              </div>
                            </PopoverContent>
                          </Popover>

                          <Button onClick={handleSave} disabled={!editedPrompt}>
                            Save and Overwrite
                          </Button>
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>
            )}
          </ScrollArea>
        </DialogContent>
      </Dialog>

      <AlertDialog open={showSaveConfirm} onOpenChange={setShowSaveConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Save Changes</AlertDialogTitle>
            <AlertDialogDescription>
              Would you like to clear your notes for this operation?
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive hover:bg-destructive/90"
              onClick={() => handleConfirmedSave(true)}
            >
              Save & Clear Notes
            </AlertDialogAction>
            <AlertDialogAction onClick={() => handleConfirmedSave(false)}>
              Save & Keep Notes
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
