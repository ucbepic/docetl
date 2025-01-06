import { createAzure } from "@ai-sdk/azure";
import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";
import { createClient } from "@supabase/supabase-js";

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

const supabase =
  process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_KEY
    ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY)
    : null;

const MAX_TOTAL_CHARS = 500000;

function truncateMessages(messages: ChatMessage[]): ChatMessage[] {
  // Calculate total length
  let totalLength = messages.reduce((sum, msg) => sum + msg.content.length, 0);

  // If under limit, return original messages
  if (totalLength <= MAX_TOTAL_CHARS) {
    return messages;
  }

  // Clone messages to avoid mutating original array
  const truncatedMessages = JSON.parse(JSON.stringify(messages));

  while (totalLength > MAX_TOTAL_CHARS) {
    console.log(`Messages are too long (${totalLength} chars), truncating...`);
    // Find longest message
    let longestMsgIndex = 0;
    let maxLength = 0;

    truncatedMessages.forEach((msg: ChatMessage, index: number) => {
      if (msg.content.length > maxLength) {
        maxLength = msg.content.length;
        longestMsgIndex = index;
      }
    });

    // Get the message to truncate
    const message = truncatedMessages[longestMsgIndex];
    const contentLength = message.content.length;

    // Calculate the middle section to remove
    const quarterLength = Math.floor(contentLength / 4);
    const startPos = quarterLength;
    const endPos = contentLength - quarterLength;

    // Truncate the middle section
    message.content =
      message.content.substring(0, startPos) +
      " ... " +
      message.content.substring(endPos);

    // Recalculate total length
    totalLength = truncatedMessages.reduce(
      (sum: number, msg: ChatMessage) => sum + msg.content.length,
      0
    );
  }

  return truncatedMessages;
}

export async function POST(req: Request) {
  let messages: ChatMessage[] = [];
  let personalApiKey: string | null = null;
  let useOpenAI: boolean = false;
  let source: string | null = null;

  try {
    ({ messages } = await req.json());
    personalApiKey = req.headers.get("x-openai-key");
    useOpenAI = req.headers.get("x-use-openai") === "true";
    const namespace = req.headers.get("x-namespace");
    source = req.headers.get("x-source");
    const truncatedMessages = truncateMessages(messages);

    let result;

    // Use OpenAI if explicitly requested via header
    if (useOpenAI) {
      const openai = createOpenAI({
        // Use personal API key if provided, otherwise fall back to environment variable
        apiKey: personalApiKey || process.env.OPENAI_API_KEY!,
        baseURL: process.env.OPENAI_API_BASE || "https://api.openai.com/v1",
        compatibility: "strict",
      });

      result = await streamText({
        model: openai(process.env.MODEL_NAME || "gpt-4o-mini"),
        system:
          truncatedMessages.find((m: ChatMessage) => m.role === "system")
            ?.content || "You are a helpful assistant.",
        messages: truncatedMessages.filter(
          (m: ChatMessage) => m.role !== "system"
        ),
      });
    } else {
      // Use Azure OpenAI as before
      const azure = createAzure({
        apiKey: process.env.AZURE_API_KEY!,
        apiVersion: process.env.AZURE_API_VERSION,
        resourceName: process.env.AZURE_RESOURCE_NAME,
      });

      result = await streamText({
        model: azure(process.env.AZURE_DEPLOYMENT_NAME || "gpt-4o-mini"),
        system:
          truncatedMessages.find((m: ChatMessage) => m.role === "system")
            ?.content || "You are a helpful assistant.",
        messages: truncatedMessages.filter(
          (m: ChatMessage) => m.role !== "system"
        ),
        onFinish: async ({ usage }) => {
          if (!supabase) return;

          const cost =
            (usage.promptTokens * 0.15) / 1_000_000 +
            (usage.completionTokens * 0.6) / 1_000_000;

          try {
            const { error } = await supabase
              .from("frontend_ai_requests")
              .insert({
                messages,
                namespace,
                cost,
                source: source || "unknown",
              });

            if (error) {
              console.error("Supabase insert error:", error);
            } else {
              console.log("Successfully logged to Supabase");
            }
          } catch (err) {
            console.error("Failed to log to Supabase:", err);
          }
        },
      });
    }

    return result.toDataStreamResponse();
  } catch (error) {
    console.error("Chat API error:", error);

    return new Response(
      JSON.stringify({
        error:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred",
      }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
