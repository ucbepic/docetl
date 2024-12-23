import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

const MAX_TOTAL_CHARS = 500000;

function truncateMessages(messages: ChatMessage[]): ChatMessage[] {
  // Calculate total length
  let totalLength = messages.reduce((sum, msg) => sum + msg.content.length, 0);

  // If under limit, return original messages
  if (totalLength <= MAX_TOTAL_CHARS) {
    return messages;
  }

  // Clone messages to avoid mutating original array
  let truncatedMessages = JSON.parse(JSON.stringify(messages));

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

export const maxDuration = 60;

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();
    const apiKey =
      req.headers.get("x-openai-key") || process.env.OPENAI_API_KEY;

    console.log("Chat API: OpenAI key present:", !!apiKey);

    if (!apiKey) {
      return new Response(
        JSON.stringify({
          error:
            "OpenAI API key is required. Please add your API key in Edit > Edit API Keys",
        }),
        {
          status: 400,
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
    }

    // Truncate messages if needed
    const truncatedMessages = truncateMessages(messages);
    const wasMessagesTruncated = truncatedMessages !== messages;
    if (wasMessagesTruncated) {
      console.log("Messages were truncated to fit within size limit");
    }

    const openai = createOpenAI({
      apiKey,
      baseURL: process.env.OPENAI_API_BASE || "https://api.openai.com/v1",
      compatibility: "strict",
    });

    const modelName = process.env.MODEL_NAME || "gpt-4o-mini";

    const result = await streamText({
      model: openai(modelName),
      system:
        truncatedMessages.find((m: ChatMessage) => m.role === "system")
          ?.content || "You are a helpful assistant.",
      messages: truncatedMessages.filter(
        (m: ChatMessage) => m.role !== "system"
      ),
    });

    return result.toDataStreamResponse();
  } catch (error) {
    console.error("Chat API error:", error);

    return new Response(
      JSON.stringify({
        error:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred",
        details: error instanceof Error ? error.stack : undefined,
      }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }
}
