import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";

// Allow streaming responses up to 60 seconds
export const maxDuration = 60;

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();
    const apiKey =
      req.headers.get("x-openai-key") || process.env.OPENAI_API_KEY;

    if (!apiKey) {
      return new Response(
        JSON.stringify({ error: "OpenAI API key is required" }),
        { status: 400 }
      );
    }

    const openai = createOpenAI({
      apiKey,
      baseURL: process.env.OPENAI_API_BASE,
      compatibility: "strict",
    });

    const result = await streamText({
      model: openai(process.env.MODEL_NAME),
      system: "You are a helpful assistant.",
      messages,
    });

    return result.toDataStreamResponse();
  } catch (error) {
    console.error("Chat API error:", error);
    return new Response(
      error instanceof Error ? error.message : "An error occurred",
      { status: 500 }
    );
  }
}
