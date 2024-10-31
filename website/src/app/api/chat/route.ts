import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";

// Allow streaming responses up to 60 seconds
export const maxDuration = 60;

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = await streamText({
    model: openai("gpt-4o-mini"),
    system: "You are a helpful assistant.",
    messages,
  });

  return result.toDataStreamResponse();
}
