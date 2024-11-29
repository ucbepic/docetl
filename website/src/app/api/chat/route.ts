import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";

// Allow streaming responses up to 60 seconds
export const maxDuration = 60;
const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_API_BASE,
  compatibility: "strict", // strict mode, enable when using the OpenAI API
});

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = await streamText({
    model: openai(process.env.MODEL_NAME),
    system: "You are a helpful assistant.",
    messages,
  });

  return result.toDataStreamResponse();
}
