import { createAzure } from "@ai-sdk/azure";
import { createOpenAI } from "@ai-sdk/openai";
import { generateText } from "ai";
import { createClient } from "@supabase/supabase-js";

const supabase =
  process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_KEY
    ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY)
    : null;

export async function POST(req: Request) {
  let prompt: string = "";
  let personalApiKey: string | null = null;
  let useOpenAI: boolean = false;
  let source: string | null = null;

  try {
    ({ prompt } = await req.json());
    personalApiKey = req.headers.get("x-openai-key");
    useOpenAI = req.headers.get("x-use-openai") === "true";
    const namespace = req.headers.get("x-namespace");
    source = req.headers.get("x-source");
    const modelName = req.headers.get("x-model") || "o1-mini";

    let text: string;

    // Use OpenAI if explicitly requested via header
    if (useOpenAI) {
      const openai = createOpenAI({
        // Use personal API key if provided, otherwise fall back to environment variable
        apiKey: personalApiKey || process.env.OPENAI_API_KEY!,
        baseURL: process.env.OPENAI_API_BASE || "https://api.openai.com/v1",
        compatibility: "strict",
      });

      const result = await generateText({
        model: openai(process.env.MODEL_NAME || "gpt-4o-mini"),
        prompt,
      });

      text = result.text;
    } else {
      // Use Azure OpenAI
      const azure = createAzure({
        apiKey: process.env.AZURE_API_KEY!,
        apiVersion: process.env.AZURE_API_VERSION,
        resourceName: process.env.AZURE_RESOURCE_NAME,
      });

      const result = await generateText({
        model: azure(modelName),
        prompt,
      });

      text = result.text;

      // Log usage to Supabase if available
      if (supabase && result.usage) {
        const cost =
          (result.usage.promptTokens * 0.15) / 1_000_000 +
          (result.usage.completionTokens * 0.6) / 1_000_000;

        try {
          const { error } = await supabase.from("frontend_ai_requests").insert({
            messages: [{ role: "user", content: prompt }],
            namespace,
            cost,
            source: source || "nl-pipeline-generator",
          });

          if (error) {
            console.error("Supabase insert error:", error);
          } else {
            console.log("Successfully logged to Supabase");
          }
        } catch (err) {
          console.error("Failed to log to Supabase:", err);
        }
      }
    }

    return new Response(JSON.stringify({ text }), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Generate API error:", error);

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
