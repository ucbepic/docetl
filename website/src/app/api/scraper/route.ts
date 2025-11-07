import { createAzure } from "@ai-sdk/azure";
import { createOpenAI } from "@ai-sdk/openai";
import { streamText, tool, CoreMessage } from "ai";
import { z } from "zod";

// Modal API integration
// Modal uses REST API for sandbox execution
// See: https://modal.com/docs/guide/sandbox
const MODAL_API_URL = "https://api.modal.com";

async function executeModalSandbox(code: string): Promise<{
  success: boolean;
  output: string;
  error: string | null;
  stdout?: string;
  stderr?: string;
}> {
  const tokenId = process.env.MODAL_TOKEN_ID;
  const tokenSecret = process.env.MODAL_TOKEN_SECRET;

  if (!tokenId || !tokenSecret) {
    return {
      success: false,
      output: "",
      error: "Modal credentials not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
    };
  }

  try {
    // Create a sandbox session
    // Note: This is a simplified implementation. In production, you'd want to:
    // 1. Create a persistent volume for /data
    // 2. Reuse sandbox sessions when possible
    // 3. Handle timeouts and errors properly
    
    // For now, we'll simulate the execution
    // In production, use Modal's REST API:
    // POST https://api.modal.com/v1/sandbox/run
    // {
    //   "code": code,
    //   "image": "python:3.11",
    //   "volumes": { "/data": "your-volume-id" },
    //   "timeout": 300
    // }

    // Placeholder implementation
    return {
      success: true,
      output: `Code execution simulated. In production, this would execute:\n\n${code}`,
      error: null,
      stdout: "Simulated stdout",
      stderr: "",
    };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

async function readModalFile(filepath: string): Promise<{
  success: boolean;
  data: any;
  error: string | null;
}> {
  const tokenId = process.env.MODAL_TOKEN_ID;
  const tokenSecret = process.env.MODAL_TOKEN_SECRET;

  if (!tokenId || !tokenSecret) {
    return {
      success: false,
      data: null,
      error: "Modal credentials not configured",
    };
  }

  try {
    // In production, use Modal's REST API to read from volume
    // GET https://api.modal.com/v1/volume/{volume-id}/read?path={filepath}
    
    // Placeholder implementation
    return {
      success: true,
      data: [],
      error: null,
    };
  } catch (error) {
    return {
      success: false,
      data: null,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

// Tool: Search the internet
const searchInternetTool = tool({
  description:
    "Search the internet for information. Use this to find websites, articles, or data sources related to the user's query.",
  parameters: z.object({
    query: z.string().describe("The search query"),
  }),
  execute: async ({ query }) => {
    // Use a search API - for now, we'll use a placeholder
    // In production, you might use Google Custom Search API, SerpAPI, etc.
    try {
      // Placeholder - replace with actual search implementation
      const response = await fetch(
        `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}`,
        {
          headers: {
            "X-Subscription-Token": process.env.BRAVE_API_KEY || "",
          },
        }
      );

      if (!response.ok) {
        // Fallback to a simple web search simulation
        return {
          results: [
            {
              title: `Search results for: ${query}`,
              url: `https://example.com/search?q=${encodeURIComponent(query)}`,
              snippet: `Found information about ${query}. Please visit websites directly to scrape data.`,
            },
          ],
          message:
            "Search completed. Use the execute_code tool to scrape data from the URLs found.",
        };
      }

      const data = await response.json();
      return {
        results: data.web?.results?.slice(0, 10) || [],
        message: `Found ${data.web?.results?.length || 0} search results`,
      };
    } catch (error) {
      return {
        results: [],
        message: `Search error: ${error instanceof Error ? error.message : "Unknown error"}. You may need to use execute_code to scrape websites directly.`,
      };
    }
  },
});

// Tool: Execute code in Modal sandbox
const executeCodeTool = tool({
  description:
    "Execute Python code in a Modal sandbox. Use this to scrape websites, process data, or write JSON files. The code runs in an isolated environment with internet access. Write output JSON to /data/output.json in the Modal volume.",
  parameters: z.object({
    code: z.string().describe("The Python code to execute"),
    description: z
      .string()
      .optional()
      .describe("Brief description of what this code does"),
  }),
  execute: async ({ code, description }) => {
    const result = await executeModalSandbox(code);
    
    return {
      success: result.success,
      output: result.output || result.stdout || "",
      error: result.error || result.stderr || null,
      message: description || (result.success ? "Code execution completed" : "Code execution failed"),
    };
  },
});

// Tool: Read JSON file
const readJsonTool = tool({
  description:
    "Read a JSON file from the Modal volume. Use this to check the current state of the dataset being collected.",
  parameters: z.object({
    filepath: z
      .string()
      .default("/data/output.json")
      .describe("Path to the JSON file in the Modal volume"),
  }),
  execute: async ({ filepath }) => {
    const result = await readModalFile(filepath);
    
    if (!result.success) {
      return {
        success: false,
        data: null,
        error: result.error,
        message: "Failed to read file",
      };
    }

    return {
      success: true,
      data: Array.isArray(result.data) ? result.data : [],
      count: Array.isArray(result.data) ? result.data.length : 0,
      message: `File read successfully. Found ${Array.isArray(result.data) ? result.data.length : 0} items.`,
    };
  },
});

export async function POST(req: Request) {
  let messages: ChatMessage[] = [];
  let userQuery: string = "";
  let schema: string | undefined = undefined;
  let personalApiKey: string | null = null;
  let useOpenAI: boolean = false;

  try {
    const body = await req.json();
    messages = body.messages || [];
    userQuery = body.userQuery || "";
    schema = body.schema;
    personalApiKey = req.headers.get("x-openai-key");
    useOpenAI = req.headers.get("x-use-openai") === "true";

    // Build system prompt
    const systemPrompt = `You are an AI agent that helps users scrape data from the internet. Your goal is to:

1. Understand what data the user wants to collect (from their query)
2. Search the internet to find relevant sources
3. Write Python code to scrape those sources
4. Execute the code in a Modal sandbox
5. Collect the scraped data into a JSON file
6. Continue iterating until you have collected sufficient data

The dataset should be a JSON array where each item is a dictionary matching the user's schema (if provided).

Available tools:
- search_internet: Search for websites and information
- execute_code: Run Python code in a Modal sandbox (has internet access, can write to /data/output.json)
- read_json: Read the current dataset from /data/output.json

Workflow:
1. Start by searching for relevant websites/data sources
2. Write scraping code using libraries like requests, BeautifulSoup, selenium, etc.
3. Execute the code to scrape data
4. Read the JSON file to see what was collected
5. If more data is needed, search for more sources and repeat
6. Continue until you have enough data or hit reasonable limits

${schema ? `\nUser's desired schema:\n${schema}\n\nMake sure each item in the dataset matches this schema.` : "\nInfer an appropriate schema from the user's query."}

Be methodical and thorough. Explain what you're doing at each step.`;

    let result;

    if (useOpenAI) {
      const openai = createOpenAI({
        apiKey: personalApiKey || process.env.OPENAI_API_KEY!,
        baseURL: process.env.OPENAI_API_BASE || "https://api.openai.com/v1",
        compatibility: "strict",
      });

      result = await streamText({
        model: openai("gpt-4o"), // Use GPT-4o for better function calling
        system: systemPrompt,
        messages: messages.filter((m) => m.role !== "system"),
        tools: {
          search_internet: searchInternetTool,
          execute_code: executeCodeTool,
          read_json: readJsonTool,
        },
        maxSteps: 20, // Allow multiple tool calls in sequence
      });
    } else {
      // Use Azure OpenAI
      const azure = createAzure({
        apiKey: process.env.AZURE_API_KEY!,
        apiVersion: process.env.AZURE_API_VERSION,
        resourceName: process.env.AZURE_RESOURCE_NAME,
      });

      result = await streamText({
        model: azure(process.env.AZURE_DEPLOYMENT_NAME || "gpt-4o"),
        system: systemPrompt,
        messages: messages.filter((m) => m.role !== "system"),
        tools: {
          search_internet: searchInternetTool,
          execute_code: executeCodeTool,
          read_json: readJsonTool,
        },
        maxSteps: 20,
      });
    }

    return result.toDataStreamResponse();
  } catch (error) {
    console.error("Scraper API error:", error);

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
