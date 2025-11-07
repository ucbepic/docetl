import { createAzure } from "@ai-sdk/azure";
import { streamText, tool } from "ai";
import { z } from "zod";
import { ModalClient } from "modal";

// Initialize Modal client
let modalClient: ModalClient | null = null;
let modalApp: any = null;
let modalVolume: any = null;

async function initializeModal() {
  if (modalClient) return;

  const tokenId = process.env.MODAL_TOKEN_ID;
  const tokenSecret = process.env.MODAL_TOKEN_SECRET;

  if (!tokenId || !tokenSecret) {
    console.warn("Modal credentials not configured");
    return;
  }

  try {
    modalClient = new ModalClient({
      tokenId,
      tokenSecret,
    });

    // Get or create app
    modalApp = await modalClient.apps.fromName("docetl-scraper", {
      createIfMissing: true,
    });

    // Get or create volume for persistent storage
    modalVolume = await modalClient.volumes.fromName("scraper-data", {
      createIfMissing: true,
    });
  } catch (error) {
    console.error("Failed to initialize Modal:", error);
  }
}

async function executeModalSandbox(code: string): Promise<{
  success: boolean;
  output: string;
  error: string | null;
  stdout?: string;
  stderr?: string;
}> {
  await initializeModal();

  if (!modalClient || !modalApp) {
    return {
      success: false,
      output: "",
      error: "Modal client not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
    };
  }

  try {
    // Create Python image with common scraping libraries
    const baseImage = modalClient.images.fromRegistry("python:3.13-slim");
    const image = baseImage.dockerfileCommands([
      "RUN pip install --no-cache-dir requests beautifulsoup4 lxml selenium pandas",
    ]);

    // Create sandbox with volume mounted
    const sb = await modalClient.sandboxes.create(modalApp, image, {
      volumes: { "/data": modalVolume },
      timeoutMs: 5 * 60 * 1000, // 5 minutes
    });

    try {
      // Write code to a temporary file and execute it
      const scriptCode = `
import json
import sys
${code}

# If the code writes to /data/output.json, it's already done
# Otherwise, try to capture any output
try:
    with open('/data/output.json', 'r') as f:
        data = json.load(f)
        print(f"Dataset has {len(data)} items")
except FileNotFoundError:
    print("No output.json file found yet")
except Exception as e:
    print(f"Error reading output: {e}")
`;

      // Execute the Python code
      const process = await sb.exec(
        ["python", "-c", scriptCode],
        { timeoutMs: 4 * 60 * 1000 } // 4 minutes timeout
      );

      // Wait for process to complete
      const exitCode = await process.wait();
      const stdout = await process.stdout.readText();
      const stderr = await process.stderr.readText();

      // Clean up sandbox
      await sb.terminate();

      if (exitCode !== 0) {
        return {
          success: false,
          output: stdout,
          error: stderr || `Process exited with code ${exitCode}`,
          stdout,
          stderr,
        };
      }

      return {
        success: true,
        output: stdout,
        error: null,
        stdout,
        stderr,
      };
    } catch (execError) {
      await sb.terminate();
      throw execError;
    }
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
  await initializeModal();

  if (!modalClient || !modalVolume) {
    return {
      success: false,
      data: null,
      error: "Modal client not configured",
    };
  }

  try {
    // Read file from volume using a sandbox
    const image = modalClient.images.fromRegistry("python:3.13-slim");
    const sb = await modalClient.sandboxes.create(modalApp, image, {
      volumes: { "/data": modalVolume },
      timeoutMs: 60 * 1000, // 1 minute
    });

    try {
      const process = await sb.exec(
        [
          "python",
          "-c",
          `import json; import sys; 
try:
    with open('${filepath}', 'r') as f:
        data = json.load(f)
        print(json.dumps(data))
except FileNotFoundError:
    print('[]')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)`,
        ],
        { timeoutMs: 30 * 1000 }
      );

      // Wait for process to complete
      const exitCode = await process.wait();
      const stdout = await process.stdout.readText();
      const stderr = await process.stderr.readText();

      await sb.terminate();

      if (exitCode !== 0) {
        return {
          success: false,
          data: null,
          error: stderr || `Process exited with code ${exitCode}`,
        };
      }

      const data = JSON.parse(stdout || "[]");
      return {
        success: true,
        data: Array.isArray(data) ? data : [],
        error: null,
      };
    } catch (readError) {
      await sb.terminate();
      throw readError;
    }
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

// Tool: Search the internet using Tavily
const searchInternetTool = tool({
  description:
    "Search the internet for information using Tavily API. Use this to find websites, articles, or data sources related to the user's query.",
  parameters: z.object({
    query: z.string().describe("The search query"),
  }),
  execute: async ({ query }) => {
    const tavilyApiKey = process.env.TAVILY_API_KEY;

    if (!tavilyApiKey) {
      return {
        results: [],
        message:
          "Tavily API key not configured. Set TAVILY_API_KEY environment variable.",
      };
    }

    try {
      const response = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          api_key: tavilyApiKey,
          query,
          search_depth: "basic",
          include_answer: false,
          include_images: false,
          include_raw_content: false,
          max_results: 10,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        return {
          results: [],
          message: `Tavily API error: ${response.status} ${errorText}`,
        };
      }

      const data = await response.json();
      const results = (data.results || []).map((result: any) => ({
        title: result.title,
        url: result.url,
        snippet: result.content,
        score: result.score,
      }));

      return {
        results,
        message: `Found ${results.length} search results`,
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

  try {
    const body = await req.json();
    messages = body.messages || [];
    userQuery = body.userQuery || "";
    schema = body.schema;

    // Build system prompt
    const systemPrompt = `You are an AI agent that helps users scrape data from the internet. Your goal is to:

1. Understand what data the user wants to collect (from their query)
2. Search the internet to find relevant sources using Tavily
3. Write Python code to scrape those sources
4. Execute the code in a Modal sandbox
5. Collect the scraped data into a JSON file at /data/output.json
6. Continue iterating until you have collected sufficient data

The dataset should be a JSON array where each item is a dictionary matching the user's schema (if provided).

Available tools:
- search_internet: Search for websites and information using Tavily API
- execute_code: Run Python code in a Modal sandbox (has internet access, can write to /data/output.json)
- read_json: Read the current dataset from /data/output.json

Important notes for code execution:
- The sandbox has requests, beautifulsoup4, lxml, selenium, and pandas pre-installed
- Write your scraped data to /data/output.json as a JSON array
- You can append to existing data by reading /data/output.json first, then writing back
- Use proper error handling in your code
- The sandbox has internet access, so you can make HTTP requests

Workflow:
1. Start by searching for relevant websites/data sources using search_internet
2. Write scraping code using libraries like requests, BeautifulSoup, selenium, etc.
3. Execute the code to scrape data and write to /data/output.json
4. Read the JSON file to see what was collected
5. If more data is needed, search for more sources and repeat
6. Continue until you have enough data or hit reasonable limits (aim for 10-50 items minimum)

${schema ? `\nUser's desired schema:\n${schema}\n\nMake sure each item in the dataset matches this schema.` : "\nInfer an appropriate schema from the user's query."}

Be methodical and thorough. Explain what you're doing at each step.`;

    // Use Azure OpenAI only
    const azure = createAzure({
      apiKey: process.env.AZURE_API_KEY!,
      apiVersion: process.env.AZURE_API_VERSION,
      resourceName: process.env.AZURE_RESOURCE_NAME,
    });

    // Use GPT-5 as default model
    const modelName = process.env.AZURE_DEPLOYMENT_NAME || "gpt-5";
    
    // Agent loop control: stop when we have sufficient data or hit limits
    let datasetItemCount = 0;
    let stepCount = 0;
    const maxSteps = 20;
    const minDatasetSize = 10;
    const maxDatasetSize = 100;

    const result = await streamText({
      model: azure(modelName),
      system: systemPrompt,
      messages: messages.filter((m) => m.role !== "system"),
      tools: {
        search_internet: searchInternetTool,
        execute_code: executeCodeTool,
        read_json: readJsonTool,
      },
      maxSteps,
      onStepFinish: async ({ toolCalls, toolResults, finishReason }) => {
        stepCount++;
        
        // Update dataset count from read_json tool results
        for (let i = 0; i < toolCalls.length; i++) {
          const toolCall = toolCalls[i];
          const toolResult = toolResults[i];
          
          if (toolCall.toolName === "read_json" && toolResult && typeof toolResult === "object") {
            const result = toolResult as { data?: unknown[]; count?: number };
            if (Array.isArray(result.data)) {
              datasetItemCount = result.data.length;
            } else if (typeof result.count === "number") {
              datasetItemCount = result.count;
            }
          }
        }

        // Stop condition: we have enough data or hit max steps
        if (datasetItemCount >= minDatasetSize || stepCount >= maxSteps) {
          return {
            finishReason: "stop" as const,
            text: datasetItemCount >= minDatasetSize
              ? `Successfully collected ${datasetItemCount} items. Task complete!`
              : `Reached maximum steps (${maxSteps}). Collected ${datasetItemCount} items.`,
          };
        }

        // Continue if we need more data
        return undefined;
      },
    });

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
