import { createAzure } from "@ai-sdk/azure";
import { streamText, tool } from "ai";
import { z } from "zod";
// Modal JavaScript SDK - see https://modal-labs.github.io/libmodal/ for documentation
import { ModalClient } from "modal";

// Initialize Modal client
let modalClient: ModalClient | null = null;
type ModalAppHandle = Awaited<ReturnType<ModalClient["apps"]["fromName"]>>;
type ModalVolumeHandle = Awaited<
  ReturnType<ModalClient["volumes"]["fromName"]>
>;
let modalApp: ModalAppHandle | null = null;
let modalVolume: ModalVolumeHandle | null = null;

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

async function executeModalSandbox(
  code: string,
  sessionId: string
): Promise<{
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
      error:
        "Modal client not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
    };
  }

  try {
    // Create Python image with common scraping libraries
    const baseImage = modalClient.images.fromRegistry("python:3.13-slim");
    const image = baseImage.dockerfileCommands([
      "RUN pip install --no-cache-dir requests httpx beautifulsoup4 lxml selenium playwright pandas PyPDF2 pdfplumber openpyxl cloudscraper aiohttp rich",
    ]);

    // Create sandbox with volume mounted
    const sb = await modalClient.sandboxes.create(modalApp, image, {
      volumes: { "/data": modalVolume },
      timeoutMs: 5 * 60 * 1000, // 5 minutes
    });

    try {
      // Use flat file structure: /data/{sessionId}.json (no subdirectory)
      const outputPath = `/data/${sessionId}.json`;

      // Write code to a temporary file and execute it
      const scriptCode = `
import json
import sys
import os

${code}

# If the code writes to ${outputPath}, it's already done
# Otherwise, try to capture any output
try:
    with open('${outputPath}', 'r') as f:
        data = json.load(f)
        print(f"Dataset has {len(data)} items")
except FileNotFoundError:
    print("No output file found yet")
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

async function readModalFile(
  filepath: string,
  sessionId: string
): Promise<{
  success: boolean;
  data: unknown;
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
      // Always use the session-specific file path
      const filePath = `/data/${sessionId}.json`;

      const process = await sb.exec(
        [
          "python",
          "-c",
          `import json; import sys;
try:
    with open('${filePath}', 'r') as f:
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

// Track search API calls per session (in-memory, resets on server restart)
// In production, consider using Redis or a database for persistence
const searchCounts = new Map<string, number>();
const tavilyCredits = new Map<string, number>();
const MAX_SEARCHES_PER_SESSION = 5; // Limit searches per session
const MAX_TAVILY_CREDITS = 25;
const BASIC_SEARCH_CREDITS = 1;
const ADVANCED_SEARCH_CREDITS = 2;
const CRAWL_CREDITS = 3;
const MAP_CREDITS = 2;
const getSearchCount = (sid: string) => searchCounts.get(sid) || 0;
const incrementSearchCount = (sid: string) => {
  const current = getSearchCount(sid);
  searchCounts.set(sid, current + 1);
  return current + 1;
};
const getCreditsUsed = (sid: string) => tavilyCredits.get(sid) || 0;
const addCredits = (sid: string, credits: number) => {
  const current = getCreditsUsed(sid) + credits;
  tavilyCredits.set(sid, current);
  return current;
};

// Tool: Search the internet using Tavily
// This will be created with sessionId and search tracking context
const createSearchInternetTool = (
  sessionId: string,
  getSearchCount: (sid: string) => number,
  incrementSearchCount: (sid: string) => number,
  maxSearches: number
) =>
  tool({
    description:
      "Discover new sources with Tavily. Modes: search (default), crawl (site sweep), map (topology). Stay within ~25 credits per session.",
    inputSchema: z.object({
      query: z
        .string()
        .describe(
          "Search query or crawl instructions. Provide concise, targeted language."
        ),
      mode: z
        .enum(["search", "crawl", "map"])
        .default("search")
        .describe(
          "Tavily capability: search (SERP style), crawl (content extraction), map (link discovery)."
        ),
      url: z
        .string()
        .optional()
        .describe("Starting URL for crawl or map operations."),
      depth: z
        .enum(["basic", "advanced"])
        .optional()
        .describe(
          "Search depth for search mode. Advanced costs extra credits but returns richer snippets."
        ),
    }),
    execute: async ({ query, mode, url, depth }) => {
      const usedCredits = getCreditsUsed(sessionId);
      if (usedCredits >= MAX_TAVILY_CREDITS) {
        return {
          results: [],
          message: `Tavily credit budget exhausted (${usedCredits}/${MAX_TAVILY_CREDITS}). Continue with existing sources and execute_code.`,
        };
      }

      if (mode === "search" && getSearchCount(sessionId) >= maxSearches) {
        return {
          results: [],
          message: `Search limit reached (${maxSearches} searches per session). Use execute_code with the URLs you already have or request specific URLs from the user.`,
        };
      }

      const tavilyApiKey = process.env.TAVILY_API_KEY;
      if (!tavilyApiKey) {
        return {
          results: [],
          message:
            "Tavily API key not configured. Set TAVILY_API_KEY environment variable.",
        };
      }

      const selectedDepth = depth ?? "basic";
      const creditCost = (() => {
        if (mode === "crawl") {
          return CRAWL_CREDITS;
        }
        if (mode === "map") {
          return MAP_CREDITS;
        }
        return selectedDepth === "advanced"
          ? ADVANCED_SEARCH_CREDITS
          : BASIC_SEARCH_CREDITS;
      })();

      if (usedCredits + creditCost > MAX_TAVILY_CREDITS) {
        return {
          results: [],
          message: `Tavily call skipped. Operation requires ${creditCost} credits but only ${
            MAX_TAVILY_CREDITS - usedCredits
          } remain. Continue scraping with execute_code.`,
        };
      }

      const endpoint = (() => {
        if (mode === "crawl") {
          return "https://api.tavily.com/crawl";
        }
        if (mode === "map") {
          return "https://api.tavily.com/map";
        }
        return "https://api.tavily.com/search";
      })();

      type TavilySearchResponse = {
        results?: Array<{
          title?: string;
          url?: string;
          content?: string;
          score?: number;
        }>;
      };

      type TavilyCrawlResponse = {
        results?: Array<{
          url?: string;
          title?: string;
          raw_content?: string;
        }>;
        data?: unknown;
      };

      type TavilyMapResponse = {
        results?: string[];
      };

      const basePayload: Record<string, unknown> = {
        api_key: tavilyApiKey,
      };

      let bodyPayload: Record<string, unknown>;

      if (mode === "crawl" || mode === "map") {
        if (!url) {
          return {
            results: [],
            message: `${mode} mode requires a url parameter.`,
          };
        }
        bodyPayload = {
          ...basePayload,
          url,
          instructions: query,
        };
      } else {
        bodyPayload = {
          ...basePayload,
          query,
          search_depth: selectedDepth,
          include_answer: false,
          include_images: false,
          include_raw_content: selectedDepth === "advanced",
          max_results: 5,
        };
      }

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(bodyPayload),
        });

        if (!response.ok) {
          const errorText = await response.text();
          return {
            results: [],
            message: `Tavily API error: ${response.status} ${errorText}`,
          };
        }

        const credited = addCredits(sessionId, creditCost);
        const currentCount =
          mode === "search"
            ? incrementSearchCount(sessionId)
            : getSearchCount(sessionId);

        if (mode === "crawl") {
          const data = (await response.json()) as TavilyCrawlResponse;
          const crawlResults = Array.isArray(data.results) ? data.results : [];
          const mappedResults = crawlResults.map((result) => ({
            title: result.title ?? "Untitled",
            url: result.url ?? "",
            snippet: result.raw_content ?? "",
          }));

          return {
            results: mappedResults,
            message: `Crawl queued ${mappedResults.length} URLs (${credited}/${MAX_TAVILY_CREDITS} credits used). Prioritise follow-up scrapes with execute_code.`,
          };
        }

        if (mode === "map") {
          const data = (await response.json()) as TavilyMapResponse;
          const mapResults = Array.isArray(data.results) ? data.results : [];
          const normalizedResults = mapResults.map((resultUrl) => ({
            title: "Discovered URL",
            url: resultUrl,
            snippet: "Discovered via Tavily map.",
          }));

          return {
            results: normalizedResults,
            message: `Map discovered ${normalizedResults.length} candidate URLs (${credited}/${MAX_TAVILY_CREDITS} credits used). Use execute_code to evaluate them.`,
          };
        }

        const data = (await response.json()) as TavilySearchResponse;
        const rawResults = Array.isArray(data.results) ? data.results : [];

        const results = rawResults.map((result) => ({
          title: result.title ?? "Untitled",
          url: result.url ?? "",
          snippet: result.content ?? "",
          score: result.score,
        }));

        return {
          results,
          message: `Found ${results.length} search results (${currentCount}/${maxSearches} searches, ${credited}/${MAX_TAVILY_CREDITS} credits used, depth=${selectedDepth}). Follow up with execute_code immediately.`,
        };
      } catch (error) {
        return {
          results: [],
          message: `Search error: ${
            error instanceof Error ? error.message : "Unknown error"
          }. Continue with execute_code and existing links.`,
        };
      }
    },
  });

// Tool: Execute code in Modal sandbox
// This will be created with sessionId context
const createExecuteCodeTool = (sessionId: string) =>
  tool({
    description: `Execute Python in a Modal sandbox with internet access.

LOGGING (REQUIRED):
- from rich.console import Console
- console = Console()
- Use console.rule(), console.log(), and console.print_exception() so output renders cleanly.
- Wrap long tasks with console.status("Scraping …", spinner="dots").

DATASET PATTERN:
- Filepath: /data/${sessionId}.json
- import json, import os
- console.log("Loading dataset", filepath=filepath)
- dataset = json.load(open(filepath)) if os.path.exists(filepath) else []
- console.log("Existing rows", total=len(dataset))
- Build new_items, extend dataset, json.dump(..., indent=2)
- console.log("Dataset updated", added=len(new_items), total=len(dataset))
- Immediately scrape any URLs returned by Tavily search/crawl/map outputs before issuing new searches.

PDF HANDLING:
- Detect PDFs by URL or response headers.
- Use pdfplumber (preferred) or PyPDF2.
- Example:
  import pdfplumber
  with pdfplumber.open("local.pdf") as pdf:
      console.log("PDF loaded", pages=len(pdf.pages))
      for page in pdf.pages:
          text = page.extract_text() or ""
          # parse text into structured rows

ERROR REPORTING:
- Catch exceptions and surface them with console.print_exception(show_locals=True).
- Always finish with dataset stats so the agent can decide next steps.`,
    inputSchema: z.object({
      code: z.string().describe("The Python code to execute"),
      description: z
        .string()
        .optional()
        .describe("Brief description of what this code does"),
    }),
    execute: async ({ code, description }) => {
      const result = await executeModalSandbox(code, sessionId);

      return {
        success: result.success,
        output: result.output || result.stdout || "",
        error: result.error || result.stderr || null,
        message:
          description ||
          (result.success
            ? "Code execution completed"
            : "Code execution failed"),
      };
    },
  });

// Tool: Read JSON file
// This will be created with sessionId context
const createReadJsonTool = (sessionId: string) =>
  tool({
    description: `Read a JSON file from the Modal volume. Use this to check the current state of the dataset being collected. The default filepath is /data/${sessionId}.json.`,
    inputSchema: z.object({
      filepath: z
        .string()
        .default(`/data/${sessionId}.json`)
        .describe("Path to the JSON file in the Modal volume"),
    }),
    execute: async ({ filepath }) => {
      const result = await readModalFile(filepath, sessionId);

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
        message: `File read successfully. Found ${
          Array.isArray(result.data) ? result.data.length : 0
        } items.`,
      };
    },
  });

export async function POST(req: Request) {
  let messages: ChatMessage[] = [];
  let schema: string | undefined = undefined;
  let sessionId: string | undefined = undefined;

  try {
    const body = await req.json();
    messages = body.messages || [];
    schema = body.schema;
    sessionId = body.sessionId;

    // Generate session ID if not provided (for backward compatibility)
    // Format: YYYYMMDD-{uuid} (e.g., 20250130-abc123-def456-...)
    if (!sessionId) {
      const now = new Date();
      const dateStr = now.toISOString().split("T")[0].replace(/-/g, ""); // YYYYMMDD
      const uuid = crypto.randomUUID();
      sessionId = `${dateStr}-${uuid}`;
    }

    // Build system prompt
    const systemPrompt = `You are DocScraper, an iterative web scraping agent.

MISSION:
- Deliver a clean, structured dataset saved at /data/${sessionId}.json that satisfies the user's request.
- Example objective: collect blog posts from academics covering NSF and government budget cuts for science funding (titles, authors, publication dates, summaries, URLs).

TOOLS & BUDGET:
- search_internet → Tavily API. Default to basic search depth (~1 credit). Escalate to advanced (≈2 credits) only when richer snippets are essential. Use crawl for structured extraction sweeps (~3 credits) and map to expand URL frontiers (~2 credits). Total allowance ≈25 credits and ${MAX_SEARCHES_PER_SESSION} searches per session. Reference parameter behaviour in Tavily's SDK guide (https://docs.tavily.com/sdk/javascript/reference).
- execute_code → Python sandbox with requests, httpx, BeautifulSoup, lxml, selenium, playwright, pandas, PyPDF2, pdfplumber, openpyxl, cloudscraper, aiohttp, rich, and more installed.
- read_json → Inspect the persisted dataset after each mutation.

ITERATIVE LOOP:
1. Inspect prior messages and the dataset (via read_json when needed).
2. Plan the next step: search/crawl/map for new sources only when URLs are missing or stale; otherwise scrape, clean, or enrich existing data.
3. Use execute_code to scrape or transform data, logging every stage with rich.
4. Immediately verify the dataset with read_json, summarise quality, and decide whether to continue scraping, cleaning, or stop.

RICH LOGGING STANDARD:
- Always import from rich.console import Console and instantiate console = Console().
- Use console.rule() for section separators, console.status() for long tasks, console.log() for structured key/value updates, and console.print_exception(show_locals=True) for errors.
- Finish every run with console.log("dataset_summary", added=..., total=...).

PDF & BINARY CONTENT:
- When a URL is or returns a PDF, download it and parse with pdfplumber (preferred) or PyPDF2. Log page counts and how each page contributes to structured rows before writing data.
- Skip images or binary formats you cannot parse; log the limitation.

DATASET QUALITY GATES:
- Every record must include source url plus fields that match the target schema.
- De-duplicate on meaningful keys, normalise dates, and fill missing values when reasonable.
- Aim for ≈30 high-quality rows unless the user requests fewer or the domain cannot yield that many.
- Use load → append → write pattern; never overwrite without merging.

SEARCH DISCIPLINE:
- Track Tavily credits and searches. Avoid repeat queries unless needed. Prefer exploring additional pages from already discovered domains with execute_code.

COMMUNICATION:
- Narrate decisions clearly. After each tool call, explain what happened, what data is now available, and the next action you plan.

SCHEMA:
${
  schema
    ? `Match the provided schema exactly:
${schema}
Include a url field for every item.`
    : `Infer a consistent schema from the user's query and include a url field for every item.`
}`;

    // Use Azure OpenAI - apiVersion defaults to 'v1'
    const azure = createAzure({
      apiKey: process.env.AZURE_API_KEY!,
      resourceName: process.env.AZURE_RESOURCE_NAME,
    });

    // Use GPT-5 as default model
    const modelName = process.env.SCRAPER_AZURE_DEPLOYMENT_NAME || "gpt-4.1";

    const result = streamText({
      model: azure(modelName),
      system: systemPrompt,
      messages: messages.filter((m) => m.role !== "system"),
      tools: {
        search_internet: createSearchInternetTool(
          sessionId,
          getSearchCount,
          incrementSearchCount,
          MAX_SEARCHES_PER_SESSION
        ),
        execute_code: createExecuteCodeTool(sessionId),
        read_json: createReadJsonTool(sessionId),
      },
      temperature: 1,
      maxRetries: 20,
      onStepFinish: async () => {
        // Tool execution completed
      },
    });

    return result.toUIMessageStreamResponse();
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
