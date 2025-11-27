import { createAzure } from "@ai-sdk/azure";
import { streamText, tool, CoreMessage } from "ai";
import { z } from "zod";
import { createClient } from "@supabase/supabase-js";
// Modal JavaScript SDK - see https://modal-labs.github.io/libmodal/ for documentation
import { ModalClient } from "modal";

// Initialize Supabase client
const supabase =
  process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_KEY
    ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY)
    : null;

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
      "RUN pip install --no-cache-dir requests httpx beautifulsoup4 lxml selenium playwright pandas PyPDF2 pdfplumber openpyxl cloudscraper aiohttp",
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

      // Clean up sandbox - data persists automatically after termination
      await sb.terminate();
      console.log(
        `[Volume] Sandbox terminated for session ${sessionId}, data should be persisted`
      );

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

interface ToolInvocation {
  toolName: string;
  toolCallId: string;
  state?: string;
  args?: Record<string, unknown>;
  result?: unknown;
}

interface ChatMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  toolInvocations?: ToolInvocation[];
  experimental_attachments?: unknown[];
  [key: string]: unknown; // Allow additional fields from AI SDK
}

// Track search API calls per session (in-memory, resets on server restart)
// In production, consider using Redis or a database for persistence
const searchCounts = new Map<string, number>();
const tavilyCredits = new Map<string, number>();
const MAX_SEARCHES_PER_SESSION = 8; // Limit searches per session (increased since we're using basic only)
const MAX_TAVILY_CREDITS = 15; // Reduced budget - focus on basic searches + code-based scraping
const BASIC_SEARCH_CREDITS = 1;
const CRAWL_CREDITS = 3; // Expensive - use sparingly
const MAP_CREDITS = 2; // Expensive - use sparingly
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
      "Discover new sources with Tavily. Returns up to 20 URLs per search. **IMMEDIATELY SCRAPE THESE URLs WITH execute_code AFTER CALLING THIS!** ONLY use 'search' mode (1 credit). Avoid crawl (3 credits) and map (2 credits) - use execute_code to extract links instead. Budget: ~15 credits per session.",
    parameters: z.object({
      query: z
        .string()
        .describe(
          "Search query. Provide concise, targeted language. Returns up to 20 URLs. Use execute_code to discover additional URLs from scraped pages."
        ),
      mode: z
        .enum(["search", "crawl", "map"])
        .default("search")
        .describe(
          "Tavily capability. STRONGLY PREFER 'search' (1 credit). Only use crawl/map if absolutely necessary and user explicitly requests it."
        ),
      url: z
        .string()
        .optional()
        .describe("Starting URL for crawl or map operations."),
      // Removed depth parameter - always use basic to save credits
    }),
    execute: async ({ query, mode, url }) => {
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

      // Always use basic depth to minimize costs
      const selectedDepth = "basic";
      const creditCost = (() => {
        if (mode === "crawl") {
          return CRAWL_CREDITS;
        }
        if (mode === "map") {
          return MAP_CREDITS;
        }
        // Always basic search, always 1 credit
        return BASIC_SEARCH_CREDITS;
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
          include_raw_content: false, // Always false since we only use basic
          max_results: 20, // Max free results per search - maximize value per credit
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

**SCRAPE ALL TAVILY URLs IMMEDIATELY**: When you get URLs from search_internet, use this tool RIGHT AWAY to scrape ALL of them!

PARALLEL FETCHING (REQUIRED):
- ALWAYS fetch multiple URLs in parallel, NEVER serially in a loop
- Use asyncio + aiohttp or concurrent.futures.ThreadPoolExecutor
- Example with aiohttp:
  import asyncio, aiohttp
  async def fetch_url(session, url):
      async with session.get(url) as resp:
          return await resp.text()
  async def fetch_all(urls):
      async with aiohttp.ClientSession() as session:
          tasks = [fetch_url(session, url) for url in urls]
          return await asyncio.gather(*tasks, return_exceptions=True)
  results = asyncio.run(fetch_all(urls))

LOGGING:
- Use simple print() statements for all output
- Print progress: "Fetching 10 URLs in parallel..."
- Print results: "Successfully scraped 8/10 URLs"
- Print summary: "Dataset updated: +15 new items, 45 total"

DATASET PATTERN:
- Filepath: /data/${sessionId}.json
- import json, os
- dataset = json.load(open(filepath)) if os.path.exists(filepath) else []
- print(f"Loaded existing dataset: {len(dataset)} items")
- Build new_items in parallel, extend dataset, json.dump(..., indent=2)
- print(f"Dataset updated: +{len(new_items)} new items, {len(dataset)} total")
- Immediately scrape any URLs returned by Tavily before issuing new searches

PDF HANDLING:
- Detect PDFs by URL or response headers
- Use pdfplumber (preferred) or PyPDF2
- Example:
  import pdfplumber
  with pdfplumber.open("local.pdf") as pdf:
      print(f"PDF loaded: {len(pdf.pages)} pages")
      for page in pdf.pages:
          text = page.extract_text() or ""
          # parse text into structured rows

ERROR REPORTING:
- Use try/except and print error messages
- Always finish with dataset stats so agent can decide next steps`,
    parameters: z.object({
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
    parameters: z.object({
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
  let namespace: string | null = null;
  let source: string | null = null;

  try {
    const body = await req.json();
    messages = body.messages || [];
    schema = body.schema;
    sessionId = body.sessionId;
    namespace = req.headers.get("x-namespace");
    source = req.headers.get("x-source");

    // Log message history for debugging
    console.log(`[Scraper] Request for session ${sessionId}`);
    console.log(`[Scraper] Message count: ${messages.length}`);
    
    // Clean up messages: remove tool invocations in "call" state without results
    // The AI SDK requires that tool invocations either have results or be properly formatted
    const cleanedMessages = messages.map((message) => {
      if (!message.toolInvocations || message.toolInvocations.length === 0) {
        return message;
      }

      // Filter out incomplete tool invocations (state: "call" without result)
      const validToolInvocations = message.toolInvocations.filter(
        (inv) => inv.state !== "call" || inv.result !== undefined
      );

      // If all tool invocations were filtered out, remove the toolInvocations field
      if (validToolInvocations.length === 0) {
        const { toolInvocations, ...rest } = message;
        return rest;
      }

      return {
        ...message,
        toolInvocations: validToolInvocations,
      };
    });
    
    const toolCallMessages = cleanedMessages.filter(
      (m) => Array.isArray(m.toolInvocations) && m.toolInvocations.length > 0
    );
    const totalToolCalls = toolCallMessages.reduce(
      (sum, m) => sum + (Array.isArray(m.toolInvocations) ? m.toolInvocations.length : 0),
      0
    );
    console.log(
      `[Scraper] Messages with tool calls: ${toolCallMessages.length}, Total tool calls: ${totalToolCalls}`
    );

    // Log which tools were called previously
    if (totalToolCalls > 0) {
      const toolNames = toolCallMessages.flatMap(
        (m) => (Array.isArray(m.toolInvocations) ? m.toolInvocations.map((inv) => inv.toolName) : [])
      );
      console.log(
        `[Scraper] Previous tools used: ${Array.from(new Set(toolNames)).join(
          ", "
        )}`
      );
    }

    // Generate session ID if not provided (for backward compatibility)
    // Format: YYYYMMDD-{uuid} (e.g., 20250130-abc123-def456-...)
    if (!sessionId) {
      const now = new Date();
      const dateStr = now.toISOString().split("T")[0].replace(/-/g, ""); // YYYYMMDD
      const uuid = crypto.randomUUID();
      sessionId = `${dateStr}-${uuid}`;
    }

    // Build system prompt
    const systemPrompt = `You are DocScraper, an iterative web scraping agent focused on cost-effective data collection.

MISSION:
- Deliver a clean, structured dataset saved at /data/${sessionId}.json that satisfies the user's request.
- Your job is to SCRAPE RAW TEXT from articles, not to summarize or transform it.
- Example objectives: scrape travel blog posts about Japan (url, scraped_text, author), food blog posts about Lisbon restaurants (url, scraped_text, date), or adventure travel stories (url, scraped_text).

TOOLS & BUDGET:
- search_internet → Tavily API (BASIC SEARCH ONLY, 1 credit each). Returns up to 20 URLs per search. Total allowance: ${MAX_TAVILY_CREDITS} credits, ${MAX_SEARCHES_PER_SESSION} searches per session.
  * Use when you need fresh URL sources or different angles on the topic
  * Each search gives you 20 URLs, so you can gather many sources with just a few searches
  * AVOID crawl (3 credits) and map (2 credits) modes - they're expensive
- execute_code → Python sandbox with requests, httpx, BeautifulSoup, lxml, selenium, playwright, pandas, PyPDF2, pdfplumber, openpyxl, cloudscraper, aiohttp installed.
  * PRIMARY TOOL for scraping and URL discovery
  * ALWAYS fetch URLs in parallel using asyncio/aiohttp or concurrent.futures - NEVER loop serially
  * Extract links from scraped pages using BeautifulSoup instead of Tavily map/crawl
  * Parse pagination, sitemaps, RSS feeds, article lists to find more URLs
- read_json → (Optional) Inspect dataset if you need to check current state. Dataset is automatically loaded when you finish.

COST-EFFECTIVE WORKFLOW:
1. Use search_internet to get initial seed URLs (costs 1 credit, returns up to 20 URLs)
2. **IMMEDIATELY use execute_code to scrape ALL URLs returned by Tavily in parallel** - don't wait, scrape them right away!
3. In the same code execution: extract additional links from the scraped pages using BeautifulSoup
4. Continue with execute_code to scrape discovered URLs - no additional Tavily calls needed
5. Extract more links from: pagination buttons, "related posts", sitemaps, category pages, author pages
6. If results are sparse or you need a different angle, use search_internet again with a refined query
7. Continue iterating: scrape → extract more URLs → scrape more → until you have enough quality data

CRITICAL: After EVERY search_internet call, your VERY NEXT action must be execute_code to scrape those URLs!

ITERATIVE LOOP:
1. **REVIEW CONVERSATION HISTORY**: Check all previous tool calls - what URLs were searched, what code was executed, what data was collected
2. **CHECK WHAT'S ALREADY DONE**: Look at previous execute_code outputs to see which URLs were already scraped and what data is in the dataset
3. Plan next step: ALWAYS prefer execute_code for scraping and link discovery over new searches
4. **AVOID REDUNDANCY**: Don't re-search or re-scrape URLs that were already processed in previous tool calls
5. Use execute_code to scrape AND discover more URLs with parallel fetching
6. Summarize what was added to the dataset in your response, decide next action
7. Use read_json only if you need to inspect the current dataset state

LOGGING STANDARD:
- Use simple print() statements for all logging
- Print progress updates: "Fetching 10 URLs in parallel..."
- Print results: "Successfully scraped 8/10 URLs"
- Print dataset summary at end: "Dataset updated: +15 new items, 45 total, 12 new URLs discovered"

URL DISCOVERY PATTERNS (in execute_code):
- Parse <a href> tags for links to similar content
- Check for pagination: next page, page numbers, "load more"
- Look for sitemaps: /sitemap.xml, /sitemap_index.xml
- Parse RSS/Atom feeds if available
- Extract links from article lists, category pages, tag pages, author pages
- Follow "related articles" or "you might also like" sections

PDF & BINARY CONTENT:
- Detect PDFs, download, parse with pdfplumber or PyPDF2
- Log page counts and extraction quality

DATASET QUALITY GATES:
- Every record MUST include: url + scraped_text (raw article content)
- SCRAPE RAW TEXT - extract the full article/blog post content as-is, don't summarize or transform
- Keep schemas simple: typically just url, scraped_text, and maybe 1-2 metadata fields (author, date)
- De-duplicate on url, aim for ~30+ high-quality rows unless user specifies otherwise
- Load → append → write pattern; never overwrite

SEARCH DISCIPLINE:
- Budget: ${MAX_TAVILY_CREDITS} credits, ${MAX_SEARCHES_PER_SESSION} searches max
- Track usage carefully - each search costs 1 credit
- Make searches count: use specific, targeted queries
- Prioritize execute_code for URL expansion over new searches

COMMUNICATION:
- Narrate decisions clearly
- After each tool call: explain what happened, what data exists, next planned action
- Emphasize how you're discovering URLs via scraping vs expensive Tavily calls
- When user sends a follow-up message: First summarize what you've already done (URLs searched, pages scraped, data collected) before proceeding

SCHEMA:
${
  schema
    ? `Match the provided schema exactly:
${schema}
ALWAYS include url and scraped_text fields (the raw article content).`
    : `Keep it simple: url, scraped_text (raw article content), and optionally 1-2 metadata fields like author or date.`
}`;

    // Use Azure OpenAI - apiVersion defaults to 'v1'
    const azure = createAzure({
      apiKey: process.env.AZURE_API_KEY!,
      resourceName: process.env.AZURE_RESOURCE_NAME,
    });

    // Use GPT-5 as default model
    const modelName = process.env.SCRAPER_AZURE_DEPLOYMENT_NAME || "gpt-4.1";

    const result = await streamText({
      model: azure(modelName),
      system: systemPrompt,
      messages: cleanedMessages.filter((m) => m.role !== "system") as CoreMessage[], // Cast to AI SDK message type
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
      maxSteps: 50,
      onFinish: async ({ usage }) => {
        if (!supabase) return;

        const cost =
          (usage.promptTokens * 2.5) / 1_000_000 +
          (usage.completionTokens * 10) / 1_000_000;

        try {
          const { error } = await supabase.from("frontend_ai_requests").insert({
            messages: cleanedMessages,
            namespace: namespace || sessionId, // Use sessionId as namespace if not provided
            cost,
            source: source || "scraper",
          });

          if (error) {
            console.error("Supabase insert error:", error);
          } else {
            console.log("Successfully logged scraper request to Supabase");
          }
        } catch (err) {
          console.error("Failed to log to Supabase:", err);
        }
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
