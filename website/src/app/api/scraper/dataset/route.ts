import { ModalClient } from "modal";

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

    modalApp = await modalClient.apps.fromName("docetl-scraper", {
      createIfMissing: true,
    });

    modalVolume = await modalClient.volumes.fromName("scraper-data", {
      createIfMissing: true,
    });
  } catch (error) {
    console.error("Failed to initialize Modal:", error);
  }
}

async function readModalFile(sessionId: string): Promise<{
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
    // Per Modal docs, volume data persists after writer sandbox terminates
    // Just create a new sandbox to read the data
    const image = modalClient.images.fromRegistry("python:3.13-slim");
    const sb = await modalClient.sandboxes.create(modalApp, image, {
      volumes: { "/data": modalVolume },
      timeoutMs: 60 * 1000,
    });
    console.log(`[Volume] Created reader sandbox for session ${sessionId}`);

    try {
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

      const exitCode = await process.wait();
      const stdout = await process.stdout.readText();
      const stderr = await process.stderr.readText();

      await sb.terminate();

      console.log(
        `[Dataset Read] SessionId: ${sessionId}, ExitCode: ${exitCode}`
      );
      console.log(`[Dataset Read] Stdout length: ${stdout.length}`);
      if (stderr) console.log(`[Dataset Read] Stderr: ${stderr}`);

      if (exitCode !== 0) {
        return {
          success: false,
          data: null,
          error: stderr || `Process exited with code ${exitCode}`,
        };
      }

      const data = JSON.parse(stdout || "[]");
      console.log(
        `[Dataset Read] Parsed ${Array.isArray(data) ? data.length : 0} items`
      );
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

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const sessionId = searchParams.get("sessionId");

    if (!sessionId) {
      return new Response(
        JSON.stringify({ error: "sessionId parameter is required" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const result = await readModalFile(sessionId);

    if (!result.success) {
      return new Response(JSON.stringify({ error: result.error, data: [] }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify({ data: result.data }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Dataset fetch error:", error);
    return new Response(
      JSON.stringify({
        error:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred",
        data: [],
      }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
