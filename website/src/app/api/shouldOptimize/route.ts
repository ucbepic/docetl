import { NextRequest, NextResponse } from "next/server";

const FASTAPI_URL = `${
  process.env.NEXT_PUBLIC_BACKEND_HTTPS ? "https" : "http"
}://${process.env.NEXT_PUBLIC_BACKEND_HOST}:${
  process.env.NEXT_PUBLIC_BACKEND_PORT
}`;

// Helper to handle errors consistently
const handleError = (error: unknown, status = 500) => {
  const message =
    error instanceof Error ? error.message : "Internal server error";
  return NextResponse.json({ error: message }, { status });
};

// Helper to proxy requests to FastAPI
async function proxyRequest(path: string, init?: RequestInit) {
  const response = await fetch(`${FASTAPI_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`FastAPI server error: ${error}`);
  }

  return response.json();
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    // Extract task ID from the URL if it exists
    const taskId = request.nextUrl.searchParams.get("taskId");
    const isCancel = request.nextUrl.searchParams.get("cancel") === "true";

    // Handle different POST scenarios
    if (taskId) {
      if (isCancel) {
        // Cancel task
        const data = await proxyRequest(`/should_optimize/${taskId}/cancel`, {
          method: "POST",
        });
        return NextResponse.json(data);
      }
      // Invalid request with taskId but no cancel
      return handleError(new Error("Invalid request"), 400);
    }

    // Submit new task
    const body = await request.json();
    const data = await proxyRequest("/should_optimize", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return NextResponse.json(data, { status: 202 });
  } catch (error) {
    return handleError(error);
  }
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  try {
    // Extract task ID from the URL
    const taskId = request.nextUrl.searchParams.get("taskId");

    if (!taskId) {
      return handleError(new Error("Task ID is required"), 400);
    }

    const data = await proxyRequest(`/should_optimize/${taskId}`);
    return NextResponse.json(data);
  } catch (error) {
    return handleError(error);
  }
}
