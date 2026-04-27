import { NextResponse } from "next/server";

const FASTAPI_URL = `${
  process.env.NEXT_PUBLIC_BACKEND_HTTPS ? "https" : "http"
}://${process.env.NEXT_PUBLIC_BACKEND_HOST}:${
  process.env.NEXT_PUBLIC_BACKEND_PORT
}`;

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const workspaceId = searchParams.get("id");
    if (!workspaceId) {
      return NextResponse.json({ error: "id parameter required" }, { status: 400 });
    }
    const response = await fetch(`${FASTAPI_URL}/fs/workspace/${workspaceId}`);
    if (response.status === 404) {
      return NextResponse.json({ exists: false }, { status: 404 });
    }
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to load workspace");
    }
    const data = await response.json();
    return NextResponse.json({ exists: true, content: data.content });
  } catch (error) {
    console.error("Error loading workspace:", error);
    return NextResponse.json({ error: "Failed to load workspace" }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const workspaceId = searchParams.get("id");
    if (!workspaceId) {
      return NextResponse.json({ error: "id parameter required" }, { status: 400 });
    }
    const body = await request.json();
    const response = await fetch(`${FASTAPI_URL}/fs/workspace/${workspaceId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: body.content }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to save workspace");
    }
    return NextResponse.json({ ok: true });
  } catch (error) {
    console.error("Error saving workspace:", error);
    return NextResponse.json({ error: "Failed to save workspace" }, { status: 500 });
  }
}
