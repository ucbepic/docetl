// app/api/documents/[...path]/route.ts
import { NextRequest, NextResponse } from "next/server";

const FASTAPI_URL = `${
  process.env.NEXT_PUBLIC_BACKEND_HTTPS ? "https" : "http"
}://${process.env.NEXT_PUBLIC_BACKEND_HOST}:${
  process.env.NEXT_PUBLIC_BACKEND_PORT
}`;

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    // Join the path segments and decode any URL encoding
    const filePath = decodeURIComponent(params.path.join("/"));

    // Forward the request to FastAPI's serve-document endpoint
    const response = await fetch(
      `${FASTAPI_URL}/fs/serve-document/${filePath}`,
      {
        method: "GET",
      }
    );

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { error: error.detail },
        { status: response.status }
      );
    }

    // Stream the response from FastAPI
    const data = await response.blob();
    return new NextResponse(data, {
      headers: {
        "Content-Type":
          response.headers.get("Content-Type") || "application/octet-stream",
        "Content-Disposition":
          response.headers.get("Content-Disposition") || "inline",
        "Cache-Control":
          response.headers.get("Cache-Control") || "public, max-age=3600",
      },
    });
  } catch (error) {
    console.error("Error serving file:", error);
    return NextResponse.json(
      { error: "Failed to serve file" },
      { status: 500 }
    );
  }
}
