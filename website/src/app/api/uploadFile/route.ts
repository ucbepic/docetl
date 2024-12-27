import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const namespace = formData.get("namespace") as string;
    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // Construct FastAPI URL from environment variables
    const FASTAPI_URL = `${
      process.env.NEXT_PUBLIC_BACKEND_HTTPS ? "https" : "http"
    }://${process.env.NEXT_PUBLIC_BACKEND_HOST}:${
      process.env.NEXT_PUBLIC_BACKEND_PORT
    }`;
    const apiFormData = new FormData();
    apiFormData.append("file", file);
    apiFormData.append("namespace", namespace);

    const response = await fetch(`${FASTAPI_URL}/fs/upload-file`, {
      method: "POST",
      body: apiFormData,
    });

    if (!response.ok) {
      throw new Error(`API responded with status ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json({ path: data.path });
  } catch (error) {
    console.error("Error uploading file:", error);
    return NextResponse.json(
      { error: "Failed to upload file" },
      { status: 500 }
    );
  }
}
