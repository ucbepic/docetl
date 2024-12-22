import { NextRequest, NextResponse } from "next/server";

const FASTAPI_URL = `${
  process.env.NEXT_PUBLIC_BACKEND_HTTPS ? "https" : "http"
}://${process.env.NEXT_PUBLIC_BACKEND_HOST}:${
  process.env.NEXT_PUBLIC_BACKEND_PORT
}`;

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const files = formData.getAll("files") as File[];
    const namespace = formData.get("namespace") as string;

    if (!files || files.length === 0) {
      return NextResponse.json({ error: "No files provided" }, { status: 400 });
    }

    // Create a new FormData object to send to the backend
    const backendFormData = new FormData();
    files.forEach((file) => {
      backendFormData.append("files", file);
    });
    backendFormData.append("namespace", namespace);

    // Send to FastAPI backend
    const response = await fetch(`${FASTAPI_URL}/fs/save-documents`, {
      method: "POST",
      body: backendFormData,
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status ${response.status}`);
    }

    const result = await response.json();
    return NextResponse.json(result, { status: 200 });
  } catch (error) {
    console.error("Error saving documents:", error);
    return NextResponse.json(
      { error: "Failed to save documents" },
      { status: 500 }
    );
  }
}
