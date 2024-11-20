import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const files = formData.getAll("files");

    if (!files || files.length === 0) {
      return NextResponse.json({ error: "No files provided" }, { status: 400 });
    }

    // Create a new FormData to forward to the Python backend
    const backendFormData = new FormData();
    files.forEach((file) => {
      backendFormData.append("files", file);
    });

    // Get Azure credentials from headers if they exist
    const azureEndpoint = request.headers.get("azure-endpoint");
    const azureKey = request.headers.get("azure-key");

    // Determine which endpoint to use
    const endpoint =
      azureEndpoint && azureKey
        ? "/api/azure-convert-documents"
        : "/api/convert-documents";

    // Prepare headers for the backend request
    const headers: HeadersInit = {};
    if (azureEndpoint && azureKey) {
      headers["azure-endpoint"] = azureEndpoint;
      headers["azure-key"] = azureKey;
    }

    // Forward the request to the Python backend
    const response = await fetch(
      `http://${process.env.NEXT_PUBLIC_BACKEND_HOST}:${process.env.NEXT_PUBLIC_BACKEND_PORT}${endpoint}`,
      {
        method: "POST",
        body: backendFormData,
        headers,
      }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `Backend returned ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      documents: data.documents,
      message: "Documents converted successfully",
    });
  } catch (error) {
    console.error("Error converting documents:", error);
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Failed to convert documents",
      },
      { status: 500 }
    );
  }
}
