import { NextRequest, NextResponse } from "next/server";

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

    // Forward the request to the Python backend
    const response = await fetch(
      `http://${process.env.NEXT_PUBLIC_BACKEND_HOST}:${process.env.NEXT_PUBLIC_BACKEND_PORT}/api/convert-documents`,
      {
        method: "POST",
        body: backendFormData,
      }
    );

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      documents: data.documents,
      message: "Documents converted successfully",
    });
  } catch (error) {
    console.error("Error converting documents:", error);
    return NextResponse.json(
      { error: "Failed to convert documents" },
      { status: 500 }
    );
  }
}

// Increase the maximum request size limit for file uploads
export const config = {
  api: {
    bodyParser: false,
  },
};
