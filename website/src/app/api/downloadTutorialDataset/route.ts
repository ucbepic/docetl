import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const fileId = searchParams.get("fileId");

  if (!fileId) {
    return new NextResponse("File ID is required", { status: 400 });
  }

  try {
    const driveUrl = `https://drive.google.com/uc?export=download&id=${fileId}`;
    const response = await fetch(driveUrl);

    if (!response.ok) {
      throw new Error("Failed to download file from Google Drive");
    }

    const data = await response.blob();
    return new NextResponse(data);
  } catch (error) {
    console.error("Error downloading tutorial dataset:", error);
    return new NextResponse("Failed to download tutorial dataset", {
      status: 500,
    });
  }
}
