import { NextResponse } from "next/server";

export async function GET() {
  try {
    // Fetch from the blob storage URL
    const response = await fetch(
      "https://docetlcloudbank.blob.core.windows.net/demos/health_quotes.json"
    );
    
    if (!response.ok) {
      throw new Error("Failed to fetch data from remote source");
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error loading health quotes data:", error);
    return NextResponse.json(
      { error: "Failed to load health quotes data" },
      { status: 500 }
    );
  }
}