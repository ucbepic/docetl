export const dynamic = "force-dynamic";

import { NextResponse } from "next/server";

export async function GET() {
  try {
    const response = await fetch(
      "https://docetlcloudbank.blob.core.windows.net/demos/emails_with_metadata.json",
      {
        headers: { "Content-Type": "application/json" },
        // Cache for 1 hour on the server side
        next: { revalidate: 3600 }
      }
    );

    if (!response.ok) {
      console.error(
        `Azure blob fetch failed: ${response.status} ${response.statusText}`
      );
      throw new Error(`Failed to fetch data: ${response.statusText}`);
    }

    const emails = await response.json();

    // Return all emails with CORS headers
    return NextResponse.json(emails, {
      headers: {
        "Cache-Control": "public, s-maxage=3600, max-age=3600",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET",
      },
    });
  } catch (error) {
    console.error("Error fetching Epstein emails:", error);
    return NextResponse.json(
      { error: "Failed to fetch Epstein emails", details: String(error) },
      { status: 500 }
    );
  }
}
