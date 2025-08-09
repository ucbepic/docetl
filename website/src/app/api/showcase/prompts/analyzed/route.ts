import { NextResponse } from "next/server";

const AZURE_ANALYZED_URL =
  "https://docetlcloudbank.blob.core.windows.net/demos/analyzed_strategies.json";

export async function GET() {
  try {
    const res = await fetch(AZURE_ANALYZED_URL, {
      // Revalidate periodically to keep the data relatively fresh without hammering the origin
      next: { revalidate: 60 * 60 },
    });

    if (!res.ok) {
      return NextResponse.json(
        { error: `Upstream fetch failed with status ${res.status}` },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data, {
      headers: {
        // Allow same-origin requests from the app; avoid open proxy behavior
        "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=600",
      },
    });
  } catch (err) {
    return NextResponse.json(
      { error: "Failed to fetch analyzed strategies" },
      { status: 500 }
    );
  }
}
