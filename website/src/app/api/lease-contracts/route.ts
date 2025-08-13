export const dynamic = "force-dynamic";

import { NextResponse } from "next/server";

export async function GET() {
  try {
    const response = await fetch(
      "https://docetlcloudbank.blob.core.windows.net/demos/red_flag_analysis.json",
      {
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
      }
    );

    if (!response.ok) {
      console.error(
        `Azure blob fetch failed: ${response.status} ${response.statusText}`
      );
      throw new Error(`Failed to fetch data: ${response.statusText}`);
    }

    const contracts = await response.json();

    return NextResponse.json(contracts, {
      headers: { "Cache-Control": "public, s-maxage=86400" },
    });
  } catch (error) {
    console.error("Error fetching lease contracts:", error);
    return NextResponse.json(
      { error: "Failed to fetch lease contracts" },
      { status: 500 }
    );
  }
}
