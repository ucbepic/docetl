export const dynamic = "force-dynamic";

import { NextResponse } from "next/server";

export async function GET(request: Request) {
  try {
    // Get URL parameters
    const { searchParams } = new URL(request.url);
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "20");
    const filterFamous = searchParams.get("famous"); // 'true', 'false', or null
    const filterConcrete = searchParams.get("concrete"); // 'true', 'false', or null

    // Fetch data from Azure blob storage with better error handling
    const response = await fetch(
      "https://docetl.blob.core.windows.net/demos/summarized_responses.json",
      {
        headers: {
          "Content-Type": "application/json",
        },
        cache: "no-store",
      }
    );

    if (!response.ok) {
      console.error(
        `Azure blob fetch failed: ${response.status} ${response.statusText}`
      );
      throw new Error(`Failed to fetch data: ${response.statusText}`);
    }

    const allResponses = await response.json();

    // Apply filters
    const filteredData = allResponses.filter((response: any) => {
      const famousMatch =
        filterFamous === null ||
        String(response.from_famous_entity) === filterFamous;
      const concreteMatch =
        filterConcrete === null ||
        String(response.concrete_proposal_described) === filterConcrete;
      return famousMatch && concreteMatch;
    });

    // Paginate
    const startIndex = (page - 1) * limit;
    const paginatedData = filteredData.slice(startIndex, startIndex + limit);

    // If no query parameters are provided, return all data without pagination
    if (
      !searchParams.has("page") &&
      !searchParams.has("limit") &&
      !searchParams.has("famous") &&
      !searchParams.has("concrete")
    ) {
      return NextResponse.json(allResponses, {
        headers: {
          "Cache-Control": "public, s-maxage=86400",
        },
      });
    }

    // Return with metadata
    return NextResponse.json(
      {
        data: paginatedData,
        meta: {
          total: filteredData.length,
          page,
          limit,
          pages: Math.ceil(filteredData.length / limit),
        },
      },
      {
        headers: {
          "Cache-Control": "public, s-maxage=86400", // Cache for 1 day
        },
      }
    );
  } catch (error) {
    console.error("Error fetching RFI responses:", error);
    return NextResponse.json(
      { error: "Failed to fetch RFI responses" },
      { status: 500 }
    );
  }
}
