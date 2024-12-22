import { NextResponse } from "next/server";

const FASTAPI_URL = `${
  process.env.NEXT_PUBLIC_BACKEND_HTTPS ? "https" : "http"
}://${process.env.NEXT_PUBLIC_BACKEND_HOST}:${
  process.env.NEXT_PUBLIC_BACKEND_PORT
}`;

export async function POST(request: Request) {
  try {
    const { namespace } = await request.json();

    if (!namespace) {
      return NextResponse.json(
        { error: "Namespace parameter is required" },
        { status: 400 }
      );
    }

    console.log(FASTAPI_URL);
    const response = await fetch(
      `${FASTAPI_URL}/fs/check-namespace?namespace=${namespace}`,
      {
        method: "POST",
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to check namespace");
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error checking namespace:", error);
    return NextResponse.json(
      { error: "Failed to check namespace" },
      { status: 500 }
    );
  }
}
