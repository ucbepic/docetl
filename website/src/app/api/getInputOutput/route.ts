import { NextResponse } from "next/server";
import { generatePipelineConfig } from "@/app/api/utils";
import os from "os";

const FASTAPI_URL = `${
  process.env.NEXT_PUBLIC_BACKEND_HTTPS ? "https" : "http"
}://${process.env.NEXT_PUBLIC_BACKEND_HOST}:${
  process.env.NEXT_PUBLIC_BACKEND_PORT
}`;

export async function POST(request: Request) {
  try {
    const {
      default_model,
      data,
      operations,
      operation_id,
      name,
      sample_size,
      namespace,
    } = await request.json();

    if (!name) {
      return NextResponse.json(
        { error: "Pipeline name is required" },
        { status: 400 }
      );
    }

    if (!data) {
      return NextResponse.json(
        { error: "Data is required. Please select a file in the sidebar." },
        { status: 400 }
      );
    }

    const homeDir = process.env.DOCETL_HOME_DIR || os.homedir();
    const { inputPath, outputPath } = generatePipelineConfig(
      namespace,
      default_model,
      data,
      operations,
      operation_id,
      name,
      homeDir,
      sample_size,
      false,
      false,
      { datasetDescription: null, persona: null },
      [],
      "",
      false
    );

    // Check if files exist using FastAPI endpoints
    const checkInputResponse = await fetch(
      `${FASTAPI_URL}/fs/check-file?path=${encodeURIComponent(inputPath)}`,
      {
        method: "GET",
      }
    );

    if (!checkInputResponse.ok) {
      console.error(`Failed to check input path: ${inputPath}`);
      return NextResponse.json(
        { error: "Failed to check input path" },
        { status: 500 }
      );
    }

    const inputResult = await checkInputResponse.json();
    if (!inputResult.exists) {
      console.error(`Input path does not exist: ${inputPath}`);
      return NextResponse.json(
        { error: "Input path does not exist" },
        { status: 400 }
      );
    }

    const checkOutputResponse = await fetch(
      `${FASTAPI_URL}/fs/check-file?path=${encodeURIComponent(outputPath)}`,
      {
        method: "GET",
      }
    );

    if (!checkOutputResponse.ok) {
      console.error(`Failed to check output path: ${outputPath}`);
      return NextResponse.json(
        { error: "Failed to check output path" },
        { status: 500 }
      );
    }

    const outputResult = await checkOutputResponse.json();
    if (!outputResult.exists) {
      console.error(`Output path does not exist: ${outputPath}`);
      return NextResponse.json(
        { error: "Output path does not exist" },
        { status: 400 }
      );
    }

    return NextResponse.json({ inputPath, outputPath });
  } catch (error) {
    console.error(error);
    return NextResponse.json(
      { error: "Failed to get input and output paths" },
      { status: 500 }
    );
  }
}
