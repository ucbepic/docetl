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
      optimize = false,
      clear_intermediate = false,
      system_prompt,
      namespace,
      apiKeys,
    } = await request.json();

    if (!name) {
      return NextResponse.json(
        { error: "Pipeline name is required" },
        { status: 400 }
      );
    }

    if (!data?.path) {
      return NextResponse.json(
        { error: "Data is required. Please select a file in the sidebar." },
        { status: 400 }
      );
    }

    const homeDir = process.env.DOCETL_HOME_DIR || os.homedir();
    const docetl_encryption_key = process.env.DOCETL_ENCRYPTION_KEY || "";

    const { yamlString, inputPath, outputPath } = generatePipelineConfig(
      namespace,
      default_model,
      data,
      operations,
      operation_id,
      name,
      homeDir,
      sample_size,
      optimize,
      clear_intermediate,
      system_prompt,
      apiKeys,
      docetl_encryption_key,
      true
    );

    // Use the FastAPI endpoint to write the pipeline config
    const response = await fetch(`${FASTAPI_URL}/fs/write-pipeline-config`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        namespace,
        name,
        config: yamlString,
        input_path: inputPath,
        output_path: outputPath,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to write pipeline configuration: ${error}`);
    }

    const result = await response.json();
    return NextResponse.json({
      filePath: result.filePath,
      inputPath: result.inputPath,
      outputPath: result.outputPath,
    });
  } catch (error) {
    console.error("Pipeline configuration error:", error);
    return NextResponse.json(
      error instanceof Error
        ? error.message
        : "An unexpected error occurred while creating the pipeline configuration",
      { status: 500 }
    );
  }
}
