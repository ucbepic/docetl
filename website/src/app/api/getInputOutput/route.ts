import { NextResponse } from "next/server";
import { generatePipelineConfig } from "@/app/api/utils";
import fs from "fs/promises";
import os from "os";
export async function POST(request: Request) {
  try {
    const { default_model, data, operations, operation_id, name, sample_size } =
      await request.json();

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
      default_model,
      data,
      operations,
      operation_id,
      name,
      homeDir,
      sample_size
    );

    // Check if inputPath exists
    try {
      await fs.access(inputPath);
    } catch (error) {
      console.error(`Input path does not exist: ${inputPath}`);
      return NextResponse.json(
        { error: "Input path does not exist" },
        { status: 400 }
      );
    }

    // Check if outputPath exists
    try {
      await fs.access(outputPath);
    } catch (error) {
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
