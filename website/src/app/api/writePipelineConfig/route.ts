import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";
import os from "os";
import { generatePipelineConfig } from "@/app/api/utils";

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
      system_prompt
    );

    // Save the YAML file in the user's home directory
    const pipelineDir = path.join(homeDir, ".docetl", namespace, "pipelines");
    const configDir = path.join(pipelineDir, "configs");
    const nameDir = path.join(pipelineDir, name, "intermediates");

    try {
      await fs.mkdir(configDir, { recursive: true });
      await fs.mkdir(nameDir, { recursive: true });
      const filePath = path.join(configDir, `${name}.yaml`);
      await fs.writeFile(filePath, yamlString, "utf8");

      return NextResponse.json({ filePath, inputPath, outputPath });
    } catch (fsError) {
      console.error("File system error:", fsError);
      return NextResponse.json(
        `Failed to write pipeline configuration: ${fsError.message}`,
        { status: 500 }
      );
    }
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
