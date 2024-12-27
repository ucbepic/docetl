import { NextResponse } from "next/server";
import { generatePipelineConfig } from "@/app/api/utils";
import os from "os";
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
      system_prompt,
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

    const { yamlString } = generatePipelineConfig(
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
      system_prompt,
      [],
      "",
      false
    );

    return NextResponse.json({ pipelineConfig: yamlString });
  } catch (error) {
    console.error(error);
    return NextResponse.json(
      { error: "Failed to generate pipeline configuration" },
      { status: 500 }
    );
  }
}
