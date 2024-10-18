import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import { generatePipelineConfig } from '@/app/api/utils';

export async function POST(request: Request) {
  try {
    const { default_model, data, operations, operation_id, name, sample_size } = await request.json();

    if (!name) {
      return NextResponse.json({ error: 'Pipeline name is required' }, { status: 400 });
    }

    if (!data) {
      return NextResponse.json({ error: 'Data is required. Please select a file in the sidebar.' }, { status: 400 });
    }

    const { yamlString, inputPath, outputPath } = generatePipelineConfig(
      default_model,
      data,
      operations,
      operation_id,
      name,
      sample_size
    );

    console.log(yamlString);

    // Save the YAML file in the user's home directory
    const homeDir = os.homedir();
    const pipelineDir = path.join(homeDir, '.docetl', 'pipelines', 'configs');
    await fs.mkdir(pipelineDir, { recursive: true });
    const filePath = path.join(pipelineDir, `${name}.yaml`);
    await fs.writeFile(filePath, yamlString, 'utf8');

    return NextResponse.json({ filePath, inputPath, outputPath });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to create pipeline config' }, { status: 500 });
  }
}