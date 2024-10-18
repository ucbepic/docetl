import { NextResponse } from 'next/server';
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

    const { yamlString } = generatePipelineConfig(
      default_model,
      data,
      operations,
      operation_id,
      name,
      sample_size
    );

    return NextResponse.json({ pipelineConfig: yamlString });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to generate pipeline configuration' }, { status: 500 });
  }
}
