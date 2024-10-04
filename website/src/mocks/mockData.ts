import { File, Operation } from '@/app/types';

export const mockFiles: File[] = [
  { name: 'debate_transcripts.json', content: '' },
];

export const initialOperations: Operation[] = [
  { id: '1', llmType: 'LLM', type: 'map', name: 'Text Generation', prompt: '', outputSchema: {  } },
];