export type File = {
  name: string;
  path: string;
  type: "json" | "document" | "pipeline-yaml";
  parentFolder?: string;
  blob?: Blob;
};

export type Operation = {
  id: string;
  llmType: "LLM" | "non-LLM";
  type:
    | "map"
    | "reduce"
    | "filter"
    | "resolve"
    | "parallel_map"
    | "unnest"
    | "split"
    | "gather"
    | "sample"
    | "code_map"
    | "code_reduce"
    | "code_filter";
  name: string;
  prompt?: string;
  output?: { schema: SchemaItem[] };
  validate?: string[];
  gleaning?: { num_rounds: number; validation_prompt: string };
  otherKwargs?: Record<string, any>;
  runIndex?: number;
  sample?: number;
  shouldOptimizeResult?: string;
  visibility: boolean;
};

export type OutputRow = Record<string, string>;

export type SchemaType =
  | "string"
  | "float"
  | "int"
  | "boolean"
  | "list"
  | "dict";

export interface SchemaItem {
  key: string;
  type: SchemaType;
  subType?: SchemaItem | SchemaItem[];
}

export interface UserNote {
  id: string;
  note: string;
  metadata: {
    columnId?: string;
    rowIndex?: number;
    mainColumnValue?: unknown;
    rowContent?: Record<string, unknown>;
    operationName?: string;
  };
}

export interface Bookmark {
  id: string;
  color: string;
  notes: UserNote[];
}

export interface BookmarkContextType {
  bookmarks: Bookmark[];
  addBookmark: (color: string, notes: UserNote[]) => void;
  removeBookmark: (id: string) => void;
  getNotesForRowAndColumn: (rowIndex: number, columnId: string) => UserNote[];
}

export interface OutputType {
  path: string;
  operationId: string;
  inputPath?: string;
}

export interface OptimizeRequest {
  yaml_config: string;
  step_name: string;
  op_name: string;
}

export type TaskStatus =
  | "pending"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled";

export interface OptimizeResult {
  task_id: string;
  status: TaskStatus;
  should_optimize?: string;
  input_data?: Array<Record<string, unknown>>;
  output_data?: Array<Record<string, unknown>>;
  cost?: number;
  error?: string;
  created_at: string;
  completed_at?: string;
}

export interface APIKey {
  name: string;
  value: string;
}
