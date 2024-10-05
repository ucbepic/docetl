export type File = {
    name: string;
    content: string;
};

export type Operation = {
    id: string;
    llmType: 'LLM' | 'non-LLM';
    type: 'map' | 'reduce' | 'filter' | 'equijoin' | 'resolve' | 'parallel-map' | 'unnest' | 'split' | 'gather';
    name: string;
    prompt?: string;
    outputSchema?: Record<string, string>;
    validate?: string[];
    otherKwargs?: Record<string, any>;
};

export type OutputRow = Record<string, string>;

export type SchemaType = 'string' | 'float' | 'int' | 'boolean' | 'list' | 'dict';

export interface SchemaItem {
  key: string;
  type: SchemaType;
  subType?: SchemaItem | SchemaItem[];
}

export interface UserNote {
    id: string;
    note: string;
}

export interface Bookmark {
    id: string;
    text: string;
    source: string;
    color: string;
    notes: UserNote[];
}
  
export interface BookmarkContextType {
    bookmarks: Bookmark[];
    addBookmark: (text: string, source: string, color: string, notes: UserNote[]) => void;
    removeBookmark: (id: string) => void;
}
  