import React, { useMemo, useEffect } from "react";
import { Operation, SchemaItem } from "@/app/types";
import { OutputSchema, PromptInput, CodeInput } from "./args";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "../ui/button";
import { Plus, X, Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "../ui/textarea";

interface PromptConfig {
  prompt: string;
  output_keys?: string[];
  model?: string;
}

interface MethodKwargs {
  delimiter?: string;
  num_tokens?: number;
  stratify_key?: string;
  embedding_keys?: string[];
  std?: number;
  keep?: boolean;
}

interface PeripheralChunkConfig {
  content_key?: string;
  count?: number;
}

interface PeripheralChunksSection {
  head?: PeripheralChunkConfig;
  middle?: PeripheralChunkConfig;
  tail?: PeripheralChunkConfig;
}

interface PeripheralChunks {
  previous?: PeripheralChunksSection;
  next?: PeripheralChunksSection;
}

interface OtherKwargs {
  prompts?: PromptConfig[];
  method?: string;
  method_kwargs?: MethodKwargs;
  reduce_key?: string[];
  comparison_prompt?: string;
  resolution_prompt?: string;
  blocking_threshold?: number;
  blocking_keys?: string[];
  split_key?: string;
  unnest_key?: string;
  recursive?: boolean;
  depth?: number;
  content_key?: string;
  doc_id_key?: string;
  order_key?: string;
  peripheral_chunks?: PeripheralChunks;
  samples?: string | number;
  code?: string;
}

interface OperationComponentProps {
  operation: Operation;
  isSchemaExpanded: boolean;
  onUpdate: (updatedOperation: Operation) => void;
  onToggleSchema: () => void;
}

export const MapFilterOperationComponent: React.FC<OperationComponentProps> = ({
  operation,
  isSchemaExpanded,
  onUpdate,
  onToggleSchema,
}) => {
  const schemaItems = useMemo(() => {
    return operation?.output?.schema || [];
  }, [operation?.output?.schema]);

  const handlePromptChange = (newPrompt: string) => {
    onUpdate({ ...operation, prompt: newPrompt });
  };

  const handleSchemaUpdate = (newSchema: SchemaItem[]) => {
    onUpdate({
      ...operation,
      output: {
        ...operation.output,
        schema: newSchema,
      },
    });
  };

  return (
    <>
      <PromptInput
        prompt={operation.prompt || ""}
        onChange={handlePromptChange}
      />
      <OutputSchema
        schema={schemaItems}
        onUpdate={handleSchemaUpdate}
        isExpanded={isSchemaExpanded}
        onToggle={onToggleSchema}
      />
    </>
  );
};

export const ReduceOperationComponent: React.FC<OperationComponentProps> = ({
  operation,
  isSchemaExpanded,
  onUpdate,
  onToggleSchema,
}) => {
  const schemaItems = useMemo(() => {
    return operation?.output?.schema || [];
  }, [operation?.output?.schema]);

  const handlePromptChange = (newPrompt: string) => {
    onUpdate({ ...operation, prompt: newPrompt });
  };

  const handleSchemaUpdate = (newSchema: SchemaItem[]) => {
    onUpdate({
      ...operation,
      output: {
        ...operation.output,
        schema: newSchema,
      },
    });
  };

  const handleReduceKeysChange = (newReduceKeys: string[]) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        reduce_key: newReduceKeys,
      },
    });
  };

  useEffect(() => {
    if (!operation.otherKwargs?.reduce_key) {
      handleReduceKeysChange(["_all"]);
    }
  }, []);

  return (
    <>
      <div className="mb-4">
        <div className="flex items-center space-x-2">
          <Label htmlFor="reduce-keys" className="w-1/4">
            Reduce Key(s)
          </Label>
          <div className="flex-grow flex items-center space-x-2 overflow-x-auto">
            <div className="flex-nowrap flex items-center space-x-2">
              {(operation.otherKwargs?.reduce_key || [""]).map(
                (key: string, index: number) => (
                  <div
                    key={index}
                    className="relative flex-shrink-0"
                    style={{ minWidth: "150px" }}
                  >
                    <div className="flex items-center">
                      <Input
                        id={`reduce-key-${index}`}
                        value={key}
                        onChange={(e) => {
                          const newKeys = [
                            ...(operation.otherKwargs?.reduce_key || [""]),
                          ];
                          newKeys[index] = e.target.value;
                          handleReduceKeysChange(newKeys);
                        }}
                        placeholder="Enter reduce key"
                        className={`w-full pr-8 ${
                          !key.trim() ? "border-red-500 focus:ring-red-500" : ""
                        }`}
                      />
                      {(operation.otherKwargs?.reduce_key?.length || 0) > 1 && (
                        <Button
                          onClick={() => {
                            const newKeys = [
                              ...(operation.otherKwargs?.reduce_key || [""]),
                            ];
                            newKeys.splice(index, 1);
                            handleReduceKeysChange(newKeys);
                          }}
                          size="sm"
                          variant="ghost"
                          className="absolute right-0 top-0 h-full"
                        >
                          <X size={12} />
                        </Button>
                      )}
                    </div>
                    {!key.trim() && (
                      <p className="text-red-500 text-sm mt-1">
                        There must be a reduce key. Set "_all" if you want all
                        documents in a group.
                      </p>
                    )}
                  </div>
                )
              )}
              <Button
                onClick={() => {
                  const newKeys = [
                    ...(operation.otherKwargs?.reduce_key || [""]),
                    "",
                  ];
                  handleReduceKeysChange(newKeys);
                }}
                size="sm"
                variant="outline"
                className="flex-shrink-0"
              >
                <Plus size={16} />
              </Button>
            </div>
          </div>
        </div>
      </div>
      <PromptInput
        prompt={operation.prompt || ""}
        onChange={handlePromptChange}
      />
      <OutputSchema
        schema={schemaItems}
        onUpdate={handleSchemaUpdate}
        isExpanded={isSchemaExpanded}
        onToggle={onToggleSchema}
      />
    </>
  );
};

export const ResolveOperationComponent: React.FC<OperationComponentProps> = ({
  operation,
  isSchemaExpanded,
  onUpdate,
  onToggleSchema,
}) => {
  const schemaItems = useMemo(() => {
    return operation?.output?.schema || [];
  }, [operation?.output?.schema]);

  const handleComparisonPromptChange = (newPrompt: string) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        comparison_prompt: newPrompt,
      },
    });
  };

  const handleResolutionPromptChange = (newPrompt: string) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        resolution_prompt: newPrompt,
      },
    });
  };

  const handleSchemaUpdate = (newSchema: SchemaItem[]) => {
    onUpdate({
      ...operation,
      output: {
        ...operation.output,
        schema: newSchema,
      },
    });
  };

  return (
    <>
      <div className="flex mb-4 space-x-4">
        <div className="flex-1">
          <label
            htmlFor="comparison-prompt"
            className="block text-sm font-medium text-gray-700"
          >
            Comparison Prompt
          </label>
          <PromptInput
            prompt={operation.otherKwargs?.comparison_prompt || ""}
            onChange={handleComparisonPromptChange}
          />
        </div>
        <div className="flex-1">
          <label
            htmlFor="resolution-prompt"
            className="block text-sm font-medium text-gray-700"
          >
            Resolution Prompt
          </label>
          <PromptInput
            prompt={operation.otherKwargs?.resolution_prompt || ""}
            onChange={handleResolutionPromptChange}
          />
        </div>
      </div>
      <div className="mb-4 flex items-end space-x-4">
        <div className="w-1/3">
          <div className="flex items-center">
            <label
              htmlFor="blocking-threshold"
              className="block text-sm font-medium text-gray-700"
            >
              Blocking Threshold
            </label>
            <TooltipProvider>
              <Tooltip delayDuration={300}>
                <TooltipTrigger asChild>
                  <div className="ml-2">
                    <Info className="h-4 w-4 text-gray-400 cursor-help" />
                  </div>
                </TooltipTrigger>
                <TooltipContent className="w-64 p-2 text-xs">
                  Unsure of what value to set here? Click the lightning button
                  to optimize this operation, which will automatically determine
                  the blocking threshold.
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <Input
            type="number"
            id="blocking-threshold"
            value={operation.otherKwargs?.blocking_threshold}
            onChange={(e) => {
              const value = parseFloat(e.target.value);
              if (!isNaN(value) && value >= 0 && value <= 1) {
                onUpdate({
                  ...operation,
                  otherKwargs: {
                    ...operation.otherKwargs,
                    blocking_threshold: value,
                  },
                });
              }
            }}
            step="0.01"
            min="0"
            max="1"
            className="mt-1"
          />
        </div>
      </div>
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Blocking Keys
        </label>
        <div className="flex flex-wrap gap-2">
          {(operation.otherKwargs?.blocking_keys || []).map(
            (key: string, index: number) => (
              <div key={index} className="flex items-center">
                <Input
                  value={key}
                  onChange={(e) => {
                    const newKeys = [
                      ...(operation.otherKwargs?.blocking_keys || []),
                    ];
                    newKeys[index] = e.target.value;
                    onUpdate({
                      ...operation,
                      otherKwargs: {
                        ...operation.otherKwargs,
                        blocking_keys: newKeys,
                      },
                    });
                  }}
                  placeholder="Enter blocking key"
                  className="w-40"
                />
                <Button
                  onClick={() => {
                    const newKeys = [
                      ...(operation.otherKwargs?.blocking_keys || []),
                    ];
                    newKeys.splice(index, 1);
                    onUpdate({
                      ...operation,
                      otherKwargs: {
                        ...operation.otherKwargs,
                        blocking_keys: newKeys,
                      },
                    });
                  }}
                  size="sm"
                  variant="ghost"
                >
                  <X size={12} />
                </Button>
              </div>
            )
          )}
          <Button
            onClick={() => {
              const newKeys = [
                ...(operation.otherKwargs?.blocking_keys || []),
                "",
              ];
              onUpdate({
                ...operation,
                otherKwargs: {
                  ...operation.otherKwargs,
                  blocking_keys: newKeys,
                },
              });
            }}
            size="sm"
            variant="outline"
          >
            <Plus size={16} />
          </Button>
        </div>
      </div>
      <OutputSchema
        schema={schemaItems}
        onUpdate={handleSchemaUpdate}
        isExpanded={isSchemaExpanded}
        onToggle={onToggleSchema}
      />
    </>
  );
};

export const SplitOperationComponent: React.FC<OperationComponentProps> = ({
  operation,
  isSchemaExpanded,
  onUpdate,
  onToggleSchema,
}) => {
  const handleSplitKeyChange = (newSplitKey: string) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        split_key: newSplitKey,
      },
    });
  };

  const handleMethodChange = (newMethod: string) => {
    const newMethodKwargs = { ...operation.otherKwargs?.method_kwargs };
    if (newMethod === "delimiter" && !newMethodKwargs.delimiter) {
      newMethodKwargs.delimiter = "";
    } else if (newMethod === "token_count" && !newMethodKwargs.num_tokens) {
      newMethodKwargs.num_tokens = 1;
    }

    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        method: newMethod,
        method_kwargs: newMethodKwargs,
      },
    });
  };

  const handleMethodKwargsChange = (key: string, value: string) => {
    let newValue: string | number = value;
    if (key === "num_tokens") {
      const numTokens = parseInt(value, 10);
      newValue = isNaN(numTokens) || numTokens <= 0 ? 1 : numTokens;
    }
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        method_kwargs: {
          ...operation.otherKwargs?.method_kwargs,
          [key]: newValue,
        },
      },
    });
  };

  const addMethodKwarg = () => {
    const newKey = `arg${
      Object.keys(operation.otherKwargs?.method_kwargs || {}).length + 1
    }`;
    handleMethodKwargsChange(newKey, "");
  };

  const removeMethodKwarg = (keyToRemove: string) => {
    const newMethodKwargs = { ...operation.otherKwargs?.method_kwargs };
    delete newMethodKwargs[keyToRemove];

    // Ensure required kwargs are present
    if (
      operation.otherKwargs?.method === "delimiter" &&
      !newMethodKwargs.delimiter
    ) {
      newMethodKwargs.delimiter = "";
    } else if (
      operation.otherKwargs?.method === "token_count" &&
      !newMethodKwargs.num_tokens
    ) {
      newMethodKwargs.num_tokens = 1;
    }

    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        method_kwargs: newMethodKwargs,
      },
    });
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <Label htmlFor="split-key" className="w-24">
          Split Key
        </Label>
        <Input
          id="split-key"
          value={operation.otherKwargs?.split_key || ""}
          onChange={(e) => handleSplitKeyChange(e.target.value)}
          className="w-64"
        />
      </div>
      <div className="flex items-center gap-4">
        <Label htmlFor="method" className="w-24">
          Method
        </Label>
        <Select
          onValueChange={handleMethodChange}
          value={operation.otherKwargs?.method || ""}
        >
          <SelectTrigger className="w-64">
            <SelectValue placeholder="Select a method" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="delimiter">Delimiter</SelectItem>
            <SelectItem value="token_count">Token Count</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="flex items-center gap-4">
        <Label className="w-24">Method Args</Label>
        <div className="flex-1 space-y-2">
          {Object.entries(operation.otherKwargs?.method_kwargs || {}).map(
            ([key, value]) => (
              <div key={key} className="flex items-center gap-2">
                <Input
                  value={key}
                  onChange={(e) => {
                    const newKwargs = {
                      ...operation.otherKwargs?.method_kwargs,
                    };
                    delete newKwargs[key];
                    newKwargs[e.target.value] = value;
                    onUpdate({
                      ...operation,
                      otherKwargs: {
                        ...operation.otherKwargs,
                        method_kwargs: newKwargs,
                      },
                    });
                  }}
                  className="w-1/3"
                  readOnly={
                    (operation.otherKwargs?.method === "delimiter" &&
                      key === "delimiter") ||
                    (operation.otherKwargs?.method === "token_count" &&
                      key === "num_tokens")
                  }
                />
                <Input
                  value={value as string}
                  onChange={(e) =>
                    handleMethodKwargsChange(key, e.target.value)
                  }
                  className="w-1/3"
                />
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => removeMethodKwarg(key)}
                  disabled={
                    (operation.otherKwargs?.method === "delimiter" &&
                      key === "delimiter") ||
                    (operation.otherKwargs?.method === "token_count" &&
                      key === "num_tokens")
                  }
                >
                  <X size={16} />
                </Button>
              </div>
            )
          )}
          <Button size="sm" variant="outline" onClick={addMethodKwarg}>
            <Plus size={16} className="mr-2" /> Add Argument
          </Button>
        </div>
      </div>
    </div>
  );
};

export const UnnestOperationComponent: React.FC<OperationComponentProps> = ({
  operation,
  onUpdate,
}) => {
  const handleUnnestKeyChange = (value: string) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        unnest_key: value,
      },
    });
  };

  const handleRecursiveChange = (checked: boolean) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        recursive: checked,
      },
    });
  };

  const handleDepthChange = (value: number) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        depth: value,
      },
    });
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-4">
        <div className="w-1/2">
          <Label htmlFor="unnest-key" className="text-sm font-medium">
            Unnest Key
          </Label>
          <Input
            id="unnest-key"
            value={operation.otherKwargs?.unnest_key || ""}
            onChange={(e) => handleUnnestKeyChange(e.target.value)}
            placeholder="Enter the key to flatten documents along"
            className="mt-1"
          />
        </div>
      </div>
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <Label
            htmlFor="recursive"
            className="text-sm font-medium cursor-pointer"
          >
            Recursive
          </Label>
          <Checkbox
            id="recursive"
            checked={operation.otherKwargs?.recursive || false}
            onCheckedChange={handleRecursiveChange}
          />
        </div>
        <div className="flex items-center space-x-2">
          <Label htmlFor="depth" className="text-sm font-medium">
            Depth
          </Label>
          <Input
            id="depth"
            type="number"
            value={operation.otherKwargs?.depth || ""}
            onChange={(e) => handleDepthChange(Number(e.target.value))}
            placeholder="Max depth"
            className="w-32"
          />
        </div>
      </div>
    </div>
  );
};

export const GatherOperationComponent: React.FC<OperationComponentProps> = ({
  operation,
  onUpdate,
  isSchemaExpanded,
  onToggleSchema,
}) => {
  const handleInputChange = (key: string, value: string) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        [key]: value || undefined,
      },
    });
  };

  const handlePeripheralChunksChange = (
    section: "previous" | "next",
    subsection: "head" | "middle" | "tail",
    key: string,
    value: any
  ) => {
    const updatedPeripheralChunks = {
      ...(operation.otherKwargs?.peripheral_chunks || {}),
      [section]: {
        ...(operation.otherKwargs?.peripheral_chunks?.[section] || {}),
        [subsection]: {
          ...(operation.otherKwargs?.peripheral_chunks?.[section]?.[
            subsection
          ] || {}),
          [key]: value || undefined,
        },
      },
    };

    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        peripheral_chunks: updatedPeripheralChunks,
      },
    });
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-4">
        <div className="w-1/3">
          <div className="flex items-center space-x-2">
            <Label htmlFor="content-key" className="text-sm font-medium">
              Content Key
            </Label>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Info className="h-4 w-4 text-gray-500" />
                </TooltipTrigger>
                <TooltipContent>
                  <p>
                    Append _chunk to the split_key you used in the split
                    operation
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <Input
            id="content-key"
            value={operation.otherKwargs?.content_key || ""}
            onChange={(e) => handleInputChange("content_key", e.target.value)}
            placeholder="Enter the content key"
            className="mt-1"
          />
        </div>
        <div className="w-1/3">
          <div className="flex items-center space-x-2">
            <Label htmlFor="doc-id-key" className="text-sm font-medium">
              Document ID Key
            </Label>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Info className="h-4 w-4 text-gray-500" />
                </TooltipTrigger>
                <TooltipContent>
                  <p>
                    Append _id to the name of the split operation you previously
                    defined
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <Input
            id="doc-id-key"
            value={operation.otherKwargs?.doc_id_key || ""}
            onChange={(e) => handleInputChange("doc_id_key", e.target.value)}
            placeholder="Enter the document ID key"
            className="mt-1"
          />
        </div>
        <div className="w-1/3">
          <div className="flex items-center space-x-2">
            <Label htmlFor="order-key" className="text-sm font-medium">
              Order Key
            </Label>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Info className="h-4 w-4 text-gray-500" />
                </TooltipTrigger>
                <TooltipContent>
                  <p>
                    Append _chunk_num to the name of the split operation you
                    previously defined
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <Input
            id="order-key"
            value={operation.otherKwargs?.order_key || ""}
            onChange={(e) => handleInputChange("order_key", e.target.value)}
            placeholder="Enter the order key"
            className="mt-1"
          />
        </div>
      </div>
      <div className="space-y-2">
        <div className="flex items-center space-x-2">
          <Label className="text-sm font-medium">Peripheral Chunks</Label>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <Info className="h-4 w-4 text-gray-500" />
              </TooltipTrigger>
              <TooltipContent>
                <p>Note: Values can be left empty to not include any context</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <div className="flex space-x-4">
          <div className="w-1/2 space-y-2">
            <Label className="text-sm font-medium">Previous</Label>
            <div className="pl-4 space-y-2">
              {["head", "middle", "tail"].map((subsection) => (
                <div key={subsection} className="flex items-center space-x-2">
                  <Label className="text-sm font-medium w-20">
                    {subsection.charAt(0).toUpperCase() + subsection.slice(1)}
                  </Label>
                  <Input
                    value={
                      operation.otherKwargs?.peripheral_chunks?.previous?.[
                        subsection
                      ]?.content_key || ""
                    }
                    onChange={(e) =>
                      handlePeripheralChunksChange(
                        "previous",
                        subsection as "head" | "middle" | "tail",
                        "content_key",
                        e.target.value
                      )
                    }
                    placeholder="Content key"
                    className="w-40"
                  />
                  {subsection !== "middle" && (
                    <Input
                      type="number"
                      value={
                        operation.otherKwargs?.peripheral_chunks?.previous?.[
                          subsection
                        ]?.count || ""
                      }
                      onChange={(e) =>
                        handlePeripheralChunksChange(
                          "previous",
                          subsection as "head" | "middle" | "tail",
                          "count",
                          Number(e.target.value)
                        )
                      }
                      placeholder="Count"
                      className="w-20"
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
          <div className="w-1/2 space-y-2">
            <Label className="text-sm font-medium">Next</Label>
            <div className="pl-4 space-y-2">
              {["head", "middle", "tail"].map((subsection) => (
                <div key={subsection} className="flex items-center space-x-2">
                  <Label className="text-sm font-medium w-20">
                    {subsection.charAt(0).toUpperCase() + subsection.slice(1)}
                  </Label>
                  <Input
                    value={
                      operation.otherKwargs?.peripheral_chunks?.next?.[
                        subsection
                      ]?.content_key || ""
                    }
                    onChange={(e) =>
                      handlePeripheralChunksChange(
                        "next",
                        subsection as "head" | "middle" | "tail",
                        "content_key",
                        e.target.value
                      )
                    }
                    placeholder="Content key"
                    className="w-40"
                  />
                  {subsection !== "middle" && (
                    <Input
                      type="number"
                      value={
                        operation.otherKwargs?.peripheral_chunks?.next?.[
                          subsection
                        ]?.count || ""
                      }
                      onChange={(e) =>
                        handlePeripheralChunksChange(
                          "next",
                          subsection as "head" | "middle" | "tail",
                          "count",
                          Number(e.target.value)
                        )
                      }
                      placeholder="Count"
                      className="w-20"
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export const ParallelMapOperationComponent: React.FC<
  OperationComponentProps
> = ({ operation, onUpdate, isSchemaExpanded, onToggleSchema }) => {
  const handlePromptChange = (index: number, field: string, value: string) => {
    const updatedPrompts = [...(operation.otherKwargs?.prompts || [])];
    updatedPrompts[index] = { ...updatedPrompts[index], [field]: value };

    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        prompts: updatedPrompts,
      },
    });
  };

  const handleOutputKeysChange = (
    index: number,
    action: "add" | "remove" | "update",
    value?: string,
    keyIndex?: number
  ) => {
    const updatedPrompts = [...(operation.otherKwargs?.prompts || [])];
    const currentOutputKeys = [...(updatedPrompts[index].output_keys || [])];

    if (action === "add") {
      currentOutputKeys.push("");
    } else if (action === "remove" && keyIndex !== undefined) {
      currentOutputKeys.splice(keyIndex, 1);
    } else if (
      action === "update" &&
      keyIndex !== undefined &&
      value !== undefined
    ) {
      currentOutputKeys[keyIndex] = value;
    }

    updatedPrompts[index] = {
      ...updatedPrompts[index],
      output_keys: currentOutputKeys,
    };

    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        prompts: updatedPrompts,
      },
    });
  };

  const addPrompt = () => {
    const updatedPrompts = [
      ...(operation.otherKwargs?.prompts || []),
      { prompt: "", output_keys: [], model: "" },
    ];
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        prompts: updatedPrompts,
      },
    });
  };

  const removePrompt = (index: number) => {
    const updatedPrompts = [...(operation.otherKwargs?.prompts || [])];
    updatedPrompts.splice(index, 1);
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        prompts: updatedPrompts,
      },
    });
  };

  return (
    <div className="space-y-4">
      {(operation.otherKwargs?.prompts || []).map(
        (prompt: PromptConfig, index: number) => (
          <div key={index} className="border p-2 rounded space-y-2">
            <div className="flex justify-between items-center">
              <Label className="text-sm font-medium">Prompt {index + 1}</Label>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => removePrompt(index)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <PromptInput
              prompt={prompt.prompt || ""}
              onChange={(value) => handlePromptChange(index, "prompt", value)}
            />
            <div className="flex items-center space-x-2">
              <div className="flex-grow">
                <Label className="text-sm font-medium">Output Keys</Label>
                <div className="flex flex-wrap gap-2 mt-1">
                  {prompt.output_keys?.map((key: string, keyIndex: number) => (
                    <div key={keyIndex} className="flex items-center">
                      <Input
                        value={key}
                        onChange={(e) =>
                          handleOutputKeysChange(
                            index,
                            "update",
                            e.target.value,
                            keyIndex
                          )
                        }
                        className="w-48"
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          handleOutputKeysChange(
                            index,
                            "remove",
                            undefined,
                            keyIndex
                          )
                        }
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>
                  ))}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleOutputKeysChange(index, "add")}
                  >
                    <Plus className="h-3 w-3" />
                  </Button>
                </div>
              </div>
              <div>
                <Label className="text-sm font-medium">Model</Label>
                <Input
                  value={prompt.model || ""}
                  onChange={(e) =>
                    handlePromptChange(index, "model", e.target.value)
                  }
                  placeholder="Model"
                  className="w-48 mt-1"
                />
              </div>
            </div>
          </div>
        )
      )}
      <Button onClick={addPrompt} size="sm">
        Add Prompt
      </Button>
      <OutputSchema
        schema={operation.output?.schema || []}
        onUpdate={(newSchema) =>
          onUpdate({
            ...operation,
            output: { ...operation.output, schema: newSchema },
          })
        }
        isExpanded={isSchemaExpanded}
        onToggle={onToggleSchema}
      />
    </div>
  );
};

export const SampleOperationComponent: React.FC<OperationComponentProps> = ({
  operation,
  onUpdate,
  isSchemaExpanded,
  onToggleSchema,
}) => {
  const handleChange = (field: string, value: string | number | boolean) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        [field]: value,
      },
    });
  };

  const handleMethodKwargsChange = (
    field: string,
    value: string | number | boolean | string[]
  ) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        method_kwargs: {
          ...operation.otherKwargs?.method_kwargs,
          [field]: value,
        },
      },
    });
  };

  return (
    <div className="space-y-4 mb-2">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="method">Method</Label>
          <Select
            value={operation.otherKwargs?.method || ""}
            onValueChange={(value) => handleChange("method", value)}
          >
            <SelectTrigger id="method">
              <SelectValue placeholder="Select a method" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="uniform">Uniform</SelectItem>
              <SelectItem value="stratify">Stratify</SelectItem>
              <SelectItem value="outliers">Outliers</SelectItem>
              <SelectItem value="custom">Custom</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="samples">Samples</Label>
          {operation.otherKwargs?.method === "custom" ? (
            <Textarea
              id="samples"
              value={operation.otherKwargs?.samples || ""}
              onChange={(e) => handleChange("samples", e.target.value)}
              placeholder="Enter JSON key-value pairs"
              className={`font-mono ${(() => {
                try {
                  const value = operation.otherKwargs?.samples;
                  if (!value) return "";
                  const parsed = JSON.parse(value);
                  if (
                    !Array.isArray(parsed) ||
                    !parsed.every((item) => typeof item === "object")
                  ) {
                    return "border-red-500";
                  }
                  return "";
                } catch {
                  return "border-red-500";
                }
              })()}`}
            />
          ) : (
            <Input
              id="samples"
              type="text"
              value={operation.otherKwargs?.samples || ""}
              onChange={(e) => handleChange("samples", e.target.value)}
              placeholder="Number or fraction of samples"
            />
          )}
        </div>
      </div>
      {operation.otherKwargs?.method === "stratify" && (
        <div>
          <Label htmlFor="stratify_key">Stratify Key</Label>
          <Input
            id="stratify_key"
            type="text"
            value={operation.otherKwargs?.method_kwargs?.stratify_key || ""}
            onChange={(e) =>
              handleMethodKwargsChange("stratify_key", e.target.value)
            }
            placeholder="Key to stratify by"
          />
        </div>
      )}
      {operation.otherKwargs?.method === "outliers" && (
        <>
          <div>
            <Label htmlFor="embedding_keys">Embedding Keys</Label>
            <Input
              id="embedding_keys"
              type="text"
              value={
                operation.otherKwargs?.method_kwargs?.embedding_keys?.join(
                  ", "
                ) || ""
              }
              onChange={(e) =>
                handleMethodKwargsChange(
                  "embedding_keys",
                  e.target.value.split(", ")
                )
              }
              placeholder="Comma-separated list of keys"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="std">Standard Deviations</Label>
              <Input
                id="std"
                type="number"
                value={operation.otherKwargs?.method_kwargs?.std || ""}
                onChange={(e) =>
                  handleMethodKwargsChange("std", parseFloat(e.target.value))
                }
                placeholder="Number of standard deviations"
              />
            </div>
            <div>
              <Label htmlFor="keep">Keep Outliers</Label>
              <Select
                value={
                  operation.otherKwargs?.method_kwargs?.keep?.toString() || ""
                }
                onValueChange={(value) =>
                  handleMethodKwargsChange("keep", value === "true")
                }
              >
                <SelectTrigger id="keep">
                  <SelectValue placeholder="Select option" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="true">Keep</SelectItem>
                  <SelectItem value="false">Remove</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export const CodeOperationComponent: React.FC<OperationComponentProps> = ({
  operation,
  onUpdate,
}) => {
  useEffect(() => {
    if (
      operation.type === "code_reduce" &&
      !operation.otherKwargs?.reduce_key
    ) {
      handleReduceKeysChange(["_all"]);
    }
  }, []);

  const handleCodeChange = (newCode: string) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        code: newCode,
      },
    });
  };

  const handleReduceKeysChange = (newReduceKeys: string[]) => {
    onUpdate({
      ...operation,
      otherKwargs: {
        ...operation.otherKwargs,
        reduce_key: newReduceKeys,
      },
    });
  };

  return (
    <div className="space-y-4">
      {operation.type === "code_reduce" && (
        <div className="mb-4">
          <div className="flex items-center space-x-2">
            <Label htmlFor="reduce-keys" className="w-1/4">
              Reduce Key(s)
            </Label>
            <div className="flex-grow flex items-center space-x-2 overflow-x-auto">
              <div className="flex-nowrap flex items-center space-x-2">
                {(operation.otherKwargs?.reduce_key || [""]).map(
                  (key: string, index: number) => (
                    <div
                      key={index}
                      className="relative flex-shrink-0"
                      style={{ minWidth: "150px" }}
                    >
                      <div className="flex items-center">
                        <Input
                          id={`reduce-key-${index}`}
                          value={key}
                          onChange={(e) => {
                            const newKeys = [
                              ...(operation.otherKwargs?.reduce_key || [""]),
                            ];
                            newKeys[index] = e.target.value;
                            handleReduceKeysChange(newKeys);
                          }}
                          placeholder="Enter reduce key"
                          className={`w-full pr-8 ${
                            !key.trim()
                              ? "border-red-500 focus:ring-red-500"
                              : ""
                          }`}
                        />
                        {(operation.otherKwargs?.reduce_key?.length || 0) >
                          1 && (
                          <Button
                            onClick={() => {
                              const newKeys = [
                                ...(operation.otherKwargs?.reduce_key || [""]),
                              ];
                              newKeys.splice(index, 1);
                              handleReduceKeysChange(newKeys);
                            }}
                            size="sm"
                            variant="ghost"
                            className="absolute right-0 top-0 h-full"
                          >
                            <X size={12} />
                          </Button>
                        )}
                      </div>
                      {!key.trim() && (
                        <p className="text-red-500 text-sm mt-1">
                          There must be a reduce key. Set "_all" if you want all
                          documents in a group.
                        </p>
                      )}
                    </div>
                  )
                )}
                <Button
                  onClick={() => {
                    const newKeys = [
                      ...(operation.otherKwargs?.reduce_key || [""]),
                      "",
                    ];
                    handleReduceKeysChange(newKeys);
                  }}
                  size="sm"
                  variant="outline"
                  className="flex-shrink-0"
                >
                  <Plus size={16} />
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
      <div>
        <CodeInput
          code={operation.otherKwargs?.code || ""}
          operationType={
            operation.type as "code_map" | "code_reduce" | "code_filter"
          }
          onChange={handleCodeChange}
        />
      </div>
    </div>
  );
};

export default function createOperationComponent(
  operation: Operation,
  onUpdate: (updatedOperation: Operation) => void,
  isSchemaExpanded: boolean,
  onToggleSchema: () => void
) {
  switch (operation.type) {
    case "reduce":
      return (
        <ReduceOperationComponent
          operation={operation}
          onUpdate={onUpdate}
          isSchemaExpanded={isSchemaExpanded}
          onToggleSchema={onToggleSchema}
        />
      );
    case "map":
    case "filter":
      return (
        <MapFilterOperationComponent
          operation={operation}
          onUpdate={onUpdate}
          isSchemaExpanded={isSchemaExpanded}
          onToggleSchema={onToggleSchema}
        />
      );
    case "resolve":
      return (
        <ResolveOperationComponent
          operation={operation}
          onUpdate={onUpdate}
          isSchemaExpanded={isSchemaExpanded}
          onToggleSchema={onToggleSchema}
        />
      );
    case "parallel_map":
      return (
        <ParallelMapOperationComponent
          operation={operation}
          onUpdate={onUpdate}
          isSchemaExpanded={isSchemaExpanded}
          onToggleSchema={onToggleSchema}
        />
      );
    case "unnest":
      return (
        <UnnestOperationComponent
          operation={operation}
          onUpdate={onUpdate}
          isSchemaExpanded={isSchemaExpanded}
          onToggleSchema={onToggleSchema}
        />
      );
    case "split":
      return (
        <SplitOperationComponent
          operation={operation}
          onUpdate={onUpdate}
          isSchemaExpanded={isSchemaExpanded}
          onToggleSchema={onToggleSchema}
        />
      );
    case "gather":
      return (
        <GatherOperationComponent
          operation={operation}
          onUpdate={onUpdate}
          isSchemaExpanded={isSchemaExpanded}
          onToggleSchema={onToggleSchema}
        />
      );
    case "sample":
      return (
        <SampleOperationComponent
          operation={operation}
          onUpdate={onUpdate}
          isSchemaExpanded={isSchemaExpanded}
          onToggleSchema={onToggleSchema}
        />
      );
    case "code_map":
    case "code_reduce":
    case "code_filter":
      return (
        <CodeOperationComponent
          operation={operation}
          onUpdate={onUpdate}
          isSchemaExpanded={isSchemaExpanded}
          onToggleSchema={onToggleSchema}
        />
      );
    default:
      console.warn(`Unsupported operation type: ${operation.type}`);
      return null;
  }
}
