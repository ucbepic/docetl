import React from "react";
import { Textarea } from "../ui/textarea";
import { Input } from "../ui/input";
import { Button } from "../ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Trash2, Plus, ChevronDown, Info } from "lucide-react";
import { SchemaItem, SchemaType } from "@/app/types";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { Label } from "../ui/label";
import Editor from "@monaco-editor/react";
import PropTypes from "prop-types";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "../ui/hover-card";

interface PromptInputProps {
  prompt: string;
  onChange: (value: string) => void;
}

export const PromptInput: React.FC<PromptInputProps> = React.memo(
  ({ prompt, onChange }) => {
    const validateJinjaTemplate = (value: string) => {
      const hasOpenBrace = value.includes("{{");
      const hasCloseBrace = value.includes("}}");
      return hasOpenBrace && hasCloseBrace;
    };

    return (
      <>
        <Textarea
          placeholder="Enter prompt (must be a Jinja2 template)"
          className={`mb-1 rounded-sm text-sm font-mono ${
            !validateJinjaTemplate(prompt) ? "border-red-500" : ""
          }`}
          rows={Math.max(3, Math.ceil(prompt.split("\n").length))}
          value={prompt}
          onChange={(e) => onChange(e.target.value)}
        />
        {!validateJinjaTemplate(prompt) && (
          <div className="text-red-500 text-sm mb-1">
            Prompt must contain Jinja2 template syntax {"{"}
            {"{"} and {"}"}
            {"}"}
          </div>
        )}
      </>
    );
  }
);

PromptInput.displayName = "PromptInput";

PromptInput.propTypes = {
  prompt: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
};

interface SchemaFormProps {
  schema: SchemaItem[];
  onUpdate: (newSchema: SchemaItem[]) => void;
  level?: number;
  isList?: boolean;
}

export const SchemaForm: React.FC<SchemaFormProps> = React.memo(
  ({ schema, onUpdate, level = 0, isList = false }) => {
    const addItem = () => {
      if (isList) return;
      onUpdate([...schema, { key: "", type: "string" }]);
    };

    const updateItem = (index: number, item: SchemaItem) => {
      const newSchema = [...schema];
      newSchema[index] = item;
      onUpdate(newSchema);
    };

    const removeItem = (index: number) => {
      if (isList) return;
      const newSchema = schema.filter((_, i) => i !== index);
      onUpdate(newSchema);
    };

    return (
      <div style={{ marginLeft: `${level * 20}px` }}>
        {schema.map((item, index) => (
          <div
            key={index}
            className="flex flex-wrap items-center space-x-2 mb-1"
          >
            {!isList && (
              <Input
                value={item.key}
                onChange={(e) =>
                  updateItem(index, { ...item, key: e.target.value })
                }
                placeholder="Key"
                className={`w-1/3 min-w-[150px] ${
                  !item.key ? "border-red-500" : ""
                }`}
              />
            )}
            <Select
              value={item.type}
              onValueChange={(value: SchemaType) => {
                updateItem(index, {
                  ...item,
                  type: value,
                  subType:
                    value === "list"
                      ? { key: "0", type: "string" }
                      : value === "dict"
                      ? [{ key: "", type: "string" }]
                      : undefined,
                });
              }}
            >
              <SelectTrigger className={`w-32 ${isList ? "flex-grow" : ""}`}>
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="string">string</SelectItem>
                <SelectItem value="float">float</SelectItem>
                <SelectItem value="int">int</SelectItem>
                <SelectItem value="boolean">boolean</SelectItem>
                <SelectItem value="list">list</SelectItem>
                <SelectItem value="dict">dict</SelectItem>
              </SelectContent>
            </Select>
            {!isList && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => removeItem(index)}
                className="p-1"
              >
                <Trash2 size={16} />
              </Button>
            )}
            {item.type === "list" && item.subType && (
              <div className="w-full mt-1 ml-4 flex items-center">
                <span className="mr-2 text-sm text-gray-500">List type:</span>
                <SchemaForm
                  schema={[item.subType as SchemaItem]}
                  onUpdate={(newSubSchema) =>
                    updateItem(index, { ...item, subType: newSubSchema[0] })
                  }
                  level={0}
                  isList={true}
                />
              </div>
            )}
            {item.type === "dict" && item.subType && (
              <div className="w-full mt-1 ml-4">
                <SchemaForm
                  schema={item.subType as SchemaItem[]}
                  onUpdate={(newSubSchema) =>
                    updateItem(index, { ...item, subType: newSubSchema })
                  }
                  level={level + 1}
                />
              </div>
            )}
            {!isList && !item.key && (
              <div className="w-full mt-1 text-red-500 text-sm">
                Key is required
              </div>
            )}
          </div>
        ))}
        {!isList && (
          <Button
            variant="outline"
            size="sm"
            onClick={addItem}
            className="mt-1"
          >
            <Plus size={16} className="mr-2" /> Add Field
          </Button>
        )}
      </div>
    );
  }
);

SchemaForm.displayName = "SchemaForm";

SchemaForm.propTypes = {
  // @ts-expect-error - PropTypes schema doesn't match TypeScript SchemaItem[] type exactly
  schema: PropTypes.arrayOf(
    PropTypes.shape({
      key: PropTypes.string,
      type: PropTypes.oneOf([
        "string",
        "float",
        "int",
        "boolean",
        "list",
        "dict",
      ]).isRequired,
      subType: PropTypes.oneOfType([
        PropTypes.object,
        PropTypes.arrayOf(PropTypes.object),
      ]),
    })
  ).isRequired,
  onUpdate: PropTypes.func.isRequired,
  level: PropTypes.number,
  isList: PropTypes.bool,
};

interface OutputSchemaProps {
  schema: SchemaItem[];
  onUpdate: (newSchema: SchemaItem[]) => void;
  isExpanded: boolean;
  onToggle: () => void;
}

export const OutputSchema: React.FC<OutputSchemaProps> = React.memo(
  ({ schema, onUpdate, isExpanded, onToggle }) => {
    const isEmpty = schema.length === 0;
    const hasEmptyKeys = schema.some((item) => !item.key);

    return (
      <div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggle}
            className={`p-0 ${isEmpty || hasEmptyKeys ? "text-red-500" : ""}`}
          >
            <ChevronDown
              size={16}
              className={`mr-1 transition-transform duration-200 ${
                isExpanded ? "transform rotate-180" : ""
              }`}
            />
            <h4 className="text-xs font-semibold">
              Output Schema {isEmpty && "(Required)"}
            </h4>
          </Button>
          <HoverCard>
            <HoverCardTrigger>
              <Info size={16} className="text-primary cursor-help" />
            </HoverCardTrigger>
            <HoverCardContent className="w-80">
              <div className="space-y-2">
                <h4 className="font-medium">Output Column Naming</h4>
                <p className="text-sm text-muted-foreground">
                  Name your columns appropriately as they influence the
                  LLM&apos;s output.
                </p>
                <div className="mt-2 rounded-md bg-muted p-2">
                  <p className="text-sm font-medium">Example:</p>
                  <p className="text-xs text-muted-foreground">
                    If your prompt extracts names from a document, use
                    &quot;names&quot; as your output column name instead of
                    &quot;extracted_data&quot; or &quot;results&quot;.
                  </p>
                </div>
              </div>
            </HoverCardContent>
          </HoverCard>
        </div>
        {isExpanded && <SchemaForm schema={schema} onUpdate={onUpdate} />}
        {isEmpty && (
          <div className="text-red-500 text-sm mt-1">
            At least one output field is required
          </div>
        )}
        {hasEmptyKeys && !isEmpty && (
          <div className="text-red-500 text-sm mt-1">
            All fields must have a key name
          </div>
        )}
      </div>
    );
  }
);

OutputSchema.displayName = "OutputSchema";

OutputSchema.propTypes = {
  schema: PropTypes.array.isRequired,
  onUpdate: PropTypes.func.isRequired,
  isExpanded: PropTypes.bool.isRequired,
  onToggle: PropTypes.func.isRequired,
};

export interface GleaningConfigProps {
  gleaning: { num_rounds: number; validation_prompt: string } | null;
  onUpdate: (
    newGleaning: {
      num_rounds: number;
      validation_prompt: string;
    } | null
  ) => void;
  isExpanded: boolean;
  onToggle: () => void;
}

export const GleaningConfig: React.FC<GleaningConfigProps> = React.memo(
  ({ gleaning, onUpdate, isExpanded, onToggle }) => {
    return (
      <div className="border-t border-primary">
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggle}
          className="w-full text-primary hover:bg-primary/10 flex justify-between items-center"
        >
          <div className="flex items-center gap-2">
            <span>
              Gleaning {gleaning?.num_rounds ? "(enabled)" : "(not enabled)"}
            </span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Info size={16} className="text-primary" />
                </TooltipTrigger>
                <TooltipContent className="max-w-md whitespace-normal break-words text-left">
                  <p>
                    Gleaning allows you to iteratively refine outputs through
                    multiple rounds of validation and improvement.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <ChevronDown
            size={16}
            className={`transition-transform duration-200 ${
              isExpanded ? "transform rotate-180" : ""
            }`}
          />
        </Button>

        {isExpanded && (
          <div className="p-2">
            <div className="grid grid-cols-8 gap-4">
              <div className="col-span-1 space-y-2">
                <Label htmlFor="num_rounds">Rounds</Label>
                <Input
                  id="num_rounds"
                  type="number"
                  min="0"
                  max="5"
                  value={gleaning?.num_rounds || 0}
                  onChange={(e) =>
                    onUpdate({
                      ...gleaning,
                      num_rounds: parseInt(e.target.value) || 0,
                    })
                  }
                  className={gleaning?.num_rounds === 0 ? "border-red-500" : ""}
                />
              </div>

              <div className="col-span-7 space-y-2">
                <Label htmlFor="validation_prompt">Validation Prompt</Label>
                <Textarea
                  id="validation_prompt"
                  value={gleaning?.validation_prompt || ""}
                  onChange={(e) =>
                    onUpdate({
                      ...gleaning,
                      validation_prompt: e.target.value,
                    })
                  }
                  className={
                    !gleaning?.validation_prompt ? "border-red-500" : ""
                  }
                />
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }
);

GleaningConfig.displayName = "GleaningConfig";

GleaningConfig.propTypes = {
  // @ts-expect-error - PropTypes null type doesn't match TypeScript optional type
  gleaning: PropTypes.shape({
    num_rounds: PropTypes.number,
    validation_prompt: PropTypes.string,
  }),
  onUpdate: PropTypes.func.isRequired,
  isExpanded: PropTypes.bool.isRequired,
  onToggle: PropTypes.func.isRequired,
};

interface GuardrailsProps {
  guardrails: string[];
  onUpdate: (newGuardrails: string[]) => void;
  isExpanded: boolean;
  onToggle: () => void;
}

export const Guardrails: React.FC<GuardrailsProps> = React.memo(
  ({ guardrails, onUpdate, isExpanded, onToggle }) => {
    const handleGuardrailChange = (index: number, value: string) => {
      const newGuardrails = [...guardrails];
      newGuardrails[index] = value;
      onUpdate(newGuardrails);
    };

    const addGuardrail = () => {
      onUpdate([...guardrails, ""]);
    };

    const removeGuardrail = (index: number) => {
      const newGuardrails = guardrails.filter((_, i) => i !== index);
      onUpdate(newGuardrails);
    };

    return (
      <div className="border-t border-orange-500">
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggle}
          className="w-full text-orange-500 hover:bg-orange-50 flex justify-between items-center"
        >
          <div className="flex items-center">
            <span>Code-Based Guardrails ({guardrails.length})</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Info size={16} className="ml-2 text-orange-500" />
                </TooltipTrigger>
                <TooltipContent className="max-w-md whitespace-normal break-words text-left">
                  <p>
                    Guardrails are Python statements to validate output.
                    Example: &quot;len(output[&quot;summary&quot;]) &gt;
                    100&quot; ensures a summary is at least 100 characters long.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <ChevronDown
            size={16}
            className={`transition-transform duration-200 ${
              isExpanded ? "transform rotate-180" : ""
            }`}
          />
        </Button>
        {isExpanded && (
          <div className="bg-orange-50">
            {guardrails.map((guardrail, index) => (
              <div key={index} className="flex items-center mb-2">
                <Input
                  value={guardrail}
                  onChange={(e) => handleGuardrailChange(index, e.target.value)}
                  placeholder="Enter guardrail"
                  className="flex-grow text-sm text-orange-700 font-mono"
                />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeGuardrail(index)}
                  className="ml-2 p-1 h-7 w-7 hover:bg-orange-100"
                >
                  <Trash2 size={15} className="text-orange-500" />
                </Button>
              </div>
            ))}
            <Button
              variant="outline"
              size="sm"
              onClick={addGuardrail}
              className="mb-1 text-orange-500 border-orange-500 hover:bg-orange-100"
            >
              <Plus size={16} className="mr-2" /> Add Guardrail
            </Button>
          </div>
        )}
      </div>
    );
  }
);

Guardrails.displayName = "Guardrails";

Guardrails.propTypes = {
  guardrails: PropTypes.arrayOf(PropTypes.string).isRequired,
  onUpdate: PropTypes.func.isRequired,
  isExpanded: PropTypes.bool.isRequired,
  onToggle: PropTypes.func.isRequired,
};

interface CodeInputProps {
  code: string;
  operationType: "code_map" | "code_reduce" | "code_filter";
  onChange: (value: string) => void;
}

export const CodeInput: React.FC<CodeInputProps> = React.memo(
  ({ code, operationType, onChange }) => {
    const getPlaceholder = () => {
      switch (operationType) {
        case "code_map":
          return `def transform(doc: dict) -> dict:
    # Transform a single document
    # Return a dictionary with new key-value pairs
    return {
        'new_key': process(doc['existing_key'])
    }`;
        case "code_filter":
          return `def transform(doc: dict) -> bool:
    # Return True to keep the document, False to filter it out
    return doc['score'] >= 0.5`;
        case "code_reduce":
          return `def transform(items: list) -> dict:
    # Aggregate multiple items into a single result
    # Return a dictionary with aggregated values
    return {
        'total': sum(item['value'] for item in items),
        'count': len(items)
    }`;
      }
    };

    const validatePythonCode = (value: string) => {
      return value.includes("def transform") && value.includes("return");
    };

    const getTooltipContent = () => {
      switch (operationType) {
        case "code_map":
          return "Transform each document independently using Python code. The transform function takes a single document as input and returns a dictionary with new key-value pairs.";
        case "code_filter":
          return "Filter documents using Python code. The transform function takes a document as input and returns True to keep it or False to filter it out.";
        case "code_reduce":
          return "Aggregate multiple documents using Python code. The transform function takes a list of documents as input and returns a single aggregated result.";
      }
    };

    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 mb-1">
          <Label>Python Code</Label>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <Info className="h-4 w-4 text-gray-500" />
              </TooltipTrigger>
              <TooltipContent className="max-w-md">
                <p className="text-sm">{getTooltipContent()}</p>
                <p className="text-sm mt-2">
                  Code operations allow you to use Python for:
                  <ul className="list-disc ml-4 mt-1">
                    <li>Deterministic processing</li>
                    <li>Complex calculations</li>
                    <li>Integration with Python libraries</li>
                    <li>Structured data transformations</li>
                  </ul>
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <div
          className="border"
          style={{
            resize: "vertical",
            overflow: "auto",
            minHeight: "200px",
            height: "200px", // Set initial height explicitly
            backgroundColor: "var(--background)", // Match editor background
          }}
        >
          <Editor
            height="100%"
            defaultLanguage="python"
            value={code || getPlaceholder()}
            onChange={(value) => onChange(value || "")}
            options={{
              minimap: { enabled: false },
              lineNumbers: "on",
              scrollBeyondLastLine: false,
              wordWrap: "on",
              wrappingIndent: "indent",
              automaticLayout: true,
              tabSize: 4,
              fontSize: 14,
              fontFamily: "monospace",
              suggest: {
                showKeywords: true,
                showSnippets: true,
              },
            }}
          />
        </div>
        {!validatePythonCode(code) && (
          <div className="text-red-500 text-sm">
            Code must define a transform function with a return statement
          </div>
        )}
      </div>
    );
  }
);

CodeInput.displayName = "CodeInput";

CodeInput.propTypes = {
  code: PropTypes.string.isRequired,
  // @ts-expect-error - PropTypes string union doesn't match TypeScript type exactly
  operationType: PropTypes.oneOf(["code_map", "code_reduce", "code_filter"])
    .isRequired,
  onChange: PropTypes.func.isRequired,
};
