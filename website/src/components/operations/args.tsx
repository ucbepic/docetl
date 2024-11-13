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
import { type FC, memo } from "react";

type PromptInputProps = {
  prompt: string;
  onChange: (value: string) => void;
};

export const PromptInput: FC<PromptInputProps> = memo(
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
          rows={3}
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

type SchemaFormProps = {
  schema: SchemaItem[];
  onUpdate: (newSchema: SchemaItem[]) => void;
  level?: number;
  isList?: boolean;
};

export const SchemaForm: FC<SchemaFormProps> = memo(
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
                className="w-1/3 min-w-[150px]"
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
              <SelectTrigger
                className={`w-1/3 min-w-[150px] ${isList ? "flex-grow" : ""}`}
              >
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

type OutputSchemaProps = {
  schema: SchemaItem[];
  onUpdate: (newSchema: SchemaItem[]) => void;
  isExpanded: boolean;
  onToggle: () => void;
};

export const OutputSchema: FC<OutputSchemaProps> = memo(
  ({ schema, onUpdate, isExpanded, onToggle }) => {
    return (
      <div>
        <Button variant="ghost" size="sm" onClick={onToggle} className="p-0">
          <ChevronDown
            size={16}
            className={`mr-1 transition-transform duration-200 ${
              isExpanded ? "transform rotate-180" : ""
            }`}
          />
          <h4 className="text-xs font-semibold">Output Schema</h4>
        </Button>
        {isExpanded && <SchemaForm schema={schema} onUpdate={onUpdate} />}
      </div>
    );
  }
);

OutputSchema.displayName = "OutputSchema";

type GleaningConfig = {
  num_rounds: number;
  validation_prompt: string;
} | null;

type GleaningConfigProps = {
  gleaning: GleaningConfig;
  onUpdate: (newGleaning: GleaningConfig) => void;
  isExpanded: boolean;
  onToggle: () => void;
};

export const GleaningConfig: FC<GleaningConfigProps> = memo(
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

type GuardrailsProps = {
  guardrails: string[];
  onUpdate: (newGuardrails: string[]) => void;
  isExpanded: boolean;
  onToggle: () => void;
};

export const Guardrails: FC<GuardrailsProps> = memo(
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
                    Example: "len(output["summary"]) {">"} 100" ensures a
                    summary is at least 100 characters long.
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
