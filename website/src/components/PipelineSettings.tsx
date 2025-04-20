import React, { useState, useMemo, useCallback } from "react";
import { File } from "@/app/types";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { AlertCircle } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import yaml from "js-yaml";

const PREDEFINED_MODELS = [
  "gpt-4o-mini",
  "gpt-4o",
  "claude-3-7-sonnet-20250219",
  "claude-3-opus-20240229",
  "azure/<your-deployment-name>",
  "gemini/gemini-2.0-flash",
] as const;

interface ModelInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  suggestions?: readonly string[];
}

export const ModelInput: React.FC<ModelInputProps> = ({
  value,
  onChange,
  placeholder,
  suggestions = PREDEFINED_MODELS,
}) => {
  const [isFocused, setIsFocused] = useState(false);

  return (
    <div className="relative">
      <Input
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        className="w-full"
        placeholder={placeholder}
        onFocus={() => setIsFocused(true)}
        onBlur={() => {
          setTimeout(() => setIsFocused(false), 200);
        }}
      />
      {isFocused &&
        (value === "" ||
          suggestions.some((model) =>
            model.toLowerCase().includes(value?.toLowerCase() || "")
          )) && (
          <div className="absolute top-full left-0 w-full mt-1 bg-popover rounded-md border shadow-md z-50 max-h-[200px] overflow-y-auto">
            {suggestions
              .filter(
                (model) =>
                  value === "" ||
                  model.toLowerCase().includes(value.toLowerCase())
              )
              .map((model) => (
                <div
                  key={model}
                  className="px-2 py-1.5 text-sm cursor-pointer hover:bg-accent hover:text-accent-foreground"
                  onClick={() => {
                    onChange(model);
                    setIsFocused(false);
                  }}
                >
                  {model}
                </div>
              ))}
          </div>
        )}
    </div>
  );
};

interface PipelineSettingsProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  pipelineName: string;
  setPipelineName: (name: string) => void;
  currentFile: File | null;
  setCurrentFile: (file: File | null) => void;
  defaultModel: string;
  setDefaultModel: (model: string) => void;
  optimizerModel: string;
  setOptimizerModel: (model: string) => void;
  autoOptimizeCheck: boolean;
  setAutoOptimizeCheck: (check: boolean) => void;
  files: File[];
  apiKeys: Array<{ name: string; value: string }>;
  extraPipelineSettings: Record<string, unknown> | null;
  setExtraPipelineSettings: (settings: Record<string, unknown> | null) => void;
}

const SAMPLE_YAML = `# Example configuration - delete or modify as needed
rate_limits:
  llm_call:
    - count: 1000000
      per: 1
      unit: minute
  llm_tokens:
    - count: 1000000000
      per: 1
      unit: minute`;

const PipelineSettings: React.FC<PipelineSettingsProps> = ({
  isOpen,
  onOpenChange,
  pipelineName,
  setPipelineName,
  currentFile,
  setCurrentFile,
  defaultModel,
  setDefaultModel,
  optimizerModel,
  setOptimizerModel,
  autoOptimizeCheck,
  setAutoOptimizeCheck,
  files,
  apiKeys,
  extraPipelineSettings,
  setExtraPipelineSettings,
}) => {
  const [tempPipelineName, setTempPipelineName] = useState(pipelineName);
  const [tempCurrentFile, setTempCurrentFile] = useState<File | null>(
    currentFile
  );
  const [tempDefaultModel, setTempDefaultModel] = useState(defaultModel);
  const [tempOptimizerModel, setTempOptimizerModel] = useState(optimizerModel);
  const [tempAutoOptimizeCheck, setTempAutoOptimizeCheck] =
    useState(autoOptimizeCheck);
  const [isLocalMode, setIsLocalMode] = useState(false);

  // Convert extraPipelineSettings to YAML string
  const initialYamlString = useMemo(() => {
    if (!extraPipelineSettings) {
      return "";
    }
    try {
      return yaml.dump(extraPipelineSettings);
    } catch (e) {
      console.error("Error converting settings to YAML:", e);
      return "";
    }
  }, [extraPipelineSettings]);

  const [tempYamlSettings, setTempYamlSettings] = useState(initialYamlString);
  const [yamlError, setYamlError] = useState<string | null>(null);

  const hasOpenAIKey = useMemo(() => {
    return apiKeys.some((key) => key.name === "OPENAI_API_KEY");
  }, [apiKeys]);

  // Update local state when props change
  React.useEffect(() => {
    setTempPipelineName(pipelineName);
    setTempCurrentFile(currentFile);
    setTempDefaultModel(defaultModel);
    setTempOptimizerModel(optimizerModel);
    setTempAutoOptimizeCheck(autoOptimizeCheck);

    // Update YAML when extraPipelineSettings changes
    if (extraPipelineSettings) {
      try {
        setTempYamlSettings(yaml.dump(extraPipelineSettings));
      } catch (e) {
        console.error("Error converting settings to YAML:", e);
      }
    } else {
      setTempYamlSettings("");
    }
  }, [
    pipelineName,
    currentFile,
    defaultModel,
    optimizerModel,
    autoOptimizeCheck,
    extraPipelineSettings,
  ]);

  const validateYaml = useCallback((yamlString: string) => {
    if (!yamlString.trim()) {
      setYamlError(null);
      return null;
    }

    try {
      const parsed = yaml.load(yamlString);
      setYamlError(null);
      return parsed as Record<string, unknown>;
    } catch (e) {
      const error = e as Error;
      setYamlError(`Invalid YAML: ${error.message}`);
      return null;
    }
  }, []);

  const handleYamlChange = useCallback(
    (value: string) => {
      setTempYamlSettings(value);
      validateYaml(value);
    },
    [validateYaml]
  );

  const handleSettingsSave = () => {
    setPipelineName(tempPipelineName);
    setCurrentFile(tempCurrentFile);
    setDefaultModel(tempDefaultModel);
    setOptimizerModel(tempOptimizerModel);
    setAutoOptimizeCheck(tempAutoOptimizeCheck);

    // Process and save YAML settings
    if (tempYamlSettings.trim()) {
      const parsedSettings = validateYaml(tempYamlSettings);
      if (parsedSettings) {
        setExtraPipelineSettings(parsedSettings);
      }
    } else {
      setExtraPipelineSettings(null);
    }

    onOpenChange(false);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Pipeline Settings</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="flex flex-col space-y-1.5">
            <Label htmlFor="pipelineName">Pipeline Name</Label>
            <Input
              id="pipelineName"
              value={tempPipelineName}
              onChange={(e) => setTempPipelineName(e.target.value)}
              placeholder="Enter pipeline name"
            />
          </div>

          <div className="flex flex-col space-y-1.5">
            <Label htmlFor="currentFile">Dataset JSON</Label>
            <Select
              value={tempCurrentFile?.path || ""}
              onValueChange={(value) =>
                setTempCurrentFile(
                  files.find((file) => file.path === value) || null
                )
              }
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a file" />
              </SelectTrigger>
              <SelectContent>
                {files
                  .filter((file) => file.type === "json")
                  .map((file) => (
                    <SelectItem key={file.path} value={file.path}>
                      {file.name}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col space-y-1.5">
            <Label htmlFor="defaultModel">Default Model</Label>
            <ModelInput
              value={tempDefaultModel}
              onChange={setTempDefaultModel}
              placeholder="Enter or select a model..."
            />
            <p className="text-xs text-muted-foreground">
              Enter any LiteLLM model name or select from suggestions. Make sure
              you&apos;ve set your API keys in Edit {">"} Edit API Keys when
              using our hosted app.{" "}
              <a
                href="https://docs.litellm.ai/docs/providers"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 hover:underline"
              >
                View all supported models {String.fromCharCode(8594)}
              </a>
            </p>
          </div>

          <div className="flex flex-col space-y-1.5">
            <Label htmlFor="optimize">Optimizer Model</Label>
            {!hasOpenAIKey && !isLocalMode ? (
              <div className="bg-destructive/10 text-destructive rounded-md p-3 text-xs">
                <div className="flex gap-2">
                  <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium">OpenAI API Key Required</p>
                    <p className="mt-1">
                      To use the optimizer, please add your OpenAI API key in
                      Edit {">"} Edit API Keys.
                    </p>
                    <button
                      className="text-destructive underline hover:opacity-80 mt-1.5 font-medium"
                      onClick={() => setIsLocalMode(true)}
                    >
                      Skip if running locally with environment variables
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                <ModelInput
                  value={tempOptimizerModel}
                  onChange={setTempOptimizerModel}
                  placeholder="Enter optimizer model name..."
                  suggestions={["gpt-4o", "gpt-4o-mini"]}
                />
                <p className="text-xs text-muted-foreground">
                  Enter any LiteLLM model name (e.g., &quot;azure/gpt-4o&quot;)
                  or select from suggestions above. Make sure the model supports
                  JSON mode.
                </p>
              </div>
            )}
          </div>

          <div className="flex flex-col space-y-1.5">
            <Label htmlFor="autoOptimize">
              Automatically Check Whether to Optimize
            </Label>
            <Switch
              id="autoOptimize"
              checked={tempAutoOptimizeCheck}
              onCheckedChange={(checked) => setTempAutoOptimizeCheck(checked)}
              disabled={!hasOpenAIKey && !isLocalMode}
            />
          </div>

          <div className="flex flex-col space-y-1.5">
            <div className="flex justify-between items-center">
              <Label htmlFor="advancedSettings">
                Advanced Pipeline Settings (YAML)
              </Label>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={() => setTempYamlSettings(SAMPLE_YAML)}
              >
                Add Example
              </Button>
            </div>
            <Textarea
              id="advancedSettings"
              value={tempYamlSettings}
              onChange={(e) => handleYamlChange(e.target.value)}
              placeholder="Enter YAML configuration for rate limits and other advanced settings"
              className="font-mono text-sm h-48 resize-y"
            />
            {yamlError && (
              <div className="text-sm text-destructive">{yamlError}</div>
            )}
            <p className="text-sm text-muted-foreground">
              Configure rate limits and other advanced settings in YAML format.
              These settings will be passed to the backend.
            </p>
          </div>
        </div>
        <DialogFooter>
          <Button
            onClick={handleSettingsSave}
            disabled={!!yamlError && tempYamlSettings.trim() !== ""}
          >
            Save changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default PipelineSettings;
