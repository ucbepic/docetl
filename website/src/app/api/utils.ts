import yaml from "js-yaml";
import path from "path";
import { Operation, SchemaItem } from "@/app/types";

export function getNamespaceDir(homeDir: string, namespace: string) {
  return path.join(homeDir, ".docetl", namespace);
}

export function generatePipelineConfig(
  namespace: string,
  default_model: string,
  data: { path: string },
  operations: Operation[],
  operation_id: string,
  name: string,
  homeDir: string,
  sample_size: number | null,
  optimize: boolean = false,
  clear_intermediate: boolean = false,
  system_prompt: {
    datasetDescription: string | null;
    persona: string | null;
  } | null = null
) {
  const datasets = {
    input: {
      type: "file",
      path: data.path,
      source: "local",
    },
  };

  // Augment the first operation with sample if sampleSize is not null
  if (operations.length > 0 && sample_size !== null) {
    operations[0] = {
      ...operations[0],
      sample: sample_size,
    };
  }

  // Fix the output schema of all operations to ensure correct typing
  const updatedOperations = operations
    .map((op: Operation) => {
      // Skip if visibility is false
      if (!op.visibility) {
        return null;
      }

      const newOp: Record<string, unknown> = {
        ...op,
        ...op.otherKwargs,
      };

      if (optimize && op.id === operation_id) {
        newOp.optimize = true;
      }

      if (clear_intermediate) {
        newOp.bypass_cache = true;
      }

      delete newOp.runIndex;
      delete newOp.otherKwargs;
      delete newOp.id;
      delete newOp.llmType;
      delete newOp.visibility;

      // Convert numeric strings in otherKwargs to numbers
      Object.entries(newOp).forEach(([key, value]) => {
        if (typeof value === "string") {
          // Try parsing as float first
          const floatVal = parseFloat(value);
          if (!isNaN(floatVal) && floatVal.toString() === value) {
            newOp[key] = floatVal;
            return;
          }
          // Try parsing as integer
          const intVal = parseInt(value, 10);
          if (!isNaN(intVal) && intVal.toString() === value) {
            newOp[key] = intVal;
          }
        }
      });

      if (
        op.gleaning &&
        (op.gleaning.num_rounds === 0 || !op.gleaning.validation_prompt)
      ) {
        delete newOp.gleaning;
      }

      if (!op.output || !op.output.schema) return newOp;

      const processSchemaItem = (item: SchemaItem): string => {
        if (item.type === "list") {
          if (!item.subType) {
            throw new Error(
              `List type must specify its elements for field: ${item.key}`
            );
          }
          const subType =
            typeof item.subType === "string"
              ? item.subType
              : processSchemaItem(item.subType as SchemaItem);
          return `list[${subType}]`;
        } else if (item.type === "dict") {
          if (!item.subType) {
            throw new Error(
              `Dict/Object type must specify its structure for field: ${item.key}`
            );
          }
          const subSchema = Object.entries(item.subType).reduce(
            (acc, [_, value]) => {
              acc[value.key] = processSchemaItem(value as SchemaItem);
              return acc;
            },
            {} as Record<string, string>
          );
          return `{${Object.entries(subSchema)
            .map(([k, v]) => `${k}: ${v}`)
            .join(", ")}}`;
        } else {
          return item.type;
        }
      };

      // If it's a sample operation with custom method, parse the samples as key-value pairs
      if (op.type === "sample" && op.otherKwargs?.method === "custom") {
        try {
          newOp.samples = JSON.parse(op.otherKwargs.samples);
        } catch (error) {
          console.warn(
            "Failed to parse custom samples as JSON, using raw value"
          );
        }
      }

      return {
        ...newOp,
        enable_observability: true,
        output: {
          schema: op.output.schema.reduce(
            (acc: Record<string, string>, item: SchemaItem) => {
              acc[item.key] = processSchemaItem(item);
              return acc;
            },
            {}
          ),
        },
      };
    })
    .filter((op) => op !== null);

  // Add check for empty operations
  if (updatedOperations.length === 0) {
    throw new Error("No valid operations found in pipeline configuration");
  }

  // Fetch all operations up until and including the operation_id
  const operationsToRun = operations.slice(
    0,
    operations.findIndex((op: Operation) => op.id === operation_id) + 1
  );

  const pipelineConfig = {
    datasets,
    default_model,
    optimizer_config: {
      force_decompose: true,
    },
    operations: updatedOperations,
    pipeline: {
      steps: [
        {
          name: "data_processing",
          input: Object.keys(datasets)[0], // Assuming the first dataset is the input
          operations: operationsToRun.map((op) => op.name),
        },
      ],
      output: {
        type: "file",
        path: path.join(
          homeDir,
          ".docetl",
          namespace,
          "pipelines",
          "outputs",
          `${name}.json`
        ),
        intermediate_dir: path.join(
          homeDir,
          ".docetl",
          namespace,
          "pipelines",
          name,
          "intermediates"
        ),
      },
    },
    system_prompt: {},
  };

  if (system_prompt) {
    if (system_prompt.datasetDescription) {
      // @ts-ignore
      pipelineConfig.system_prompt!.dataset_description =
        system_prompt.datasetDescription;
    }
    if (system_prompt.persona) {
      // @ts-ignore
      pipelineConfig.system_prompt!.persona = system_prompt.persona;
    }
  }

  // Get the inputPath from the intermediate_dir
  let inputPath;
  let outputPath;
  const prevOpIndex = operationsToRun.length - 2;
  const currentOpIndex = operationsToRun.length - 1;

  if (prevOpIndex >= 0) {
    const inputBase = pipelineConfig.pipeline.output.intermediate_dir;
    const opName = operationsToRun[prevOpIndex].name;
    inputPath = path.join(inputBase, "data_processing", opName + ".json");
  } else {
    // If there are no previous operations, use the dataset path
    inputPath = data.path;
  }

  const outputBase = pipelineConfig.pipeline.output.intermediate_dir;
  const outputOpName = operationsToRun[currentOpIndex].name;
  outputPath = path.join(outputBase, "data_processing", outputOpName + ".json");

  const yamlString = yaml.dump(pipelineConfig);

  console.log(yamlString);

  return {
    yamlString,
    inputPath,
    outputPath,
  };
}
