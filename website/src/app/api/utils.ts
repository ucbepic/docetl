import yaml from "js-yaml";
import path from "path";
import os from "os";
import { Operation, SchemaItem } from "@/app/types";

export function generatePipelineConfig(
  default_model: string,
  data: { path: string },
  operations: Operation[],
  operation_id: string,
  name: string,
  sample_size: number | null,
  optimize: boolean = false,
  clear_intermediate: boolean = false
) {
  const homeDir = os.homedir();

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
  const updatedOperations: Record<string, any> = operations.map(
    (op: Operation) => {
      const newOp: Record<string, any> = {
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
          console.log(item);
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
        } catch (e) {
          console.warn(
            "Failed to parse custom samples as JSON, using raw value"
          );
        }
      }

      return {
        ...newOp,
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
    }
  );

  // Fetch all operations up until and including the operation_id
  const operationsToRun = operations.slice(
    0,
    operations.findIndex((op: Operation) => op.id === operation_id) + 1
  );

  const pipelineConfig = {
    datasets,
    default_model,
    operations: updatedOperations,
    pipeline: {
      steps: [
        {
          name: "data_processing",
          input: Object.keys(datasets)[0], // Assuming the first dataset is the input
          operations: operationsToRun.map((op: any) => op.name),
        },
      ],
      output: {
        type: "file",
        path: path.join(
          homeDir,
          ".docetl",
          "pipelines",
          "outputs",
          `${name}.json`
        ),
        intermediate_dir: path.join(
          homeDir,
          ".docetl",
          "pipelines",
          name,
          "intermediates"
        ),
      },
    },
  };

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

  return {
    yamlString,
    inputPath,
    outputPath,
  };
}
