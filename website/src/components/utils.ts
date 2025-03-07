import { SchemaItem, SchemaType } from "../app/types";

export const schemaDictToItemSet = (
  schema: Record<string, string>
): SchemaItem[] => {
  return Object.entries(schema).map(([key, type]): SchemaItem => {
    if (typeof type === "string") {
      if (type.startsWith("list[")) {
        const subType = type.slice(5, -1);

        // Handle objects inside lists
        if (subType.startsWith("{") && subType.endsWith("}")) {
          try {
            // Extract the object definition from between curly braces
            const objectContent = subType.slice(1, -1);

            // Convert "key: value" pairs to a proper JSON object format
            const objectPairs = objectContent
              .split(",")
              .map((pair) => pair.trim());
            const objectSchema: Record<string, string> = {};

            for (const pair of objectPairs) {
              const [pairKey, pairValue] = pair.split(":").map((p) => p.trim());
              if (pairKey && pairValue) {
                objectSchema[pairKey] = pairValue;
              }
            }

            // Recursively process the object schema
            return {
              key,
              type: "list",
              subType: {
                key: "0",
                type: "dict",
                subType: schemaDictToItemSet(objectSchema),
              },
            };
          } catch (error) {
            console.error(
              `Error parsing object inside list for ${key}:`,
              error
            );
            return { key, type: "list", subType: { key: "0", type: "string" } };
          }
        }

        return {
          key,
          type: "list",
          subType: { key: "0", type: subType as SchemaType },
        };
      } else if (type.startsWith("enum[") && type.endsWith("]")) {
        const enumValuesStr = type.slice(5, -1);
        const enumValues = enumValuesStr
          .split(",")
          .map((v) => v.trim())
          .filter((v) => v);
        if (enumValues.length < 2) {
          console.error(
            `Invalid enum values for ${key}: ${type}. Using string type instead.`
          );
          return { key, type: "string" };
        }
        return {
          key,
          type: "enum",
          enumValues,
        };
      } else if (type.startsWith("{") && type.endsWith("}")) {
        try {
          // Extract the object definition from between curly braces
          const objectContent = type.slice(1, -1);

          // Convert "key: value" pairs to a proper JSON object format
          const objectPairs = objectContent
            .split(",")
            .map((pair) => pair.trim());
          const objectSchema: Record<string, string> = {};

          for (const pair of objectPairs) {
            const [pairKey, pairValue] = pair.split(":").map((p) => p.trim());
            if (pairKey && pairValue) {
              objectSchema[pairKey] = pairValue;
            }
          }

          return {
            key,
            type: "dict",
            subType: schemaDictToItemSet(objectSchema),
          };
        } catch (error) {
          console.error(`Error parsing dict schema for ${key}:`, error);
          return { key, type: "string" };
        }
      } else {
        return { key, type: type as SchemaType };
      }
    } else {
      return { key, type: "string" };
    }
  });
};
