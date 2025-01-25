import { SchemaItem, SchemaType } from "../app/types";

export const schemaDictToItemSet = (
  schema: Record<string, string>
): SchemaItem[] => {
  return Object.entries(schema).map(([key, type]): SchemaItem => {
    if (typeof type === "string") {
      if (type.startsWith("list[")) {
        const subType = type.slice(5, -1);
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
          const subSchema = JSON.parse(type);
          return {
            key,
            type: "dict",
            subType: schemaDictToItemSet(subSchema),
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
