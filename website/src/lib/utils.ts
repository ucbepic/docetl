import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { Operation } from "@/app/types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function canBeOptimized(operationType: string) {
  return ["resolve", "map", "reduce", "filter"].includes(operationType);
}
