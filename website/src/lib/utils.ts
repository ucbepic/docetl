import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function canBeOptimized(operationType: string) {
  return ["resolve", "map", "reduce", "filter"].includes(operationType);
}

export const generateId = () => {
  return Math.random().toString(36).substr(2, 9);
};
