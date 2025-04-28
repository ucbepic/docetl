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

export const DOCWRANGLER_HOSTED_COST_LIMIT = 10;

export const isDocWranglerHosted = () => {
  const backendHost = process.env.NEXT_PUBLIC_BACKEND_HOST || "";
  const isHostedVar = process.env.NEXT_PUBLIC_HOSTED_DOCWRANGLER || "false";
  return backendHost.includes("modal.run") || isHostedVar === "true";
};
