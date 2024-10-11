import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import os from 'os';
import path from 'path';
import { Operation, SchemaItem } from '@/app/types';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
