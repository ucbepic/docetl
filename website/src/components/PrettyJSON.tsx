import React, { useState } from "react";
import { ChevronDown } from "lucide-react";

interface PrettyJSONProps {
  data: unknown;
}

export const PrettyJSON = React.memo(({ data }: PrettyJSONProps) => {
  const [expandedPaths, setExpandedPaths] = useState<string[]>(() => {
    const paths: string[] = [];

    const collectPaths = (value: unknown, path: string = "") => {
      if (Array.isArray(value)) {
        paths.push(path);
        value.forEach((item, index) => {
          if (typeof item === "object" && item !== null) {
            collectPaths(item, `${path}.${index}`);
          }
        });
      } else if (typeof value === "object" && value !== null) {
        paths.push(path);
        Object.entries(value).forEach(([key, val]) => {
          if (typeof val === "object" && val !== null) {
            collectPaths(val, `${path}.${key}`);
          }
        });
      }
    };

    collectPaths(data);
    return paths;
  });

  const renderValue = (
    value: unknown,
    path: string = "",
    level: number = 0
  ): React.ReactNode => {
    if (value === null)
      return <span className="text-red-600 font-bold">null</span>;
    if (value === undefined)
      return <span className="text-red-600 font-bold">undefined</span>;

    if (Array.isArray(value)) {
      if (value.length === 0)
        return <span className="text-slate-700 font-bold">[]</span>;

      const isExpanded = expandedPaths.includes(path);
      return (
        <div className="relative">
          <button
            onClick={() => {
              setExpandedPaths((prev) =>
                prev.includes(path)
                  ? prev.filter((p) => p !== path)
                  : [...prev, path]
              );
            }}
            className="text-left hover:text-primary transition-colors inline-flex items-center gap-1 font-bold"
          >
            <ChevronDown
              className={`h-4 w-4 transition-transform ${
                isExpanded ? "rotate-180" : ""
              }`}
            />
            <span className="text-slate-900">Array ({value.length} items)</span>
          </button>
          {isExpanded && (
            <div className="ml-4 border-l pl-4 mt-2 space-y-2">
              {value.map((item, index) => (
                <div key={index} className="relative">
                  <span className="text-slate-600 font-bold">{index}: </span>
                  {renderValue(item, `${path}.${index}`, level + 1)}
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    if (typeof value === "object") {
      const entries = Object.entries(value as Record<string, unknown>);
      if (entries.length === 0)
        return <span className="text-slate-700 font-bold">{}</span>;

      const isExpanded = expandedPaths.includes(path);
      return (
        <div className="relative">
          <button
            onClick={() => {
              setExpandedPaths((prev) =>
                prev.includes(path)
                  ? prev.filter((p) => p !== path)
                  : [...prev, path]
              );
            }}
            className="text-left hover:text-primary transition-colors inline-flex items-center gap-1 font-bold"
          >
            <ChevronDown
              className={`h-4 w-4 transition-transform ${
                isExpanded ? "rotate-180" : ""
              }`}
            />
            <span className="text-slate-900">
              Object ({entries.length} properties)
            </span>
          </button>
          {isExpanded && (
            <div className="ml-4 border-l pl-4 mt-2 space-y-2">
              {entries.map(([key, val]) => (
                <div key={key} className="relative">
                  <span className="text-slate-900 font-bold">{key}: </span>
                  {renderValue(val, `${path}.${key}`, level + 1)}
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    if (typeof value === "string") {
      return <span className="text-emerald-600 font-bold">"{value}"</span>;
    }

    if (typeof value === "number") {
      return <span className="text-blue-700 font-bold">{value}</span>;
    }

    if (typeof value === "boolean") {
      return (
        <span className="text-purple-700 font-bold">{value.toString()}</span>
      );
    }

    return <span className="text-slate-900 font-bold">{String(value)}</span>;
  };

  return (
    <div className="font-mono text-sm space-y-1.5">{renderValue(data)}</div>
  );
});

PrettyJSON.displayName = "PrettyJSON";
