import React, { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

const CollapsibleCode = ({ code }: { code: string }) => {
  const [expandedSections, setExpandedSections] = useState<
    Record<number, boolean>
  >({});

  const colorMap: Record<string, string> = {
    "# Blue": "bg-blue-100",
    "# Green": "bg-green-100",
    "# Yellow": "bg-yellow-100",
    "# Orange": "bg-orange-100",
  };

  const toggleSection = (index: number) => {
    setExpandedSections((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  const renderOperation = (operation: string, index: number) => {
    const color = Object.entries(colorMap).find(([key]) =>
      operation.includes(key),
    );
    const isExpanded = expandedSections[index];
    const name = operation.split("\n")[0].split(":")[1].trim();

    return (
      <div
        key={index}
        className={`border-b border-gray-200 ${color ? color[1] : ""}`}
      >
        <button
          onClick={() => toggleSection(index)}
          className="w-full text-left py-2 px-4 flex justify-between items-center hover:bg-gray-50"
        >
          <span>{name.replace(/# (Blue|Green|Yellow|Red)/g, "").trim()}</span>
          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
        {isExpanded && (
          <pre className="p-4 bg-gray-100">
            <code>{operation}</code>
          </pre>
        )}
      </div>
    );
  };

  const renderNonCollapsible = (section: string) => (
    <pre className="p-4 bg-gray-100">
      <code>{section}</code>
    </pre>
  );

  const [preOperations, rest] = code.split(/(?=^operations:)/m);
  const [operations, postOperations] = rest.split(/(?=pipeline:)/);
  const operationsList = operations.split(/(?= {2}- name:)/).slice(1); // Remove the "operations:" line

  return (
    <div className="text-sm overflow-x-auto bg-gray-100 rounded-md text-left">
      {renderNonCollapsible(preOperations)}
      <pre className="p-4 bg-gray-100">
        <code>operations:</code>
      </pre>
      <div className="bg-white">
        {operationsList.map((operation, index) =>
          renderOperation(operation, index),
        )}
      </div>
      {renderNonCollapsible(postOperations)}
    </div>
  );
};

export default CollapsibleCode;
