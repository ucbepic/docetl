import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight, Check, X } from "lucide-react";
import { DecomposeResult } from "@/app/types";

interface DecompositionComparisonDialogProps {
  isOpen: boolean;
  result: DecomposeResult | null;
  operationName?: string;
  onOpenChange: (open: boolean) => void;
  onApply: () => void;
  onCancel: () => void;
}

export const DecompositionComparisonDialog: React.FC<
  DecompositionComparisonDialogProps
> = ({ isOpen, result, operationName, onOpenChange, onApply, onCancel }) => {
  const [currentPage, setCurrentPage] = React.useState(1);

  const originalOutputs = result?.original_outputs || [];
  const decomposedOutputs = result?.decomposed_outputs || [];
  const maxRows = Math.max(originalOutputs.length, decomposedOutputs.length);

  const renderOutputTable = (
    data: Array<Record<string, unknown>>,
    title: string
  ) => {
    if (!data.length) {
      return (
        <div className="text-center text-muted-foreground py-8">
          No outputs available
        </div>
      );
    }

    const columns = Object.keys(data[0]).filter(
      (column) => !column.startsWith("_observability")
    );

    const currentRow = data[currentPage - 1];
    if (!currentRow) return null;

    return (
      <div className="space-y-2">
        <h4 className="font-medium text-sm text-muted-foreground">{title}</h4>
        <div className="border rounded-md">
          <div className="max-h-[250px] overflow-auto">
            <Table className="relative w-full border-collapse">
              <TableHeader>
                <TableRow className="sticky top-0 bg-background z-10 border-b">
                  {columns.map((column) => (
                    <TableHead
                      key={column}
                      className="h-8 px-3 text-left align-middle bg-background text-xs"
                    >
                      {column}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow>
                  {columns.map((column) => (
                    <TableCell key={column} className="p-2 align-top">
                      <pre className="whitespace-pre-wrap font-mono text-xs text-left max-w-[300px]">
                        {typeof currentRow[column] === "object"
                          ? JSON.stringify(currentRow[column], null, 2)
                          : String(currentRow[column] ?? "")}
                      </pre>
                    </TableCell>
                  ))}
                </TableRow>
              </TableBody>
            </Table>
          </div>
        </div>
      </div>
    );
  };

  React.useEffect(() => {
    if (isOpen) {
      setCurrentPage(1);
    }
  }, [isOpen]);

  if (!result) return null;

  const isOriginalWinner = result.winning_directive === "original";

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-7xl max-h-[90vh] flex flex-col">
        <DialogHeader className="flex-shrink-0 border-b pb-4">
          <DialogTitle className="text-xl">
            Review Decomposition Results
          </DialogTitle>
          <DialogDescription>
            Compare the original operation outputs with the decomposed version
            before applying changes.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto py-4 space-y-6">
          {/* Summary */}
          <div className="p-4 bg-muted rounded-lg space-y-2">
            <div className="flex items-center gap-4 flex-wrap">
              {operationName && (
                <div className="flex items-center">
                  <span className="font-medium text-sm mr-2">Operation:</span>
                  <span className="bg-primary/15 text-primary rounded-md px-2 py-1 text-sm">
                    {operationName}
                  </span>
                </div>
              )}
              <div className="flex items-center">
                <span className="font-medium text-sm mr-2">
                  Winning Strategy:
                </span>
                <span
                  className={`rounded-md px-2 py-1 text-sm ${
                    isOriginalWinner
                      ? "bg-yellow-100 text-yellow-800"
                      : "bg-green-100 text-green-800"
                  }`}
                >
                  {result.winning_directive}
                </span>
              </div>
              <div className="flex items-center">
                <span className="font-medium text-sm mr-2">
                  Candidates Evaluated:
                </span>
                <span className="text-sm">{result.candidates_evaluated}</span>
              </div>
              {result.cost && (
                <div className="flex items-center">
                  <span className="font-medium text-sm mr-2">Cost:</span>
                  <span className="text-sm">${result.cost.toFixed(4)}</span>
                </div>
              )}
            </div>
          </div>

          {/* Comparison Rationale */}
          {result.comparison_rationale && (
            <div className="space-y-2">
              <h3 className="font-medium text-base">Why This Was Chosen</h3>
              <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm">
                {result.comparison_rationale}
              </div>
            </div>
          )}

          {/* Side by side comparison */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-base">
                Output Comparison (Sample {currentPage} of {maxRows || 1})
              </h3>
              <div className="flex items-center space-x-1">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setCurrentPage((prev) => Math.max(prev - 1, 1))
                  }
                  disabled={currentPage === 1}
                  className="px-2 py-1"
                >
                  <ChevronLeft className="h-4 w-4" />
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setCurrentPage((prev) => Math.min(prev + 1, maxRows))
                  }
                  disabled={currentPage === maxRows || maxRows === 0}
                  className="px-2 py-1"
                >
                  Next
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {renderOutputTable(originalOutputs, "Original Output")}
              {renderOutputTable(
                decomposedOutputs,
                `Decomposed Output (${result.winning_directive})`
              )}
            </div>
          </div>

          {/* Decomposed Operations Preview */}
          {result.decomposed_operations &&
            result.decomposed_operations.length > 0 &&
            !isOriginalWinner && (
              <div className="space-y-2">
                <h3 className="font-medium text-base">
                  New Operations ({result.decomposed_operations.length})
                </h3>
                <div className="space-y-2">
                  {result.decomposed_operations.map((op, idx) => (
                    <div
                      key={idx}
                      className="p-3 border rounded-lg bg-muted/50"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-medium text-sm">
                          {(op.name as string) || `Operation ${idx + 1}`}
                        </span>
                        <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">
                          {op.type as string}
                        </span>
                      </div>
                      {op.prompt && (
                        <pre className="text-xs font-mono bg-background p-2 rounded border max-h-[100px] overflow-auto">
                          {String(op.prompt).slice(0, 500)}
                          {String(op.prompt).length > 500 ? "..." : ""}
                        </pre>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
        </div>

        <div className="flex justify-end items-center gap-3 pt-4 border-t mt-4">
          <Button variant="outline" onClick={onCancel}>
            <X className="h-4 w-4 mr-2" />
            Keep Original
          </Button>
          <Button
            onClick={onApply}
            disabled={isOriginalWinner}
            title={
              isOriginalWinner
                ? "Original was determined to be the best option"
                : undefined
            }
          >
            <Check className="h-4 w-4 mr-2" />
            Apply Decomposition
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

DecompositionComparisonDialog.displayName = "DecompositionComparisonDialog";
