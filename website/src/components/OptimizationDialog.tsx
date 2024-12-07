import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
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
import { ChevronLeft, ChevronRight } from "lucide-react";

interface OptimizationDialogProps {
  isOpen: boolean;
  content: string;
  prompt?: string;
  operationName?: string;
  inputData?: Array<Record<string, unknown>>;
  outputData?: Array<Record<string, unknown>>;
  onOpenChange: (open: boolean) => void;
  onDecompose?: () => void;
}

export const OptimizationDialog: React.FC<OptimizationDialogProps> = ({
  isOpen,
  content,
  prompt,
  inputData,
  outputData,
  operationName,
  onOpenChange,
  onDecompose,
}) => {
  const [currentPage, setCurrentPage] = React.useState(1);
  const rowsPerPage = 1;

  const shouldShowInputData = React.useMemo(() => {
    if (!inputData?.length || !outputData?.length) return true;

    const inputKeys = new Set(Object.keys(inputData[0]));
    const outputKeys = new Set(Object.keys(outputData[0]));

    return Array.from(inputKeys).some((key) => !outputKeys.has(key));
  }, [inputData, outputData]);

  const renderTable = (data: Array<Record<string, unknown>>) => {
    if (!data.length) return null;
    const columns = Object.keys(data[0]).filter(
      (column) => !column.startsWith("_observability")
    );

    const totalPages = Math.ceil(data.length / rowsPerPage);
    const startIndex = (currentPage - 1) * rowsPerPage;
    const paginatedData = data.slice(startIndex, startIndex + rowsPerPage);

    return (
      <div className="space-y-2">
        <div className="border rounded-md">
          <div className="max-h-[300px] overflow-auto">
            <Table className="relative w-full border-collapse">
              <TableHeader>
                <TableRow className="sticky top-0 bg-background z-10 border-b">
                  {columns.map((column) => (
                    <TableHead
                      key={column}
                      className="h-10 px-4 text-left align-middle bg-background"
                    >
                      {column}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {paginatedData.map((row, rowIndex) => (
                  <TableRow key={rowIndex}>
                    {columns.map((column) => (
                      <TableCell key={column} className="p-2 align-top">
                        <pre className="whitespace-pre-wrap font-mono text-sm text-left">
                          {typeof row[column] === "object"
                            ? JSON.stringify(row[column], null, 2)
                            : String(row[column])}
                        </pre>
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>

        <div className="flex items-center justify-between px-2">
          <div className="text-sm">
            Row {startIndex + 1} of {data.length}
          </div>
          <div className="flex items-center space-x-1">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
              className="px-2 py-1"
            >
              Previous
              <ChevronLeft className="h-4 w-4 ml-1" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setCurrentPage((prev) => Math.min(prev + 1, totalPages))
              }
              disabled={currentPage === totalPages}
              className="px-2 py-1"
            >
              Next
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
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

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl max-h-[90vh] flex flex-col">
        <DialogHeader className="flex-shrink-0 border-b pb-4">
          <DialogTitle className="text-xl">Operation Too Complex</DialogTitle>
          <p className="text-base mt-2">
            This operation might be too complex for the LLM to handle
            efficiently. We recommend breaking it down into smaller, more
            manageable steps.
          </p>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto py-4">
          {(operationName || prompt) && (
            <div className="p-4 bg-muted rounded-lg space-y-2 mb-6">
              {operationName && (
                <div className="flex items-center">
                  <span className="font-medium text-base mr-2">
                    Current Operation:
                  </span>
                  <span className="bg-primary/15 text-primary rounded-md px-3 py-1 text-base">
                    {operationName}
                  </span>
                </div>
              )}
              {prompt && (
                <div className="space-y-2">
                  <h3 className="font-medium text-base">Original Prompt:</h3>
                  <div className="bg-background rounded-lg p-3 text-base font-mono leading-relaxed">
                    {prompt}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="space-y-6">
            {shouldShowInputData && inputData && (
              <section className="space-y-3">
                <h3 className="text-base font-medium">Input Data Sample</h3>
                <div className="overflow-auto">{renderTable(inputData)}</div>
              </section>
            )}
            {outputData && (
              <section className="space-y-3">
                <h3 className="text-base font-medium">Sample Output</h3>
                <div className="overflow-auto">{renderTable(outputData)}</div>
              </section>
            )}
            <section className="space-y-3">
              <h3 className="text-base font-medium">Suggested Improvements</h3>
              <div className="whitespace-pre-wrap text-base leading-relaxed">
                {content}
              </div>
            </section>
          </div>
        </div>

        <div className="flex justify-end items-center gap-3 pt-4 border-t mt-4">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Ignore
          </Button>
          <Button
            onClick={() => {
              onDecompose?.();
              onOpenChange(false);
            }}
          >
            Automatically Decompose
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

OptimizationDialog.displayName = "OptimizationDialog";
