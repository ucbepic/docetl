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
}

export const OptimizationDialog: React.FC<OptimizationDialogProps> = ({
  isOpen,
  content,
  prompt,
  inputData,
  outputData,
  operationName,
  onOpenChange,
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
    const columns = Object.keys(data[0]);

    const totalPages = Math.ceil(data.length / rowsPerPage);
    const startIndex = (currentPage - 1) * rowsPerPage;
    const paginatedData = data.slice(startIndex, startIndex + rowsPerPage);

    return (
      <div className="space-y-2">
        <div className="border">
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
          <div className="text-sm text-muted-foreground">
            Row {startIndex + 1} of {data.length}
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setCurrentPage((prev) => Math.min(prev + 1, totalPages))
              }
              disabled={currentPage === totalPages}
            >
              <ChevronRight className="h-4 w-4" />
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
        <DialogHeader className="flex-shrink-0">
          <DialogTitle>Decomposition Suggestions</DialogTitle>
          <p className="text-sm text-muted-foreground">
            We&apos;ve detected that the operation you&apos;re trying to run
            might be too complex for the LLM. Consider breaking it down into
            smaller operations. You can use our decomposition tool (lightning
            button) to do this.
          </p>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto">
          {(operationName || prompt) && (
            <div className="p-3 bg-muted/50 rounded-lg space-y-1">
              {operationName && (
                <div className="flex items-center">
                  <span className="font-semibold text-sm text-muted-foreground mr-2">
                    Operation:
                  </span>
                  <span className="bg-primary/15 text-primary hover:bg-primary/20 transition-colors rounded-md px-2 py-0.5 text-sm font-medium">
                    {operationName}
                  </span>
                </div>
              )}
              {prompt && (
                <div className="space-y-1">
                  <h3 className="font-semibold text-sm text-muted-foreground">
                    Prompt:
                  </h3>
                  <div className="bg-background/60 hover:bg-background/80 transition-colors duration-200 rounded-lg p-2.5 text-sm whitespace-pre-wrap leading-relaxed font-mono">
                    {prompt}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="mt-4 space-y-4">
            {shouldShowInputData && inputData && (
              <div className="space-y-2">
                <h3 className="text-sm font-bold uppercase">
                  Sample Input Data
                </h3>
                <div className="overflow-auto">{renderTable(inputData)}</div>
              </div>
            )}
            {outputData && (
              <div className="space-y-2">
                <h3 className="text-sm font-bold uppercase">
                  Sample Output Data
                </h3>
                <div className="overflow-auto">{renderTable(outputData)}</div>
              </div>
            )}
          </div>

          <div className="mt-4 whitespace-pre-wrap">{content}</div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

OptimizationDialog.displayName = "OptimizationDialog";
