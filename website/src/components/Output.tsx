import React from 'react';
import { ColumnType } from '@/components/ResizableDataTable';
import ResizableDataTable from '@/components/ResizableDataTable';
import { usePipelineContext } from '@/contexts/PipelineContext';
import { Loader2, Maximize2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogTrigger,
} from "@/components/ui/dialog";
import BookmarkableText from '@/components/BookmarkableText';

export const Output: React.FC = () => {
  const { outputs, isLoadingOutputs } = usePipelineContext();

  React.useEffect(() => {
    if (outputs.length > 0 && 'date' in outputs[0]) {
      outputs.sort((a, b) => {
        const dateA = (a as any).date;
        const dateB = (b as any).date;
        if (dateA && dateB) {
          return new Date(dateB).getTime() - new Date(dateA).getTime();
        }
        return 0;
      });
    }
  }, [outputs]);

  const columns: ColumnType<any>[] = React.useMemo(() => 
    outputs.length > 0
      ? Object.keys(outputs[0]).map((key) => ({
          accessorKey: key,
          header: key,
          cell: ({ getValue }) => {
            const value = getValue();
            const stringValue = typeof value === 'object' && value !== null ? JSON.stringify(value, null, 2) : String(value);
            return (
              <pre className="whitespace-pre-wrap font-mono text-sm">{stringValue}</pre>
            );
          },
        }))
      : [],
    [outputs]
  );
  
  const OutputContent = () => (
    <div className="flex-grow overflow-y-auto">
      {isLoadingOutputs ? (
        <div className="flex items-center justify-center h-full">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
          <span className="ml-2 text-muted-foreground">Loading outputs...</span>
        </div>
      ) : outputs.length > 0 ? (
        <BookmarkableText source="output">
        <ResizableDataTable
          data={outputs}
          columns={columns}
          startingRowHeight={180}
        />
        </BookmarkableText>
      ) : (
        <p className="text-center text-muted-foreground">No outputs available.</p>
      )}
    </div>
  );

  return (
    <div className="h-full p-4 bg-white flex flex-col">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-sm font-bold mb-2 flex items-center uppercase">
          Output
        </h2>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="ghost" size="sm" className="p-1">
              <Maximize2 size={16} />
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-[90vw] w-[90vw] max-h-[90vh] h-[90vh] flex flex-col pointer-events-auto bg-white">
          <div className="spotlight-overlay-target">
            <h2 className="text-lg font-bold mb-2">Output</h2>
            <OutputContent />
          </div>
          </DialogContent>
        </Dialog>
      </div>
      <OutputContent />
    </div>
  );
};
