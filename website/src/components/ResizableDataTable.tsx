import React, { useState, useEffect, useCallback } from "react";
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
  ColumnResizeMode,
  ColumnDef,
  ColumnSizingState,
  Header,
  Row,
  getPaginationRowModel,
  VisibilityState,
} from "@tanstack/react-table";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight, ChevronDown } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { TABLE_SETTINGS_KEY } from "@/app/localStorageKeys";

export type DataType = Record<string, any>;
export type ColumnType<T extends DataType> = ColumnDef<T> & {
  initialWidth?: number;
};

const ColumnResizer = <T extends DataType>({
  header,
}: {
  header: Header<T, unknown>;
}) => {
  return (
    <div
      onMouseDown={header.getResizeHandler()}
      onTouchStart={header.getResizeHandler()}
      className={`absolute right-0 top-0 h-full w-2 cursor-col-resize bg-slate-400 opacity-0 hover:opacity-100`}
      style={{
        userSelect: "none",
        touchAction: "none",
      }}
    />
  );
};

interface ResizableRow<T extends DataType> extends Row<T> {
  getSize: () => number;
  setSize: (size: number) => void;
}

const RowResizer = <T extends DataType>({
  row,
  saveSettings,
}: {
  row: ResizableRow<T>;
  saveSettings: () => void;
}) => {
  return (
    <tr>
      <td colSpan={100}>
        <div
          onMouseDown={(e) => {
            e.preventDefault();
            const startY = e.clientY;
            const startHeight = row.getSize();

            const onMouseMove = (e: MouseEvent) => {
              const newHeight = Math.max(startHeight + e.clientY - startY, 30);
              row.setSize(newHeight);
            };

            const onMouseUp = () => {
              document.removeEventListener("mousemove", onMouseMove);
              document.removeEventListener("mouseup", onMouseUp);
            };

            document.addEventListener("mousemove", onMouseMove);
            document.addEventListener("mouseup", onMouseUp);
          }}
          className={`h-2 cursor-row-resize bg-slate-400 opacity-0 hover:opacity-100`}
          style={{
            userSelect: "none",
            touchAction: "none",
          }}
        />
      </td>
    </tr>
  );
};

interface ResizableDataTableProps<T extends DataType> {
  data: T[];
  columns: ColumnType<T>[];
  boldedColumns: string[];
  startingRowHeight?: number;
}

function ResizableDataTable<T extends DataType>({
  data,
  columns,
  boldedColumns,
  startingRowHeight = 60, // Default starting height
}: ResizableDataTableProps<T>) {
  const [columnSizing, setColumnSizing] = useState<ColumnSizingState>(() => {
    const savedSettings = localStorage.getItem(TABLE_SETTINGS_KEY);
    if (savedSettings) {
      const parsedSettings = JSON.parse(savedSettings);
      return parsedSettings.columnSizing || {};
    }
    const initialSizing: ColumnSizingState = {};
    columns.forEach((column) => {
      if (column.initialWidth) {
        initialSizing[column.id as string] = column.initialWidth;
      }
    });
    return initialSizing;
  });
  const [rowSizing, setRowSizing] = useState<Record<string, number>>(() => {
    const savedSettings = localStorage.getItem(TABLE_SETTINGS_KEY);
    if (savedSettings) {
      const parsedSettings = JSON.parse(savedSettings);
      return parsedSettings.rowSizing || {};
    }
    return {};
  });
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});

  const saveSettings = useCallback(() => {
    localStorage.setItem(
      TABLE_SETTINGS_KEY,
      JSON.stringify({ columnSizing, rowSizing }),
    );
  }, [columnSizing, rowSizing]);

  useEffect(() => {
    // Initialize row heights when data changes
    const initialRowSizing = data.reduce(
      (acc, _, index) => {
        acc[index] = startingRowHeight;
        return acc;
      },
      {} as Record<string, number>,
    );
    setRowSizing(initialRowSizing);

    // Initialize all columns as visible
    const initialColumnVisibility = columns.reduce((acc, column) => {
      acc[column.id as string] = true;
      return acc;
    }, {} as VisibilityState);
    setColumnVisibility(initialColumnVisibility);
  }, [data, startingRowHeight, columns]);

  // Add this before creating the table instance
  const sortedColumns = [...columns].sort((a, b) => {
    const aHeader = a.header as string;
    const bHeader = b.header as string;
    const aIsBold = boldedColumns.includes(aHeader);
    const bIsBold = boldedColumns.includes(bHeader);

    if (aIsBold && !bIsBold) return -1;
    if (!aIsBold && bIsBold) return 1;
    return 0;
  });

  const table = useReactTable({
    data,
    // Replace columns with sortedColumns here
    columns: sortedColumns,
    columnResizeMode: "onChange" as ColumnResizeMode,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onColumnSizingChange: (newColumnSizing) => {
      setColumnSizing(newColumnSizing);
      saveSettings();
    },
    onColumnVisibilityChange: setColumnVisibility,
    state: {
      columnSizing,
      columnVisibility,
    },
    enableColumnResizing: true,
    defaultColumn: {
      minSize: 30,
      size: 150,
      maxSize: Number.MAX_SAFE_INTEGER,
    },
    initialState: {
      pagination: {
        pageSize: 5,
      },
    },
  });

  return (
    <div className="w-full overflow-auto">
      <div className="mb-4 flex justify-between items-center">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="flex items-center">
              Show/Hide Columns
              <ChevronDown className="ml-2 h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="w-56">
            {table.getAllLeafColumns().map((column) => {
              return (
                <DropdownMenuCheckboxItem
                  key={column.id}
                  checked={column.getIsVisible()}
                  onCheckedChange={(value) => column.toggleVisibility(!!value)}
                >
                  {column.id}
                </DropdownMenuCheckboxItem>
              );
            })}
          </DropdownMenuContent>
        </DropdownMenu>
        <div className="flex items-center space-x-2">
          {data.length > 0 && (
            <div className="flex items-center justify-end space-x-2 py-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => table.previousPage()}
                disabled={!table.getCanPreviousPage()}
              >
                <ChevronLeft className="mr-2 h-4 w-4" /> Previous
              </Button>
              <span className="text-sm text-gray-600">
                Page {table.getState().pagination.pageIndex + 1} of{" "}
                {table.getPageCount()}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => table.nextPage()}
                disabled={!table.getCanNextPage()}
              >
                Next <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </div>
      <Table style={{ width: "max-content", minWidth: "100%" }}>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              <TableHead style={{ width: "30px" }}>#</TableHead>
              {headerGroup.headers.map((header) => (
                <TableHead
                  key={header.id}
                  style={{
                    width: header.getSize(),
                    position: "relative",
                    minWidth: `${header.column.columnDef.minSize}px`,
                    maxWidth: `${header.column.columnDef.maxSize}px`,
                    fontWeight: boldedColumns.includes(
                      header.column.columnDef.header as string,
                    )
                      ? "bold"
                      : "normal",
                  }}
                >
                  {header.isPlaceholder
                    ? null
                    : flexRender(
                        header.column.columnDef.header,
                        header.getContext(),
                      )}
                  <ColumnResizer header={header} />
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.map((row, index) => (
            <React.Fragment key={row.id}>
              <TableRow>
                <TableCell
                  style={{
                    width: "30px",
                    padding: "0.25rem",
                    textAlign: "center",
                  }}
                >
                  <span style={{ fontSize: "0.75rem", color: "#888" }}>
                    {table.getState().pagination.pageIndex *
                      table.getState().pagination.pageSize +
                      index +
                      1}
                  </span>
                </TableCell>
                {row.getVisibleCells().map((cell) => (
                  <TableCell
                    key={cell.id}
                    style={{
                      width: cell.column.getSize(),
                      minWidth: cell.column.columnDef.minSize,
                      height: `${rowSizing[index] || startingRowHeight}px`,
                      padding: "0",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        overflowY: "auto",
                        padding: "0.5rem",
                        fontWeight: "normal",
                      }}
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </div>
                  </TableCell>
                ))}
              </TableRow>
              <RowResizer
                row={{
                  ...row,
                  getSize: () => rowSizing[index] || startingRowHeight,
                  setSize: (size: number) => {
                    setRowSizing((prev) => {
                      const newRowSizing = { ...prev, [index]: size };
                      saveSettings();
                      return newRowSizing;
                    });
                  },
                }}
                saveSettings={saveSettings}
              />
            </React.Fragment>
          ))}
        </TableBody>
      </Table>
      {data.length > 0 && (
        <div className="flex items-center justify-end space-x-2 py-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            <ChevronLeft className="mr-2 h-4 w-4" /> Previous
          </Button>
          <span className="text-sm text-gray-600">
            Page {table.getState().pagination.pageIndex + 1} of{" "}
            {table.getPageCount()}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            Next <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}

export default ResizableDataTable;
