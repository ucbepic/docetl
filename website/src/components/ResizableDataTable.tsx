import React, {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
  memo,
} from "react";
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
  SortingState,
  getSortedRowModel,
  getFilteredRowModel,
  FilterFn,
  ColumnFiltersState,
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
import {
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Search,
  Eye,
  Maximize2,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { TABLE_SETTINGS_KEY } from "@/app/localStorageKeys";
import ReactMarkdown from "react-markdown";
import debounce from "lodash/debounce";
import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Input } from "@/components/ui/input";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { ColumnDialog } from "@/components/ColumnDialog";
import { SearchableCell } from "@/components/SearchableCell";
import { PrettyJSON } from "@/components/PrettyJSON";
export type DataType = Record<string, unknown>;
export type ColumnType<T> = {
  accessorKey: string;
  header: string;
  cell?: ({ getValue }: { getValue: () => unknown }) => React.ReactNode;
  initialWidth?: number;
  id?: string;
};

export interface ColumnStats {
  min: number;
  max: number;
  avg: number;
  distribution: number[];
  bucketSize: number;
  type: "number" | "array" | "string-words" | "string-chars" | "boolean";
  distinctCount: number;
  totalCount: number;
  isLowCardinality: boolean;
  sortedValueCounts: { value: string; count: number }[];
}

function calculateDistinctValueCounts(
  data: Record<string, unknown>[],
  accessor: string
): Map<string | number | boolean, number> {
  const valueCounts = new Map<string | number | boolean, number>();

  data.forEach((row) => {
    const value = row[accessor];
    if (value != null) {
      const key =
        typeof value === "object" ? JSON.stringify(value) : String(value);
      valueCounts.set(
        key as string | number | boolean,
        (valueCounts.get(key as string | number | boolean) || 0) + 1
      );
    }
  });

  return valueCounts;
}

export function calculateColumnStats(
  data: Record<string, unknown>[],
  accessor: string
): ColumnStats | null {
  let type: ColumnStats["type"] = "string-words";
  const firstValue = data.find((row) => row[accessor] != null)?.[accessor];

  // Determine type based on first non-null value
  if (typeof firstValue === "number") type = "number";
  if (Array.isArray(firstValue)) type = "array";
  if (typeof firstValue === "boolean") type = "boolean";

  // For strings, check first 5 non-null values to determine if we should count chars
  if (typeof firstValue === "string") {
    const first5Values = data
      .filter(
        (row) => typeof row[accessor] === "string" && row[accessor] != null
      )
      .slice(0, 5)
      .map((row) => row[accessor] as string);

    // Use char count if all sampled values are single words
    type =
      first5Values.length > 0 &&
      first5Values.every((val) => !/\s/.test(val.trim()))
        ? "string-chars"
        : "string-words";
  }

  const values = data
    .map((row) => {
      const value = row[accessor];

      if (value === null || value === undefined) {
        return null;
      }

      // For booleans, convert to 0 or 1
      if (typeof value === "boolean") {
        return value ? 1 : 0;
      }

      // For numbers, use the value directly
      if (typeof value === "number") {
        return value;
      }

      // For arrays, count number of elements
      if (Array.isArray(value)) {
        return value.length;
      }

      // For strings, count either chars or words based on earlier determination
      if (typeof value === "string") {
        const trimmedValue = value.trim();
        return type === "string-chars"
          ? trimmedValue.length
          : trimmedValue.split(/\s+/).length;
      }

      // For other types, convert to string and count words
      const stringValue = JSON.stringify(value);
      return stringValue.split(/\s+/).length;
    })
    .filter((length): length is number => length !== null);

  if (values.length === 0) return null;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const avg = values.reduce((sum, val) => sum + val, 0) / values.length;

  const valueCounts = calculateDistinctValueCounts(data, accessor);
  const distinctCount = valueCounts.size;
  const totalCount = data.filter((row) => row[accessor] != null).length;
  const isLowCardinality = distinctCount < totalCount * 0.5;

  // Convert value counts to sorted array for bar chart
  const sortedValueCounts = Array.from(valueCounts.entries())
    .sort((a, b) => b[1] - a[1]) // Sort by count in descending order
    .map(([value, count]) => ({
      value: String(value),
      count,
    }));

  // For boolean values, create a special two-bucket distribution
  if (type === "boolean") {
    const distribution = [0, 0]; // [false count, true count]
    values.forEach((value) => {
      distribution[value]++;
    });
    return {
      min,
      max,
      avg,
      distribution,
      bucketSize: 1,
      type,
      distinctCount,
      totalCount,
      isLowCardinality,
      sortedValueCounts,
    };
  }

  // Special handling for single distinct value
  if (min === max) {
    return {
      min,
      max,
      avg,
      distribution: [values.length], // Put all values in a single bucket
      bucketSize: 1,
      type,
      distinctCount,
      totalCount,
      isLowCardinality,
      sortedValueCounts,
    };
  }

  // For numbers, use more precise bucketing
  const bucketSize =
    type === "number" ? (max - min) / 7 : Math.ceil((max - min) / 7);

  const distribution = new Array(7).fill(0);

  values.forEach((value) => {
    const bucketIndex = Math.min(
      Math.floor((value - min) / bucketSize),
      distribution.length - 1
    );
    distribution[bucketIndex]++;
  });

  return {
    min,
    max,
    avg,
    distribution,
    bucketSize,
    type,
    distinctCount,
    totalCount,
    isLowCardinality,
    sortedValueCounts,
  };
}

const truncateString = (str: string, maxLength: number = 20) => {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength) + "...";
};

export const WordCountHistogram = memo(
  ({
    histogramData,
    height = 60,
  }: {
    histogramData: { range: string; count: number; fullRange: string }[];
    height?: number;
  }) => {
    // Memoize total count calculation
    const totalCount = useMemo(
      () => histogramData.reduce((sum, item) => sum + item.count, 0),
      [histogramData]
    );

    // Memoize truncated data
    const truncatedData = useMemo(
      () =>
        histogramData.map((item) => ({
          ...item,
          range: truncateString(item.range, 10),
          fullRange: item.fullRange,
        })),
      [histogramData]
    );

    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={truncatedData} barCategoryGap={1}>
          <XAxis
            dataKey="range"
            tick={{ fontSize: 8 }}
            interval={2}
            tickLine={false}
            stroke="hsl(var(--muted-foreground))"
            height={15}
          />
          <Tooltip
            formatter={(value: number) => [
              `${value.toLocaleString()} (${(
                (value / totalCount) *
                100
              ).toFixed(1)}%)`,
              "Count",
            ]}
            labelFormatter={(_, payload) =>
              payload[0]?.payload?.fullRange || ""
            }
            contentStyle={{
              backgroundColor: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "var(--radius)",
              color: "hsl(var(--popover-foreground))",
              padding: "8px 12px",
              boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
            }}
            wrapperStyle={{ zIndex: 1000 }}
          />
          <Bar
            dataKey="count"
            fill="hsl(var(--chart-2))"
            radius={[4, 4, 0, 0]}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    );
  },
  // Deep comparison for histogramData
  (prevProps, nextProps) =>
    JSON.stringify(prevProps.histogramData) ===
    JSON.stringify(nextProps.histogramData)
);
WordCountHistogram.displayName = "WordCountHistogram";

export const CategoricalBarChart = memo(
  ({
    data,
    height = 60,
  }: {
    data: { value: string; count: number }[];
    height?: number;
  }) => {
    // Memoize total count calculation
    const totalCount = useMemo(
      () => data.reduce((sum, item) => sum + item.count, 0),
      [data]
    );

    // Memoize truncated and limited data
    const displayData = useMemo(() => {
      return data.slice(0, 10).map((item) => ({
        ...item,
        displayValue: truncateString(item.value, 10),
        fullValue: item.value,
      }));
    }, [data]);

    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={displayData} barCategoryGap={1}>
          <XAxis
            dataKey="displayValue"
            tick={{ fontSize: 8 }}
            interval={0}
            tickLine={false}
            stroke="hsl(var(--muted-foreground))"
            height={15}
          />
          <Tooltip
            formatter={(value: number) => [
              `${value.toLocaleString()} (${(
                (value / totalCount) *
                100
              ).toFixed(1)}%)`,
              "Count",
            ]}
            labelFormatter={(_, payload) =>
              payload[0]?.payload?.fullValue || ""
            }
            contentStyle={{
              backgroundColor: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "var(--radius)",
              color: "hsl(var(--popover-foreground))",
              padding: "8px 12px",
              boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
            }}
            wrapperStyle={{ zIndex: 1000 }}
          />
          <Bar
            dataKey="count"
            fill="hsl(var(--chart-2))"
            radius={[4, 4, 0, 0]}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    );
  },
  // Deep comparison for data
  (prevProps, nextProps) =>
    JSON.stringify(prevProps.data) === JSON.stringify(nextProps.data)
);
CategoricalBarChart.displayName = "CategoricalBarChart";

interface ColumnHeaderProps {
  header: string;
  stats: ColumnStats | null;
  isBold: boolean;
  onFilter: (value: string) => void;
  filterValue: string;
  onSort: () => void;
  sortDirection: false | "asc" | "desc";
  onExpand: () => void;
}

const ColumnHeader = memo(
  ({
    header,
    stats,
    isBold,
    onFilter,
    filterValue,
    onSort,
    sortDirection,
    onExpand,
  }: ColumnHeaderProps) => {
    const histogramData = useMemo(() => {
      if (!stats) return [];

      const getUnitLabel = () => {
        switch (stats.type) {
          case "number":
            return "";
          case "array":
            return " items";
          case "boolean":
            return "";
          case "string-chars":
            return " chars";
          case "string-words":
            return " words";
          default:
            return " words";
        }
      };

      // Special handling for boolean values
      if (stats.type === "boolean") {
        return [
          {
            range: "False",
            count: stats.distribution[0],
            fullRange: "False",
          },
          {
            range: "True",
            count: stats.distribution[1],
            fullRange: "True",
          },
        ];
      }

      // Special handling for single distinct value
      if (stats.min === stats.max) {
        return [
          {
            range: `${Math.round(stats.min)}`,
            count: stats.distribution[0],
            fullRange: `${Math.round(stats.min)}${getUnitLabel()}`,
          },
        ];
      }

      return stats.distribution.map((count, i) => ({
        range: `${Math.round(stats.min + i * stats.bucketSize)}`,
        count,
        fullRange: `${Math.round(
          stats.min + i * stats.bucketSize
        )} - ${Math.round(
          stats.min + (i + 1) * stats.bucketSize
        )}${getUnitLabel()}`,
      }));
    }, [stats]);

    return (
      <div className="space-y-1">
        <div
          className={`${
            isBold ? "font-bold" : ""
          } text-sm px-1 flex items-center`}
        >
          <div className="flex items-center">
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0"
              onClick={onSort}
            >
              {sortDirection === false && (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="text-primary"
                >
                  <path d="m3 16 4 4 4-4" />
                  <path d="M7 20V4" />
                  <path d="m21 8-4-4-4 4" />
                  <path d="M17 4v16" />
                </svg>
              )}
              {sortDirection === "asc" && (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="text-primary"
                >
                  <path d="m3 8 4-4 4 4" />
                  <path d="M7 4v16" />
                </svg>
              )}
              {sortDirection === "desc" && (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="text-primary"
                >
                  <path d="m3 16 4 4 4-4" />
                  <path d="M7 20V4" />
                </svg>
              )}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0"
              onClick={onExpand}
            >
              <Maximize2 className="h-3 w-3 text-primary" />
            </Button>
          </div>
          <span className="ml-2">{header}</span>
        </div>
        <div
          className={`${isBold ? "font-bold" : ""} space-y-2 ${
            filterValue ? "bg-primary/5 rounded-md" : ""
          }`}
        >
          <div className="flex items-center h-6">
            <div className="flex items-center w-6">
              <Search className="h-3 w-3 text-muted-foreground ml-1.5" />
            </div>
            <Input
              placeholder="Filter..."
              value={filterValue}
              onChange={(e) => onFilter(e.target.value)}
              className={`h-6 text-xs border-none shadow-none focus-visible:ring-0 ${
                filterValue ? "bg-primary/5" : ""
              }`}
            />
          </div>
        </div>
        {stats && (
          <div className="space-y-0.5">
            <div className="flex justify-between text-[10px] text-muted-foreground">
              {stats.isLowCardinality ? (
                <span className="w-full text-center">
                  {stats.distinctCount} distinct values
                </span>
              ) : (
                <>
                  <span>
                    {stats.min}
                    {stats.type === "array"
                      ? " items"
                      : stats.type === "string-words"
                      ? " words"
                      : stats.type === "string-chars"
                      ? " chars"
                      : ""}
                  </span>
                  <span>avg: {Math.round(stats.avg)}</span>
                  <span>
                    {stats.max}
                    {stats.type === "array"
                      ? " items"
                      : stats.type === "string-words"
                      ? " words"
                      : stats.type === "string-chars"
                      ? " chars"
                      : ""}
                  </span>
                </>
              )}
            </div>
            <div className="h-[60px] w-full">
              {stats.isLowCardinality ? (
                <CategoricalBarChart data={stats.sortedValueCounts} />
              ) : (
                <WordCountHistogram histogramData={histogramData} />
              )}
            </div>
          </div>
        )}
      </div>
    );
  }
);
ColumnHeader.displayName = "ColumnHeader";

const ColumnResizer = memo(({ header }: { header: Header<any, unknown> }) => {
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
});
ColumnResizer.displayName = "ColumnResizer";

interface ResizableRow<T extends DataType> extends Row<T> {
  getSize: () => number;
  setSize: (size: number) => void;
}

const RowResizer = memo(({ row }: { row: ResizableRow<any> }) => {
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
});
RowResizer.displayName = "RowResizer";

interface ResizableDataTableProps<T extends Record<string, unknown>> {
  data: T[];
  columns: ColumnType<T>[];
  boldedColumns?: string[];
  startingRowHeight?: number;
  currentOperation?: string;
}

interface MarkdownCellProps {
  content: string;
}

const MarkdownCell = memo(({ content }: MarkdownCellProps) => {
  return (
    <ReactMarkdown
      components={{
        h1: ({ children }) => (
          <div style={{ fontWeight: "bold", fontSize: "1.5rem" }}>
            {children}
          </div>
        ),
        h2: ({ children }) => (
          <div style={{ fontWeight: "bold", fontSize: "1.25rem" }}>
            {children}
          </div>
        ),
        h3: ({ children }) => (
          <div style={{ fontWeight: "bold", fontSize: "1.1rem" }}>
            {children}
          </div>
        ),
        h4: ({ children }) => (
          <div style={{ fontWeight: "bold" }}>{children}</div>
        ),
        ul: ({ children }) => (
          <ul
            style={{
              listStyleType: "•",
              paddingLeft: "1rem",
              margin: "0.25rem 0",
            }}
          >
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol
            style={{
              listStyleType: "decimal",
              paddingLeft: "1rem",
              margin: "0.25rem 0",
            }}
          >
            {children}
          </ol>
        ),
        li: ({ children }) => (
          <li style={{ marginBottom: "0.125rem" }}>{children}</li>
        ),
        code: ({
          className,
          children,
          inline,
          ...props
        }: {
          className?: string;
          children: React.ReactNode;
          inline?: boolean;
        }) => {
          const match = /language-(\w+)/.exec(className || "");
          return !inline && match ? (
            <pre className="bg-slate-100 p-2 rounded">
              <code className={className} {...props}>
                {children}
              </code>
            </pre>
          ) : (
            <code className="bg-slate-100 px-1 rounded" {...props}>
              {children}
            </code>
          );
        },
        pre: ({ children }) => (
          <pre className="bg-slate-100 p-2 rounded">{children}</pre>
        ),
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-slate-300 pl-4 my-2 italic">
            {children}
          </blockquote>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
});
MarkdownCell.displayName = "MarkdownCell";

interface ObservabilityIndicatorProps {
  row: Record<string, unknown>;
  currentOperation: string;
}

const ObservabilityIndicator = memo(
  ({ row, currentOperation }: ObservabilityIndicatorProps) => {
    // Only show observability data for the current operation
    const observabilityEntries = Object.entries(row).filter(
      ([key]) => key === `_observability_${currentOperation}`
    );

    if (observabilityEntries.length === 0) return null;

    return (
      <HoverCard>
        <HoverCardTrigger asChild>
          <div className="cursor-help">
            <Eye className="h-4 w-4 text-muted-foreground hover:text-primary" />
          </div>
        </HoverCardTrigger>
        <HoverCardContent
          className="w-[800px] max-h-[600px] overflow-auto"
          side="right"
          align="start"
        >
          <div className="space-y-4">
            <h3 className="text-lg font-semibold border-b pb-2">
              LLM Call(s) for {currentOperation}
            </h3>
            <div className="space-y-2">
              {observabilityEntries.map(([key, value]) => (
                <div key={key} className="flex flex-col gap-1">
                  <div className="text-sm text-muted-foreground">
                    {typeof value === "object"
                      ? JSON.stringify(value, null, 2)
                      : String(value)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </HoverCardContent>
      </HoverCard>
    );
  }
);
ObservabilityIndicator.displayName = "ObservabilityIndicator";

// Move the sortingFns definition outside of the table config
const createSortingFns = <T extends DataType>(
  data: T[],
  originalIndices: number[]
) => ({
  preserveIndex: (rowA: Row<T>, rowB: Row<T>) => {
    const a = rowA.original;
    const b = rowB.original;
    const aIndex = originalIndices[data.indexOf(a)];
    const bIndex = originalIndices[data.indexOf(b)];
    return aIndex - bIndex;
  },
});

export default function ResizableDataTable<T extends Record<string, unknown>>({
  data,
  columns,
  boldedColumns = [],
  startingRowHeight = 40,
  currentOperation,
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
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);

  const saveSettings = useCallback(() => {
    localStorage.setItem(
      TABLE_SETTINGS_KEY,
      JSON.stringify({ columnSizing, rowSizing })
    );
  }, [columnSizing, rowSizing]);

  useEffect(() => {
    // Initialize row heights when data changes
    const initialRowSizing = data.reduce((acc, _, index) => {
      acc[index] = startingRowHeight;
      return acc;
    }, {} as Record<string, number>);
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

  const [isResizing, setIsResizing] = useState(false);
  const debouncedSetIsResizing = useMemo(
    () => debounce((value: boolean) => setIsResizing(value), 150),
    []
  );

  const columnStats = useMemo(() => {
    const stats: Record<string, ColumnStats | null> = {};
    columns.forEach((column) => {
      const accessorKey = (column as { accessorKey?: string }).accessorKey;
      if (accessorKey) {
        stats[accessorKey] = calculateColumnStats(data, accessorKey);
      }
    });
    return stats;
  }, [data, columns]);

  const fuzzyFilter: FilterFn<T> = (row, columnId, value) => {
    const searchValue = value.toLowerCase();
    const cellValue = row.getValue(columnId);

    if (cellValue == null) return false;

    // Convert the cell value to a searchable string based on its type
    let searchableString = "";

    if (typeof cellValue === "object") {
      // Handle both arrays and objects by converting to JSON string
      searchableString = JSON.stringify(cellValue);
    } else {
      searchableString = String(cellValue);
    }

    return searchableString.toLowerCase().includes(searchValue);
  };

  // Add this state to store original row indices
  const [originalIndices] = useState(() => data.map((_, index) => index));

  // Create sorting functions
  const sortingFns = useMemo(
    () => createSortingFns(data, originalIndices),
    [data, originalIndices]
  );

  // Modify the table configuration
  const table = useReactTable({
    data,
    columns: sortedColumns
      .filter((col) => {
        const columnId = col.accessorKey || col.id;
        return !columnId?.startsWith("_observability_");
      })
      .map((col) => ({
        ...col,
        id: col.accessorKey,
        enableSorting: true,
        filterFn: fuzzyFilter,
        sortingFn: (rowA: Row<T>, rowB: Row<T>) => {
          const accessor = col.accessorKey;
          if (!accessor) return 0;

          const a = rowA.getValue(accessor);
          const b = rowB.getValue(accessor);

          // Handle null/undefined values
          if (a == null) return -1;
          if (b == null) return 1;

          // Sort based on type
          if (typeof a === "number" && typeof b === "number") {
            return a - b;
          }

          if (Array.isArray(a) && Array.isArray(b)) {
            return a.length - b.length;
          }

          // For strings, do alphabetical comparison
          return String(a).localeCompare(String(b));
        },
      })),
    columnResizeMode: "onChange" as ColumnResizeMode,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onColumnSizingChange: (newColumnSizing) => {
      setColumnSizing(newColumnSizing);
      setIsResizing(true);
      debouncedSetIsResizing(false);
      saveSettings();
    },
    onColumnVisibilityChange: setColumnVisibility,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    state: {
      columnSizing,
      columnVisibility,
      sorting,
      columnFilters,
    },
    enableSorting: true,
    enableColumnResizing: true,
    defaultColumn: {
      minSize: 30,
      size: 150,
    },
    initialState: {
      pagination: {
        pageSize: 5,
      },
    },
    filterFns: {
      fuzzy: fuzzyFilter,
    },
    sortingFns,
  });

  const resetColumnWidths = useCallback(() => {
    const initialSizing: ColumnSizingState = {};
    columns.forEach((column) => {
      if (column.initialWidth) {
        initialSizing[column.id as string] = column.initialWidth;
      } else {
        // Get all values for this column
        const values = data.map((row) => {
          const value = row[column.accessorKey as string];
          return value ? String(value) : "";
        });

        // Estimate width based on content (including header)
        const header = column.header as string;
        const maxContentLength = Math.max(
          header.length,
          ...values.map((v) => v.length)
        );

        // Estimate width: ~8px per character, with min 150px and max 400px
        const estimatedWidth = Math.min(
          Math.max(maxContentLength * 8, 150),
          400
        );

        initialSizing[column.id as string] = estimatedWidth;
      }
    });

    // Update the table's column sizing state
    table.setColumnSizing(initialSizing);

    // Update our local state and save settings
    setColumnSizing(initialSizing);
    saveSettings();
  }, [columns, data, saveSettings, table]);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [activeColumn, setActiveColumn] = useState<string | null>(null);
  const [currentValueIndex, setCurrentValueIndex] = useState(0);

  const handleColumnExpand = (columnId: string) => {
    setActiveColumn(columnId);
    setCurrentValueIndex(0);
    setDialogOpen(true);
  };

  return (
    <div className="w-full overflow-auto">
      <div className="mb-2 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center ml-2 h-7"
              >
                Show/Hide Columns
                <ChevronDown className="ml-1 h-3 w-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56">
              {table.getAllLeafColumns().map((column) => {
                return (
                  <DropdownMenuCheckboxItem
                    key={column.id}
                    checked={column.getIsVisible()}
                    onCheckedChange={(value) =>
                      column.toggleVisibility(!!value)
                    }
                  >
                    {column.id}
                  </DropdownMenuCheckboxItem>
                );
              })}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button
            variant="ghost"
            size="sm"
            className="h-7"
            onClick={resetColumnWidths}
          >
            Reset Widths
          </Button>
        </div>
        <div className="flex items-center space-x-2">
          {data.length > 0 && (
            <div className="flex items-center justify-end space-x-2 py-1 mr-2">
              <Button
                variant="outline"
                size="sm"
                className="h-7"
                onClick={() => table.previousPage()}
                disabled={!table.getCanPreviousPage()}
              >
                <ChevronLeft className="mr-1 h-3 w-3" /> Previous
              </Button>
              <span className="text-xs text-gray-600">
                Page {table.getState().pagination.pageIndex + 1} of{" "}
                {table.getPageCount()}
              </span>
              <Button
                variant="outline"
                size="sm"
                className="h-7"
                onClick={() => table.nextPage()}
                disabled={!table.getCanNextPage()}
              >
                Next <ChevronRight className="ml-1 h-3 w-3" />
              </Button>
            </div>
          )}
        </div>
      </div>
      <div style={{ width: "100%", overflow: "auto" }}>
        <Table
          style={{
            width: table.getTotalSize() + 100,
            minWidth: "100%",
          }}
        >
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                <TableHead style={{ width: "30px", minWidth: "30px" }}>
                  #
                </TableHead>
                {headerGroup.headers.map((header) => (
                  <TableHead
                    key={header.id}
                    style={{
                      width: header.getSize(),
                      position: "relative",
                      minWidth: `${header.column.columnDef.minSize}px`,
                    }}
                  >
                    {header.isPlaceholder ? null : (
                      <ColumnHeader
                        header={header.column.columnDef.header as string}
                        stats={
                          columnStats[
                            (
                              header.column.columnDef as {
                                accessorKey?: string;
                              }
                            ).accessorKey || ""
                          ]
                        }
                        isBold={boldedColumns.includes(
                          header.column.columnDef.header as string
                        )}
                        onFilter={(value) =>
                          header.column.setFilterValue(value)
                        }
                        filterValue={
                          (header.column.getFilterValue() as string) ?? ""
                        }
                        onSort={() => {
                          const currentSortDirection =
                            header.column.getIsSorted();
                          if (currentSortDirection === false) {
                            setSorting([{ id: header.column.id, desc: false }]);
                          } else if (currentSortDirection === "asc") {
                            setSorting([{ id: header.column.id, desc: true }]);
                          } else {
                            setSorting([]);
                          }
                        }}
                        sortDirection={header.column.getIsSorted()}
                        onExpand={() => handleColumnExpand(header.column.id)}
                      />
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
                      minWidth: "30px",
                      padding: "0.25rem",
                      textAlign: "center",
                    }}
                  >
                    <div className="flex flex-col items-center gap-1">
                      <span style={{ fontSize: "0.75rem", color: "#888" }}>
                        {row.index + 1}
                      </span>
                      <ObservabilityIndicator
                        row={row.original}
                        currentOperation={currentOperation}
                      />
                    </div>
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
                        {typeof cell.getValue() === "string" ? (
                          <SearchableCell
                            content={cell.getValue() as string}
                            isResizing={isResizing}
                          />
                        ) : typeof cell.getValue() === "object" ? (
                          <SearchableCell
                            content={JSON.stringify(cell.getValue(), null, 2)}
                            isResizing={isResizing}
                          >
                            {(searchTerm) =>
                              searchTerm ? null : (
                                <PrettyJSON data={cell.getValue()} />
                              )
                            }
                          </SearchableCell>
                        ) : (
                          flexRender(
                            cell.column.columnDef.cell,
                            cell.getContext()
                          )
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
                />
              </React.Fragment>
            ))}
          </TableBody>
        </Table>
      </div>

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

      {activeColumn && (
        <ColumnDialog
          isOpen={dialogOpen}
          onClose={() => {
            setDialogOpen(false);
            setActiveColumn(null);
          }}
          columnId={activeColumn}
          columnHeader={
            columns.find((col) => (col.accessorKey || col.id) === activeColumn)
              ?.header as string
          }
          data={data}
          currentIndex={currentValueIndex}
          onNavigate={(direction) => {
            if (direction === "next") {
              setCurrentValueIndex((prev) =>
                Math.min(prev + 1, data.length - 1)
              );
            } else {
              setCurrentValueIndex((prev) => Math.max(prev - 1, 0));
            }
          }}
          onJumpToRow={(index) => setCurrentValueIndex(index)}
          currentOperation={currentOperation}
          columnStats={columnStats[activeColumn]}
        />
      )}
    </div>
  );
}
