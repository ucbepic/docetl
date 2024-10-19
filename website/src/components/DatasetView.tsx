import { File } from '@/app/types';
import React, { useRef, useMemo, useState, useCallback, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useInfiniteQuery } from '@tanstack/react-query';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { ChevronUp, ChevronDown, Search } from 'lucide-react';
import BookmarkableText from '@/components/BookmarkableText';

const CHUNK_SIZE = 100000; // This should match the CHUNK_SIZE in the API

interface FileChunk {
  content: string;
  totalSize: number;
  page: number;
  hasMore: boolean;
}

interface Match {
  lineIndex: number;
  startIndex: number;
  endIndex: number;
}

const DatasetView: React.FC<{ file: File | null }> = ({ file }) => {
  const parentRef = useRef<HTMLDivElement>(null);
  const [keys, setKeys] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);
  const [matches, setMatches] = useState<Match[]>([]);

  const fetchFileContent = async ({ pageParam = 0 }): Promise<FileChunk> => {
    if (!file?.path) throw new Error('No file selected');
    const response = await fetch(`/api/readFilePage?path=${encodeURIComponent(file.path)}&page=${pageParam}`);
    if (!response.ok) throw new Error('Failed to fetch file content');
    return response.json();
  };

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetching,
    isError,
    error
  } = useInfiniteQuery<FileChunk, Error>({
    queryKey: ['fileContent', file?.path],
    queryFn: fetchFileContent,
    getNextPageParam: (lastPage) => lastPage.hasMore ? lastPage.page + 1 : undefined,
    enabled: !!file?.path,
  });

  const lines = useMemo(() => {
    const fullContent = data?.pages.map(page => page.content).join('') ?? '';
    return fullContent.split('\n');
  }, [data]);

  const totalSize = data?.pages[0]?.totalSize ?? 0;

  const rowVirtualizer = useVirtualizer({
    count: lines.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 30,
    overscan: 5,
    measureElement: useCallback((element) => {
      return element?.getBoundingClientRect().height || 30;
    }, []),
  });

  // Extract keys from the first valid JSON object in the data
  useMemo(() => {
    let jsonString = '';
    let braceCount = 0;
    let inObject = false;

    for (const line of lines) {
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '{') {
          if (!inObject) inObject = true;
          braceCount++;
        } else if (char === '}') {
          braceCount--;
        }

        if (inObject) {
          jsonString += char;
        }

        if (inObject && braceCount === 0) {
          try {
            const parsedObject = JSON.parse(jsonString);
            setKeys(Object.keys(parsedObject));
            return; // Exit after finding the first valid object
          } catch (error) {
            console.error('Error parsing JSON:', error);
          }
          // Reset for next attempt
          jsonString = '';
          inObject = false;
        }
      }
    }
  }, [lines]);

  // Load more data when approaching the end of the list
  useEffect(() => {
    const lastItem = rowVirtualizer.getVirtualItems().at(-1);
    if (lastItem && lastItem.index >= lines.length - 1 && hasNextPage && !isFetching) {
      fetchNextPage();
    }
  }, [rowVirtualizer.getVirtualItems(), hasNextPage, isFetching, fetchNextPage, lines.length]);

  // Perform search and update matches
  useEffect(() => {
    if (searchTerm) {
      const newMatches: Match[] = [];
      const regex = new RegExp(searchTerm, 'gi');
      lines.forEach((line, lineIndex) => {
        let match;
        while ((match = regex.exec(line)) !== null) {
          newMatches.push({
            lineIndex,
            startIndex: match.index,
            endIndex: match.index + match[0].length
          });
        }
      });
      setMatches(newMatches);
      setCurrentMatchIndex(0);
      if (newMatches.length > 0) {
        rowVirtualizer.scrollToIndex(newMatches[0].lineIndex);
      }
    } else {
      setMatches([]);
      setCurrentMatchIndex(0);
    }
  }, [searchTerm, lines, rowVirtualizer]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    // The search is already performed in the useEffect above
  };

  const navigateMatch = (direction: 'next' | 'prev') => {
    if (matches.length === 0) return;
    let newIndex = direction === 'next' ? currentMatchIndex + 1 : currentMatchIndex - 1;
    if (newIndex < 0) newIndex = matches.length - 1;
    if (newIndex >= matches.length) newIndex = 0;
    setCurrentMatchIndex(newIndex);
    rowVirtualizer.scrollToIndex(matches[newIndex].lineIndex);
  };

  const highlightMatches = (text: string, lineIndex: number) => {
    if (!searchTerm) return text;
    const parts = [];
    let lastIndex = 0;
    matches.filter(match => match.lineIndex === lineIndex).forEach(match => {
      if (lastIndex < match.startIndex) {
        parts.push(text.slice(lastIndex, match.startIndex));
      }
      parts.push(
        <mark key={match.startIndex} className="bg-yellow-200">
          {text.slice(match.startIndex, match.endIndex)}
        </mark>
      );
      lastIndex = match.endIndex;
    });
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }
    return parts;
  };

  if (isError) return <div>Error: {error.message}</div>;

  return (
    <BookmarkableText source="dataset">
    <div className="h-full p-4 bg-white flex flex-col">
      <h2 className="text-lg font-bold mb-2">{file?.name}</h2>
      <div className="mb-4">
        <p className="mb-2 font-semibold">Available keys:</p>
        {keys.map((key) => (
          <Badge key={key} className="mr-2 mb-2">{key}</Badge>
        ))}
      </div>
      <form onSubmit={handleSearch} className="flex items-center mb-4">
        <Input
          type="text"
          placeholder="Search..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="mr-1"
        />
        <Button type="submit" variant="outline" size="icon"><Search className="h-4 w-4" /></Button>
        <Button type="button" size="icon" variant="outline" onClick={() => navigateMatch('prev')} disabled={matches.length === 0} className="ml-1">
          <ChevronUp className="h-4 w-4" />
        </Button>
        <Button type="button" variant="outline" size="icon" onClick={() => navigateMatch('next')} disabled={matches.length === 0} className="ml-1">
          <ChevronDown className="h-4 w-4" />
        </Button>
        <span className="ml-2">
          {matches.length > 0 ? `${currentMatchIndex + 1} of ${matches.length} matches` : 'No matches'}
        </span>
      </form>
      <div ref={parentRef} className="flex-grow overflow-auto">
        <div
          style={{
            height: `${rowVirtualizer.getTotalSize()}px`,
            width: '100%',
            position: 'relative',
          }}
        >
          {rowVirtualizer.getVirtualItems().map((virtualRow) => {
            const lineContent = lines[virtualRow.index];

            return (
              <div
                key={virtualRow.index}
                data-index={virtualRow.index}
                ref={rowVirtualizer.measureElement}
                className="absolute top-0 left-0 w-full"
                style={{
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                <div className="flex">
                  <span className="inline-block w-12 flex-shrink-0 text-gray-500 select-none text-right pr-2 py-1">
                    {virtualRow.index + 1}
                  </span>
                  <div className="flex-grow overflow-hidden py-1">
                    <pre className="whitespace-pre-wrap break-words font-mono text-sm">
                      {highlightMatches(lineContent, virtualRow.index)}
                    </pre>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
    </BookmarkableText>
  );
};

export default DatasetView;