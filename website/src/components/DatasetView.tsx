import { File } from '@/app/types';
import React, { useRef, useMemo, useState, useCallback, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { useInfiniteQuery } from '@tanstack/react-query';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { ChevronUp, ChevronDown, Search } from 'lucide-react';
import BookmarkableText from '@/components/BookmarkableText';

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
    // @ts-ignore
    queryFn: fetchFileContent,
    getNextPageParam: (lastPage) => lastPage.hasMore ? lastPage.page + 1 : undefined,
    enabled: !!file?.path,
  });

  const lines = useMemo(() => {
    // @ts-ignore
    return data?.pages.flatMap(page => page.content.split('\n')) ?? [];
  }, [data]);

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

  // Perform search and update matches
  useEffect(() => {
    if (searchTerm.length >= 5) {
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
    } else {
      setMatches([]);
      setCurrentMatchIndex(0);
    }
  }, [searchTerm, lines]);

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

    // Scroll to the new match
    const matchElement = document.getElementById(`match-${newIndex}`);
    if (matchElement) {
      matchElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const highlightMatches = (text: string, lineIndex: number) => {
    if (!searchTerm || searchTerm.length < 5) return text;
    const parts = [];
    let lastIndex = 0;
    matches.filter(match => match.lineIndex === lineIndex).forEach((match, index) => {
      if (lastIndex < match.startIndex) {
        parts.push(text.slice(lastIndex, match.startIndex));
      }
      parts.push(
        <mark
          key={match.startIndex}
          id={`match-${matches.findIndex(m => m.lineIndex === lineIndex && m.startIndex === match.startIndex)}`}
          className={`bg-yellow-200 ${currentMatchIndex === matches.findIndex(m => m.lineIndex === lineIndex && m.startIndex === match.startIndex) ? 'ring-2 ring-blue-500' : ''}`}
        >
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

  // Keep fetching in the background
  useEffect(() => {
    const fetchNextPageIfNeeded = () => {
      if (hasNextPage && !isFetching) {
        fetchNextPage();
      }
    };

    const intervalId = setInterval(fetchNextPageIfNeeded, 1000); // Check every second

    return () => clearInterval(intervalId);
  }, [fetchNextPage, hasNextPage, isFetching]);

  useEffect(() => {
    // Scroll to the current match when it changes
    const currentMatchElement = document.getElementById(`match-${currentMatchIndex}`);
    if (currentMatchElement) {
      currentMatchElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [currentMatchIndex]);

  if (isError) return <div>Error: {error.message}</div>;

  return (
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
          placeholder="Search (min 5 characters)..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="mr-1"
        />
        <Button type="submit" variant="outline" size="icon" disabled={searchTerm.length < 5}><Search className="h-4 w-4" /></Button>
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
      <BookmarkableText source="dataset">
      <div ref={parentRef} className="flex-grow overflow-y-auto">
        {lines.map((lineContent, index) => (
          <div key={index} className="flex">
            <span className="inline-block w-12 flex-shrink-0 text-gray-500 select-none text-right pr-2">
              {index + 1}
            </span>
            <div className="flex-grow">
              <pre className="whitespace-pre-wrap break-words font-mono text-sm">
                {highlightMatches(lineContent, index)}
              </pre>
            </div>
          </div>
        ))}
        {isFetching && <div className="text-center py-4">Loading more...</div>}
      </div>
      </BookmarkableText>
    </div>
  );
};

export default DatasetView;