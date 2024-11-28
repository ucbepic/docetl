import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import { useBookmarkContext } from "@/contexts/BookmarkContext";
import {
  Bookmark,
  ChevronDown,
  ChevronUp,
  Filter,
  Trash2,
  X,
  MessageSquare,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { UserNote } from "@/app/types";

const BookmarksPanel: React.FC = () => {
  const { bookmarks, removeBookmark } = useBookmarkContext();
  const [searchTerm, setSearchTerm] = useState("");
  const [expandedBookmarkId, setExpandedBookmarkId] = useState<string | null>(
    null
  );
  const [selectedColor, setSelectedColor] = useState<string | "all">("all");

  const uniqueColors = Array.from(
    new Set(bookmarks.map((bookmark) => bookmark.color))
  );

  const filteredBookmarks = bookmarks.filter(
    (bookmark) =>
      bookmark.notes.some((note) =>
        note.note.toLowerCase().includes(searchTerm.toLowerCase())
      ) &&
      (selectedColor === "all" || bookmark.color === selectedColor)
  );

  const toggleBookmarkExpansion = (id: string) => {
    setExpandedBookmarkId(expandedBookmarkId === id ? null : id);
  };

  const handleDeleteBookmark = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    removeBookmark(id);
    if (expandedBookmarkId === id) {
      setExpandedBookmarkId(null);
    }
  };

  const handleClearAll = () => {
    bookmarks.forEach((bookmark) => removeBookmark(bookmark.id));
    setSearchTerm("");
    setSelectedColor("all");
  };

  const renderNoteContent = (note: UserNote) => {
    return (
      <div className="mt-2 space-y-1">
        {note.metadata?.columnId && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <MessageSquare size={12} />
            <span>Column: {note.metadata.columnId}</span>
            {note.metadata.rowIndex !== undefined && (
              <span>• Row: {note.metadata.rowIndex}</span>
            )}
          </div>
        )}
        {note.metadata?.rowContent && (
          <div className="text-xs bg-muted/50 p-2 rounded-md mt-1">
            <div className="font-semibold mb-1">Row Context:</div>
            <pre className="overflow-x-auto">
              {JSON.stringify(note.metadata.rowContent, null, 2)}
            </pre>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="h-full p-4 flex flex-col">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-sm font-bold flex items-center uppercase">
          <Bookmark className="mr-2" size={16} />
          Notes & Feedback
        </h2>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleClearAll}
          className="text-gray-500 hover:text-gray-700"
        >
          <X size={16} className="mr-1" />
          Clear All
        </Button>
      </div>
      <div className="flex mb-2">
        <Input
          type="text"
          placeholder="Search..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="flex-grow mr-2"
        />
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon" className="relative">
              <Filter className="h-4 w-4" />
              {selectedColor !== "all" && (
                <span className="absolute -top-2 -right-2 bg-primary text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                  {1}
                </span>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-48">
            <div className="space-y-2">
              <Select
                value={selectedColor}
                onValueChange={(value) => setSelectedColor(value as string)}
              >
                <SelectTrigger className="border-none shadow-none">
                  <SelectValue placeholder="Filter by color" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All colors</SelectItem>
                  {uniqueColors.map((color) => (
                    <SelectItem key={color} value={color}>
                      <div className="flex items-center">
                        <div
                          className="w-3 h-3 rounded-full mr-2"
                          style={{ backgroundColor: color }}
                        />
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </PopoverContent>
        </Popover>
      </div>
      <div className="overflow-y-auto flex-grow">
        {filteredBookmarks.map((bookmark) => (
          <div
            key={bookmark.id}
            className="mb-1 rounded hover:bg-muted/50 transition-colors"
          >
            <div
              className="flex items-start cursor-pointer px-1.5 py-1"
              onClick={() => toggleBookmarkExpansion(bookmark.id)}
            >
              <div
                className="w-3 h-3 rounded-full mr-2 mt-1 flex-shrink-0"
                style={{
                  backgroundColor: bookmark.color,
                  minWidth: "0.75rem",
                  minHeight: "0.75rem",
                }}
              />
              <div className="flex-grow min-w-0 mr-1">
                <div
                  className={`text-sm italic ${
                    expandedBookmarkId === bookmark.id
                      ? "whitespace-normal"
                      : "whitespace-nowrap overflow-hidden text-ellipsis"
                  }`}
                >
                  &quot;{bookmark.notes[0]?.note || "No notes"}&quot;
                </div>
              </div>
              <div className="flex-shrink-0">
                {expandedBookmarkId === bookmark.id ? (
                  <ChevronUp className="text-muted-foreground" size={16} />
                ) : (
                  <ChevronDown className="text-muted-foreground" size={16} />
                )}
              </div>
            </div>
            {expandedBookmarkId === bookmark.id && (
              <div className="mt-2 ml-5">
                {bookmark.notes.map((note, index) => (
                  <div key={note.id || index} className="border-l-2 pl-2 mb-2">
                    {renderNoteContent(note)}
                  </div>
                ))}
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={(e) => handleDeleteBookmark(e, bookmark.id)}
                  className="mt-2"
                >
                  <Trash2 size={16} className="mr-2" />
                  Delete
                </Button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default BookmarksPanel;
