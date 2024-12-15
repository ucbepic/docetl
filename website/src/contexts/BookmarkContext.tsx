import React, {
  createContext,
  useContext,
  useState,
  ReactNode,
  useEffect,
} from "react";
import { Bookmark, BookmarkContextType, UserNote } from "@/app/types";
import { BOOKMARKS_STORAGE_KEY } from "@/app/localStorageKeys";

const BookmarkContext = createContext<BookmarkContextType | undefined>(
  undefined
);

export const useBookmarkContext = () => {
  const context = useContext(BookmarkContext);
  if (!context) {
    throw new Error(
      "useBookmarkContext must be used within a BookmarkProvider"
    );
  }
  return context;
};

export const BookmarkProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [bookmarks, setBookmarks] = useState<Bookmark[]>(() => {
    if (typeof window !== "undefined") {
      const storedBookmarks = localStorage.getItem(BOOKMARKS_STORAGE_KEY);
      return storedBookmarks ? JSON.parse(storedBookmarks) : [];
    }
    return [];
  });

  useEffect(() => {
    localStorage.setItem(BOOKMARKS_STORAGE_KEY, JSON.stringify(bookmarks));
  }, [bookmarks]);

  const addBookmark = (color: string, notes: UserNote[]) => {
    const newBookmark: Bookmark = {
      id: Date.now().toString(),
      color,
      notes,
    };
    setBookmarks((prevBookmarks) => [...prevBookmarks, newBookmark]);
  };

  const removeBookmark = (id: string) => {
    setBookmarks((prevBookmarks) =>
      prevBookmarks.filter((bookmark) => bookmark.id !== id)
    );
  };

  const getNotesForRowAndColumn = (
    rowIndex: number,
    columnId: string
  ): UserNote[] => {
    return bookmarks.flatMap((bookmark) =>
      bookmark.notes.filter(
        (note) =>
          note.metadata?.rowIndex === rowIndex &&
          note.metadata?.columnId === columnId
      )
    );
  };

  return (
    <BookmarkContext.Provider
      value={{
        bookmarks,
        addBookmark,
        removeBookmark,
        getNotesForRowAndColumn,
      }}
    >
      {children}
    </BookmarkContext.Provider>
  );
};
