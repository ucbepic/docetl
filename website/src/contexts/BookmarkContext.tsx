import React, { createContext, useContext, useState, ReactNode } from 'react';
import { Bookmark, BookmarkContextType, UserNote } from '@/app/types';
const BookmarkContext = createContext<BookmarkContextType | undefined>(undefined);

export const useBookmarkContext = () => {
  const context = useContext(BookmarkContext);
  if (!context) {
    throw new Error('useBookmarkContext must be used within a BookmarkProvider');
  }
  return context;
};

export const BookmarkProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);

  const addBookmark = (text: string, source: string, color: string, notes: UserNote[]) => {
    const newBookmark: Bookmark = {
      id: Date.now().toString(),
      text,
      source,
      color,
      notes: [],
    };
    setBookmarks([...bookmarks, newBookmark]);
  };

  const removeBookmark = (id: string) => {
    setBookmarks(bookmarks.filter(bookmark => bookmark.id !== id));
  };

  return (
    <BookmarkContext.Provider value={{ bookmarks, addBookmark, removeBookmark }}>
      {children}
    </BookmarkContext.Provider>
  );
};