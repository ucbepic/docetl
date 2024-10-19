import React from 'react';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { Command, CommandInput, CommandList, CommandEmpty, CommandGroup, CommandItem } from '@/components/ui/command';
import { Bookmark, Search } from 'lucide-react';
import { useBookmarkContext } from '@/contexts/BookmarkContext';

interface CommandMenuProps {
  isOpen: boolean;
  onClose: () => void;
  onBookmarkCurrent: () => void;
}

const CommandMenu: React.FC<CommandMenuProps> = ({ isOpen, onClose, onBookmarkCurrent }) => {
  const { bookmarks } = useBookmarkContext();

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="p-0">
        <Command className="rounded-lg border shadow-md">
          <CommandInput placeholder="Type a command or search..." />
          <CommandList>
            <CommandEmpty>No results found.</CommandEmpty>
            <CommandGroup heading="Actions">
              <CommandItem onSelect={onBookmarkCurrent}>
                <Bookmark className="mr-2 h-4 w-4" />
                <span>Bookmark current selection</span>
              </CommandItem>
              {/* Add more actions here */}
            </CommandGroup>
            <CommandGroup heading="Bookmarks">
              {bookmarks.map((bookmark) => (
                <CommandItem key={bookmark.id} onSelect={() => console.log('Navigate to bookmark', bookmark.id)}>
                  <Search className="mr-2 h-4 w-4" />
                  <span>{bookmark.text.substring(0, 30)}...</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </DialogContent>
    </Dialog>
  );
};

export default CommandMenu;