import React, { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Bookmark, BookmarkPlus, X } from "lucide-react";
import { useBookmarkContext } from "@/contexts/BookmarkContext";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { UserNote } from "@/app/types";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
} from "@/components/ui/form";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { useToast } from "@/hooks/use-toast";

interface BookmarkableTextProps {
  children: React.ReactNode;
  source: string;
}

const formSchema = z.object({
  editedText: z.string().min(1, "Edited text is required"),
  color: z.string(),
  note: z.string(),
});

const BookmarkableText: React.FC<BookmarkableTextProps> = ({
  children,
  source,
}) => {
  const [buttonPosition, setButtonPosition] = useState({ x: 0, y: 0 });
  const [showButton, setShowButton] = useState(false);
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);
  const textRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const { addBookmark } = useBookmarkContext();
  const { toast } = useToast();

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      editedText: "",
      color: "#FF0000",
      note: "",
    },
  });

  const handleBookmark = (values: z.infer<typeof formSchema>) => {
    const userNotes: UserNote[] = [{ id: "default", note: values.note }];
    addBookmark(values.editedText, source, values.color, userNotes);
    setShowButton(false);
    setIsPopoverOpen(false);
    toast({
      title: "Bookmark Added",
      description: "Your bookmark has been successfully added.",
    });
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      //   if (
      //     isPopoverOpen &&
      //     popoverRef.current &&
      //     !popoverRef.current.contains(event.target as Node) &&
      //     buttonRef.current &&
      //     !buttonRef.current.contains(event.target as Node)
      //   ) {
      //     setIsPopoverOpen(false);
      //   }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isPopoverOpen]);

  const handleMultiElementSelection = (
    event: React.MouseEvent | React.TouchEvent,
  ) => {
    event.stopPropagation();
    const selection = window.getSelection();

    if (selection && !selection.isCollapsed) {
      const range = selection.getRangeAt(0);
      const fragment = range.cloneContents();
      const tempDiv = document.createElement("div");
      tempDiv.appendChild(fragment);
      const text = tempDiv.innerText.trim();
      if (text) {
        form.setValue("editedText", text);
        const rect = range.getBoundingClientRect();
        setButtonPosition({
          x: rect.left + rect.width / 2,
          y: rect.top,
        });
        setShowButton(true);
      } else {
        // setShowButton(false);
      }
    } else {
      // if (!isPopoverOpen) {
      //     setShowButton(false);
      // } else {
      //     setShowButton(true);
      // }
    }
  };

  const handlePopoverOpenChange = (open: boolean) => {
    setIsPopoverOpen(open);
  };

  const handleClosePopover = () => {
    setShowButton(false);
    setIsPopoverOpen(false);
  };

  return (
    <div
      ref={textRef}
      onMouseUp={handleMultiElementSelection}
      onTouchEnd={handleMultiElementSelection}
      className="overflow-y-auto"
    >
      {children}
      {showButton && (
        <Popover
          open={isPopoverOpen}
          onOpenChange={handlePopoverOpenChange}
          modal={true}
        >
          <PopoverTrigger asChild>
            <Button
              size="icon"
              ref={buttonRef}
              variant="default"
              aria-label="Bookmark"
              className="shadow-md"
              style={{
                position: "absolute",
                left: `${buttonPosition.x}px`,
                top: `${buttonPosition.y}px`,
                transform: "translate(-50%, -100%)",
              }}
              onClick={(e) => {
                e.stopPropagation();
                setIsPopoverOpen(true);
              }}
            >
              <BookmarkPlus />
            </Button>
          </PopoverTrigger>
          <PopoverContent
            className="w-[300px]"
            ref={popoverRef}
            onInteractOutside={(e) => {
              if (!buttonRef.current?.contains(e.target as Node)) {
                e.preventDefault();
              }
            }}
            side="top"
            align="start"
            sideOffset={5}
          >
            <div className="flex justify-between items-center mb-2">
              <div className="flex items-center">
                <Bookmark className="h-5 w-5 text-primary mr-2" />
                <h4 className="font-medium">Edit Bookmark</h4>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleClosePopover}
                className="h-6 w-6"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(handleBookmark)}>
                <FormField
                  control={form.control}
                  name="editedText"
                  render={({ field }) => (
                    <FormItem>
                      <FormControl>
                        <Textarea {...field} className="min-h-[100px] mb-2" />
                      </FormControl>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="color"
                  render={({ field }) => (
                    <FormItem>
                      <Select
                        onValueChange={field.onChange}
                        defaultValue={field.value}
                      >
                        <FormControl>
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select a color" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="#FF0000">
                            <div className="flex items-center">
                              <div className="w-4 h-4 rounded-full bg-[#FF0000] mr-2"></div>
                              Red
                            </div>
                          </SelectItem>
                          <SelectItem value="#00FF00">
                            <div className="flex items-center">
                              <div className="w-4 h-4 rounded-full bg-[#00FF00] mr-2"></div>
                              Green
                            </div>
                          </SelectItem>
                          <SelectItem value="#0000FF">
                            <div className="flex items-center">
                              <div className="w-4 h-4 rounded-full bg-[#0000FF] mr-2"></div>
                              Blue
                            </div>
                          </SelectItem>
                          <SelectItem value="#FFFF00">
                            <div className="flex items-center">
                              <div className="w-4 h-4 rounded-full bg-[#FFFF00] mr-2"></div>
                              Yellow
                            </div>
                          </SelectItem>
                          <SelectItem value="#FF00FF">
                            <div className="flex items-center">
                              <div className="w-4 h-4 rounded-full bg-[#FF00FF] mr-2"></div>
                              Magenta
                            </div>
                          </SelectItem>
                          <SelectItem value="#00FFFF">
                            <div className="flex items-center">
                              <div className="w-4 h-4 rounded-full bg-[#00FFFF] mr-2"></div>
                              Cyan
                            </div>
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="note"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Notes:</FormLabel>
                      <FormControl>
                        <Textarea {...field} className="min-h-[50px] mb-1" />
                      </FormControl>
                    </FormItem>
                  )}
                />
                <Button type="submit" className="w-full mt-2">
                  Add Bookmark
                </Button>
              </form>
            </Form>
          </PopoverContent>
        </Popover>
      )}
    </div>
  );
};

export default BookmarkableText;
