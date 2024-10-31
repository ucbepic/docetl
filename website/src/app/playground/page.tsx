"use client";

import dynamic from "next/dynamic";
import React, { useEffect, useState, useRef } from "react";
import { Scroll, Info, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
const Output = dynamic(
  () => import("../../components/Output").then((mod) => mod.Output),
  {
    ssr: false,
  }
);
import { File } from "@/app/types";
import {
  PipelineProvider,
  usePipelineContext,
} from "@/contexts/PipelineContext";
const FileExplorer = dynamic(
  () => import("@/components/FileExplorer").then((mod) => mod.FileExplorer),
  {
    ssr: false,
  }
);
const DatasetView = dynamic(
  () => import("@/components/DatasetView").then((mod) => mod.default),
  {
    ssr: false,
  }
);
const PipelineGUI = dynamic(
  () => import("@/components/PipelineGui").then((mod) => mod.default),
  {
    ssr: false,
  }
);
const BookmarksPanel = dynamic(
  () => import("@/components/BookmarksPanel").then((mod) => mod.default),
  {
    ssr: false,
  }
);
import { BookmarkProvider } from "@/contexts/BookmarkContext";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { WebSocketProvider } from "@/contexts/WebSocketContext";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Menubar,
  MenubarContent,
  MenubarItem,
  MenubarMenu,
  MenubarTrigger,
} from "@/components/ui/menubar";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  saveToFile,
  loadFromFile,
  saveToFileClassic,
  loadFromFileClassic,
} from "@/utils/fileOperations";
import * as localStorageKeys from "@/app/localStorageKeys";
import { toast } from "@/hooks/use-toast";
import AIChatPanel from "@/components/AIChatPanel";

const LeftPanelIcon: React.FC<{ isActive: boolean }> = ({ isActive }) => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 16 16"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      fill={isActive ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="1.2"
      d="M2 2.5h3.5v11H2z"
    />
    <path
      fill="none"
      stroke="currentColor"
      strokeWidth="1.2"
      d="M5.5 2.5h8.5v11H5.5z"
    />
  </svg>
);

const BottomPanelIcon: React.FC<{ isActive: boolean }> = ({ isActive }) => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 16 16"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      fill="none"
      stroke="currentColor"
      strokeWidth="1.2"
      d="M2 2.5h12v7.5H2z"
    />
    <path
      fill={isActive ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="1.2"
      d="M2 10h12v3.5H2z"
    />
  </svg>
);

const RightPanelIcon: React.FC<{ isActive: boolean }> = ({ isActive }) => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 16 16"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      fill="none"
      stroke="currentColor"
      strokeWidth="1.2"
      d="M2 2.5h8.5v11H2z"
    />
    <path
      fill={isActive ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="1.2"
      d="M10.5 2.5h3.5v11h-3.5z"
    />
  </svg>
);

const CodeEditorPipelineApp: React.FC = () => {
  const [isLocalhost, setIsLocalhost] = useState(true);
  // Add client-side only rendering for the cost display
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsLocalhost(
      window.location.hostname === "localhost" ||
        window.location.hostname === "127.0.0.1"
    );
    setIsMounted(true);
  }, []);

  if (!isLocalhost) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="max-w-2xl p-6 bg-white rounded-lg shadow-md">
          <h1 className="text-2xl font-bold text-primary mb-4">
            DocETL Playground
          </h1>
          <p className="mb-4">
            The DocETL playground is designed to run locally. To use it, please
            follow these steps:
          </p>
          <ol className="list-decimal list-inside mb-4">
            <li>
              Clone the GitHub repo:{" "}
              <a
                href="https://github.com/ucbepic/docetl"
                className="text-blue-500 hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                https://github.com/ucbepic/docetl
              </a>
            </li>
            <li>
              Set up the project by running:
              <pre className="bg-gray-100 p-2 rounded mt-2 mb-2">
                make install make install-ui
              </pre>
            </li>
            <li>
              Start the application:
              <pre className="bg-gray-100 p-2 rounded mt-2 mb-2">
                make run-ui-prod
              </pre>
            </li>
            <li>
              Navigate to{" "}
              <a
                href="http://localhost:3000/playground"
                className="text-blue-500 hover:underline"
              >
                http://localhost:3000/playground
              </a>
            </li>
          </ol>
          <p>
            Once you've completed these steps, you'll be able to use the DocETL
            playground locally.
          </p>
        </div>
      </div>
    );
  }

  const [showFileExplorer, setShowFileExplorer] = useState(true);
  const [showOutput, setShowOutput] = useState(true);
  const [showDatasetView, setShowDatasetView] = useState(false);
  const [showChat, setShowChat] = useState(false);

  const {
    operations,
    currentFile,
    setOperations,
    setCurrentFile,
    cost,
    files,
    setFiles,
    clearPipelineState,
    saveProgress,
    unsavedChanges,
  } = usePipelineContext();

  const handleSaveAs = async () => {
    try {
      // Collect all localStorage data
      const data: Record<string, any> = {};
      Object.values(localStorageKeys).forEach((key) => {
        const value = localStorage.getItem(key);
        if (value) {
          data[key] = JSON.parse(value);
        }
      });

      // Try modern API first, fall back to classic if not supported
      try {
        await saveToFile(data, "pipeline.dtl");
      } catch (err) {
        if (
          err instanceof TypeError &&
          err.message.includes("showSaveFilePicker")
        ) {
          // Fall back to classic method if File System Access API is not supported
          await saveToFileClassic(data, "pipeline.dtl");
        } else {
          throw err;
        }
      }
    } catch (error) {
      console.error("Error saving pipeline session:", error);
      toast({
        title: "Error Saving Pipeline Session",
        description:
          "There was an error saving your pipeline session. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleOpen = async () => {
    try {
      let data;
      try {
        data = await loadFromFile();
      } catch (err) {
        if (
          err instanceof TypeError &&
          err.message.includes("showOpenFilePicker")
        ) {
          // Fall back to classic method if File System Access API is not supported
          data = await loadFromFileClassic();
        } else {
          throw err;
        }
      }

      if (data) {
        // Clear current state
        clearPipelineState();

        // Restore all data to localStorage
        Object.entries(data).forEach(([key, value]) => {
          localStorage.setItem(key, JSON.stringify(value));
        });

        // Reload the page to apply changes
        window.location.reload();
      }
    } catch (error) {
      console.error("Error loading pipeline:", error);
      toast({
        title: "Error Loading Pipeline",
        description:
          "There was an error loading your pipeline. Please check the file and try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <BookmarkProvider>
      <div className="h-screen flex flex-col bg-gray-50">
        <div className="p-1 flex justify-between items-center border-b">
          <div className="flex-1 flex">
            <Menubar className="border-none bg-transparent shadow-none">
              <MenubarMenu>
                <MenubarTrigger>File</MenubarTrigger>
                <MenubarContent>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <MenubarItem onSelect={(e) => e.preventDefault()}>
                        New
                      </MenubarItem>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>
                          Clear Pipeline State
                        </AlertDialogTitle>
                        <AlertDialogDescription>
                          Are you sure you want to clear the pipeline state?
                          This will take you to a default pipeline and clear all
                          notes and outputs. This action cannot be undone.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={() => {
                            clearPipelineState();
                            window.location.reload();
                          }}
                        >
                          Clear
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                  <MenubarItem onSelect={handleOpen}>Open</MenubarItem>
                  <MenubarItem onSelect={handleSaveAs}>Save As</MenubarItem>
                </MenubarContent>
              </MenubarMenu>
              <MenubarMenu>
                <MenubarTrigger>Assistant</MenubarTrigger>
                <MenubarContent>
                  <MenubarItem onSelect={() => setShowChat(!showChat)}>
                    Toggle Chat
                  </MenubarItem>
                </MenubarContent>
              </MenubarMenu>
            </Menubar>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      saveProgress();
                      toast({
                        title: "Progress Saved",
                        description: "Your pipeline progress has been saved.",
                        duration: 3000,
                      });
                    }}
                    className={`relative ${
                      unsavedChanges ? "border-orange-500" : ""
                    }`}
                  >
                    <Save
                      size={16}
                      className={unsavedChanges ? "text-orange-500" : ""}
                    />
                    {unsavedChanges && (
                      <span className="absolute top-0 right-0 w-2 h-2 bg-orange-500 rounded-full" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {unsavedChanges
                    ? "Save changes to avoid losing progress!"
                    : "No unsaved changes"}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <div className="flex items-center">
            <Scroll className="mr-2 text-primary" size={20} />
            <h1 className="text-lg font-bold text-primary">DocETL</h1>
          </div>
          <div className="flex-1 flex justify-end items-center space-x-1">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button variant="ghost" size="icon">
                        <Info size={20} />
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-80">
                      <h3 className="font-semibold mb-2">About DocETL</h3>
                      <p className="text-sm text-gray-600">
                        This is a research project from the EPIC Data Lab at the
                        University of California, Berkeley. To learn more, visit{" "}
                        <a
                          href="https://docetl.org"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-500 hover:underline"
                        >
                          docetl.org
                        </a>
                        .
                      </p>
                    </PopoverContent>
                  </Popover>
                </TooltipTrigger>
                <TooltipContent>
                  <p>About DocETL</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            {/* Only render the cost when client-side */}
            {isMounted && (
              <span className="text-sm font-medium text-gray-600">
                Cost: ${cost.toFixed(2)}
              </span>
            )}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowFileExplorer(!showFileExplorer)}
                    className="w-10 h-10"
                  >
                    <LeftPanelIcon isActive={showFileExplorer} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Toggle File Explorer</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowOutput(!showOutput)}
                    className="w-10 h-10"
                  >
                    <BottomPanelIcon isActive={showOutput} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Toggle Output Panel</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowDatasetView(!showDatasetView)}
                    className="w-10 h-10"
                  >
                    <RightPanelIcon isActive={showDatasetView} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Toggle Dataset View</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
        {showChat && <AIChatPanel onClose={() => setShowChat(false)} />}
        {/* Main content */}
        <ResizablePanelGroup direction="horizontal" className="flex-grow">
          {/* File Explorer and Bookmarks */}
          {showFileExplorer && (
            <ResizablePanel defaultSize={10} minSize={6}>
              <ResizablePanelGroup direction="vertical">
                <ResizablePanel defaultSize={40} minSize={20}>
                  <FileExplorer
                    files={files}
                    onFileClick={(file) => {
                      setCurrentFile(file);
                    }}
                    onFileUpload={(file: File) =>
                      setFiles((prevFiles) => [...prevFiles, file])
                    }
                    onFileDelete={(file: File) => {
                      setFiles((prevFiles) =>
                        prevFiles.filter((f) => f.name !== file.name)
                      );
                    }}
                    setCurrentFile={setCurrentFile}
                    setShowDatasetView={setShowDatasetView}
                    currentFile={currentFile}
                  />
                </ResizablePanel>
                <ResizableHandle
                  withHandle
                  className="h-2 bg-gray-200 hover:bg-gray-300 transition-colors duration-200"
                />
                <ResizablePanel defaultSize={60} minSize={20}>
                  <BookmarksPanel />
                </ResizablePanel>
              </ResizablePanelGroup>
            </ResizablePanel>
          )}
          {showFileExplorer && (
            <ResizableHandle
              withHandle
              className="w-2 bg-gray-200 hover:bg-gray-300 transition-colors duration-200"
            />
          )}

          {/* Pipeline GUI and Output */}
          <ResizablePanel defaultSize={60} minSize={30}>
            <ResizablePanelGroup direction="vertical">
              <ResizablePanel defaultSize={70} minSize={5}>
                <PipelineGUI />
              </ResizablePanel>
              {showOutput && (
                <ResizableHandle
                  withHandle
                  className="h-2 bg-gray-200 hover:bg-gray-300 transition-colors duration-200"
                />
              )}
              {showOutput && (
                <ResizablePanel defaultSize={105} minSize={20}>
                  <Output />
                </ResizablePanel>
              )}
            </ResizablePanelGroup>
          </ResizablePanel>

          {/* Dataset View */}
          {showDatasetView && (
            <ResizableHandle
              withHandle
              className="w-2 bg-gray-200 hover:bg-gray-300 transition-colors duration-200"
            />
          )}
          {showDatasetView && currentFile && (
            <ResizablePanel defaultSize={20} minSize={10}>
              <DatasetView file={currentFile} />
            </ResizablePanel>
          )}
        </ResizablePanelGroup>
      </div>
    </BookmarkProvider>
  );
};

const WrappedCodeEditorPipelineApp: React.FC = () => (
  <WebSocketProvider>
    <PipelineProvider>
      <CodeEditorPipelineApp />
    </PipelineProvider>
  </WebSocketProvider>
);

export default WrappedCodeEditorPipelineApp;
