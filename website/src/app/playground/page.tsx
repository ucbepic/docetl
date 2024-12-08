"use client";

import dynamic from "next/dynamic";
import React, { useEffect, useState } from "react";
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
  MenubarSub,
  MenubarSubContent,
  MenubarSubTrigger,
  MenubarRadioGroup,
  MenubarRadioItem,
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
const NamespaceDialog = dynamic(
  () =>
    import("@/components/NamespaceDialog").then((mod) => mod.NamespaceDialog),
  {
    ssr: false,
  }
);
import { ThemeProvider, useTheme, Theme } from "@/contexts/ThemeContext";

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
  const [isMounted, setIsMounted] = useState(false);
  const [showFileExplorer, setShowFileExplorer] = useState(true);
  const [showOutput, setShowOutput] = useState(true);
  const [showDatasetView, setShowDatasetView] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [showNamespaceDialog, setShowNamespaceDialog] = useState(false);
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const {
    currentFile,
    setCurrentFile,
    cost,
    files,
    setFiles,
    clearPipelineState,
    saveProgress,
    unsavedChanges,
    namespace,
    setNamespace,
  } = usePipelineContext();

  useEffect(() => {
    const savedNamespace = localStorage.getItem(localStorageKeys.NAMESPACE_KEY);
    if (!savedNamespace) {
      setShowNamespaceDialog(true);
    }
  }, []);

  const handleSaveAs = async () => {
    try {
      // Collect all localStorage data
      const data: Record<string, unknown> = {};
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

  const topBarStyles =
    "p-2 flex justify-between items-center border-b bg-white shadow-sm";
  const controlGroupStyles = "flex items-center gap-2";
  const panelControlsStyles = "flex items-center gap-1 px-2 border-l";
  const saveButtonStyles = `relative h-8 px-3 ${
    unsavedChanges
      ? "bg-orange-100 border-orange-500 hover:bg-orange-200"
      : "hover:bg-gray-100"
  }`;
  const costDisplayStyles =
    "px-3 py-1.5 text-sm text-gray-600 flex items-center gap-1";
  const panelToggleStyles =
    "flex items-center gap-2 px-3 py-1.5 rounded-md transition-colors duration-200";
  const mainContentStyles = "flex-grow overflow-hidden bg-gray-50";
  const resizeHandleStyles =
    "w-2 bg-gray-100 hover:bg-blue-200 transition-colors duration-200";

  return (
    <BookmarkProvider>
      <div className="h-screen flex flex-col bg-gray-50">
        <div className={topBarStyles}>
          <div className={controlGroupStyles}>
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
                <MenubarTrigger>Edit</MenubarTrigger>
                <MenubarContent>
                  <MenubarItem onSelect={() => setShowNamespaceDialog(true)}>
                    Change Namespace
                  </MenubarItem>
                  <MenubarSub>
                    <MenubarSubTrigger>Change Theme</MenubarSubTrigger>
                    <MenubarSubContent>
                      <MenubarRadioGroup
                        value={theme}
                        onValueChange={(value) => setTheme(value as Theme)}
                      >
                        <MenubarRadioItem value="default">
                          Default
                        </MenubarRadioItem>
                        <MenubarRadioItem value="forest">
                          Forest
                        </MenubarRadioItem>
                        <MenubarRadioItem value="magestic">
                          Magestic
                        </MenubarRadioItem>
                        <MenubarRadioItem value="sunset">
                          Sunset
                        </MenubarRadioItem>
                        <MenubarRadioItem value="ruby">Ruby</MenubarRadioItem>
                        <MenubarRadioItem value="monochrome">
                          Monochrome
                        </MenubarRadioItem>
                      </MenubarRadioGroup>
                    </MenubarSubContent>
                  </MenubarSub>
                </MenubarContent>
              </MenubarMenu>
              <MenubarMenu>
                <MenubarTrigger>Help</MenubarTrigger>
                <MenubarContent>
                  <MenubarItem
                    onSelect={() =>
                      window.open("https://ucbepic.github.io/docetl/", "_blank")
                    }
                  >
                    Show Documentation
                  </MenubarItem>
                  <MenubarItem onSelect={() => setShowChat(!showChat)}>
                    Show Chat
                  </MenubarItem>
                </MenubarContent>
              </MenubarMenu>
            </Menubar>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      saveProgress();
                      toast({
                        title: "Progress Saved",
                        description:
                          "Your pipeline progress has been saved to browser storage.",
                        duration: 3000,
                      });
                    }}
                    className={saveButtonStyles}
                  >
                    <Save
                      size={16}
                      className={
                        unsavedChanges ? "text-orange-500 mr-2" : "mr-2"
                      }
                    />
                    {unsavedChanges ? "Quick Save" : "Quick Save"}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {unsavedChanges
                    ? "Save changes to browser storage (use File > Save As to save to disk)"
                    : "No changes compared to the version in browser storage"}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="ghost" size="sm" className="px-2">
                  <Info size={16} className="mr-2" />
                  Info
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
          </div>
          <div className="flex items-center gap-2">
            <Scroll className="text-primary" size={20} />
            <h1 className="text-lg font-bold text-primary">DocETL</h1>
            {isMounted && (
              <span className="text-sm text-gray-600">({namespace})</span>
            )}
          </div>
          <div className={controlGroupStyles}>
            {isMounted && (
              <div className={costDisplayStyles}>
                <span className="text-gray-500">Cost:</span>
                <span className="font-medium">${cost.toFixed(2)}</span>
              </div>
            )}
            <div className={panelControlsStyles}>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      onClick={() => setShowFileExplorer(!showFileExplorer)}
                      className={panelToggleStyles}
                    >
                      <LeftPanelIcon isActive={showFileExplorer} />
                      Files
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Toggle File Explorer</TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      onClick={() => setShowOutput(!showOutput)}
                      className={panelToggleStyles}
                    >
                      <BottomPanelIcon isActive={showOutput} />
                      Output
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Toggle Output Panel</TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      onClick={() => setShowDatasetView(!showDatasetView)}
                      className={panelToggleStyles}
                    >
                      <RightPanelIcon isActive={showDatasetView} />
                      Dataset
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Toggle Dataset View</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
        </div>
        {showChat && <AIChatPanel onClose={() => setShowChat(false)} />}
        {/* Main content */}
        <ResizablePanelGroup
          direction="horizontal"
          className={mainContentStyles}
        >
          {showFileExplorer && (
            <ResizablePanel defaultSize={10} minSize={6} className="h-full">
              <ResizablePanelGroup direction="vertical" className="h-full">
                <ResizablePanel
                  defaultSize={40}
                  minSize={20}
                  className="overflow-auto"
                >
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
                    namespace={namespace}
                  />
                </ResizablePanel>
                <ResizableHandle withHandle className={resizeHandleStyles} />
                <ResizablePanel
                  defaultSize={60}
                  minSize={20}
                  className="overflow-auto"
                >
                  <BookmarksPanel />
                </ResizablePanel>
              </ResizablePanelGroup>
            </ResizablePanel>
          )}
          {showFileExplorer && (
            <ResizableHandle withHandle className={resizeHandleStyles} />
          )}

          <ResizablePanel defaultSize={60} minSize={30} className="h-full">
            <ResizablePanelGroup direction="vertical" className="h-full">
              <ResizablePanel
                defaultSize={60}
                minSize={5}
                className="overflow-auto"
              >
                <PipelineGUI />
              </ResizablePanel>
              {showOutput && (
                <ResizableHandle withHandle className={resizeHandleStyles} />
              )}
              {showOutput && (
                <ResizablePanel
                  defaultSize={40}
                  minSize={20}
                  className="overflow-auto"
                >
                  <Output />
                </ResizablePanel>
              )}
            </ResizablePanelGroup>
          </ResizablePanel>

          {showDatasetView && currentFile && (
            <>
              <ResizableHandle withHandle className={resizeHandleStyles} />
              <ResizablePanel
                defaultSize={20}
                minSize={10}
                className="h-full overflow-auto"
              >
                <DatasetView file={currentFile} />
              </ResizablePanel>
            </>
          )}
        </ResizablePanelGroup>
        <NamespaceDialog
          open={showNamespaceDialog}
          onOpenChange={setShowNamespaceDialog}
          currentNamespace={namespace}
          onSave={(newNamespace) => {
            setNamespace(newNamespace);
            setShowNamespaceDialog(false);
            saveProgress();
          }}
        />
      </div>
    </BookmarkProvider>
  );
};

const WrappedCodeEditorPipelineApp: React.FC = () => {
  const [isLocalhost, setIsLocalhost] = useState(true);

  useEffect(() => {
    setIsLocalhost(
      window.location.hostname === "localhost" ||
        window.location.hostname === "127.0.0.1"
    );
  }, []);

  if (!isLocalhost) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="max-w-2xl p-6 bg-card rounded-lg shadow-md">
          <h1 className="text-2xl font-bold text-primary mb-4">
            DocETL Playground
          </h1>
          <p className="mb-4 text-foreground">
            The DocETL playground is designed to run locally. To use it, please
            follow these steps:
          </p>
          <ol className="list-decimal list-inside mb-4 text-foreground">
            <li>
              Clone the GitHub repo:{" "}
              <a
                href="https://github.com/ucbepic/docetl"
                className="text-primary hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                https://github.com/ucbepic/docetl
              </a>
            </li>
            <li>
              Set up the project by running:
              <pre className="bg-muted text-muted-foreground p-2 rounded mt-2 mb-2">
                make install
              </pre>
              <pre className="bg-muted text-muted-foreground p-2 rounded mt-2 mb-2">
                make install-ui
              </pre>
            </li>
            <li>
              Start the application:
              <pre className="bg-muted text-muted-foreground p-2 rounded mt-2 mb-2">
                make run-ui-prod
              </pre>
            </li>
            <li>
              Navigate to{" "}
              <a
                href="http://localhost:3000/playground"
                className="text-primary hover:underline"
              >
                http://localhost:3000/playground
              </a>
            </li>
          </ol>
          <p className="text-foreground">
            Once you&apos;ve completed these steps, you&apos;ll be able to use
            the DocETL playground locally.
          </p>
        </div>
      </div>
    );
  }

  return (
    <ThemeProvider>
      <WebSocketProvider>
        <PipelineProvider>
          <CodeEditorPipelineApp />
        </PipelineProvider>
      </WebSocketProvider>
    </ThemeProvider>
  );
};

export default WrappedCodeEditorPipelineApp;
