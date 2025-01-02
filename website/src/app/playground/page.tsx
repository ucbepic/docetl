"use client";

import dynamic from "next/dynamic";
import React, { useEffect, useState, useRef, Suspense } from "react";
import {
  Scroll,
  Info,
  Save,
  Monitor,
  AlertCircle,
  Loader2,
} from "lucide-react";
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
    loading: () => (
      <div className="h-full flex items-center justify-center">
        <div className="animate-spin h-6 w-6 border-2 border-primary border-r-transparent rounded-full" />
      </div>
    ),
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
import { APIKeysDialog } from "@/components/APIKeysDialog";
import { TutorialsDialog, TUTORIALS } from "@/components/TutorialsDialog";

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

const MobileWarning: React.FC = () => (
  <div className="h-screen flex items-center justify-center p-4 bg-background">
    <div className="max-w-md text-center p-8 bg-card rounded-[var(--radius)] shadow-lg border border-border">
      <div className="flex justify-center mb-6">
        <div className="relative">
          <Monitor className="w-16 h-16 text-primary" />
          <div className="absolute -top-1 -right-1">
            <AlertCircle className="w-6 h-6 text-destructive" />
          </div>
        </div>
      </div>
      <h2 className="text-2xl font-bold text-foreground mb-4">
        Desktop Required
      </h2>
      <p className="text-muted-foreground mb-6 leading-relaxed">
        DocWrangler requires a larger screen for the best experience. Please
        switch to a desktop or laptop computer to access all features.
      </p>
      <div className="text-sm text-muted-foreground/80 bg-muted px-4 py-2 rounded-[var(--radius)] mb-6">
        Recommended minimum screen width: 768px
      </div>
      <Button
        variant="default"
        className="bg-primary hover:bg-primary/90 text-primary-foreground"
        onClick={() => {
          window.location.href = "/";
        }}
      >
        Back to DocETL Home
      </Button>
    </div>
  </div>
);
const LoadingScreen: React.FC = () => (
  <div className="h-screen flex flex-col items-center justify-center gap-6 bg-background">
    <div className="relative">
      <Loader2 className="h-12 w-12 animate-spin text-primary" />
    </div>
    <div className="flex flex-col items-center gap-2">
      <div className="flex items-center gap-2">
        <Scroll className="h-6 w-6 text-primary" />
        <h2 className="text-2xl font-bold text-primary tracking-tight">
          DocWrangler
        </h2>
      </div>
      <div className="text-muted-foreground text-lg">
        <span className="inline-block animate-pulse">Loading...</span>
      </div>
    </div>
  </div>
);

const PerformanceWrapper: React.FC<{
  children: React.ReactNode;
  className?: string;
}> = ({ children, className }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [size, setSize] = useState<{ width: number; height: number }>();
  const ref = useRef<HTMLDivElement>(null);

  // Capture size on mount and resize
  useEffect(() => {
    if (ref.current) {
      const observer = new ResizeObserver((entries) => {
        if (!isDragging) {
          const { width, height } = entries[0].contentRect;
          setSize({ width, height });
        }
      });

      observer.observe(ref.current);
      return () => observer.disconnect();
    }
  }, [isDragging]);

  return (
    <div
      ref={ref}
      className={className}
      style={{
        visibility: isDragging ? "hidden" : "visible",
      }}
      data-dragging={isDragging}
    >
      {children}
    </div>
  );
};

const CodeEditorPipelineApp: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [isMobileView, setIsMobileView] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const [showFileExplorer, setShowFileExplorer] = useState(true);
  const [showOutput, setShowOutput] = useState(true);
  const [showDatasetView, setShowDatasetView] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [showNamespaceDialog, setShowNamespaceDialog] = useState(false);
  const [showAPIKeysDialog, setShowAPIKeysDialog] = useState(false);
  const [showTutorialsDialog, setShowTutorialsDialog] = useState(false);
  const [selectedTutorial, setSelectedTutorial] =
    useState<(typeof TUTORIALS)[0]>();
  const { theme, setTheme } = useTheme();

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
    setOperations,
    setPipelineName,
    setSampleSize,
    setDefaultModel,
    setSystemPrompt,
  } = usePipelineContext();

  useEffect(() => {
    const checkScreenSize = () => {
      const isMobile = window.innerWidth < 768;
      setIsMobileView(isMobile);
      setIsLoading(false);
    };

    checkScreenSize();
    setIsMounted(true);
    window.addEventListener("resize", checkScreenSize);

    return () => window.removeEventListener("resize", checkScreenSize);
  }, []);

  useEffect(() => {
    if (isMounted && !namespace) {
      setShowNamespaceDialog(true);
    }
  }, [isMounted, namespace]);

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (isMobileView) {
    return <MobileWarning />;
  }

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

  const handleNew = () => {
    clearPipelineState();
    if (!namespace) {
      setShowNamespaceDialog(true);
    } else {
      window.location.reload();
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
  const resizeHandleStyles = `
    w-2 bg-gray-100 hover:bg-blue-200 transition-colors duration-200
    data-[dragging=true]:bg-blue-400
  `;

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
                        <AlertDialogAction onClick={handleNew}>
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
                  <MenubarItem onSelect={() => setShowAPIKeysDialog(true)}>
                    Edit API Keys
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
                        <MenubarRadioItem value="majestic">
                          Majestic
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
                  <MenubarItem
                    onSelect={() =>
                      window.open(
                        "https://discord.com/invite/fHp7B2X3xx",
                        "_blank"
                      )
                    }
                  >
                    Ask a Question on Discord
                  </MenubarItem>
                  <MenubarSub>
                    <MenubarSubTrigger>Tutorials</MenubarSubTrigger>
                    <MenubarSubContent>
                      {TUTORIALS.map((tutorial) => (
                        <MenubarItem
                          key={tutorial.id}
                          onSelect={() => {
                            setSelectedTutorial(tutorial);
                            setShowTutorialsDialog(true);
                          }}
                        >
                          {tutorial.title}
                        </MenubarItem>
                      ))}
                    </MenubarSubContent>
                  </MenubarSub>
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
              <PopoverContent className="w-[32rem]">
                <h3 className="text-lg font-semibold mb-4 text-foreground">
                  About DocWrangler and DocETL
                </h3>
                <div className="space-y-4 text-sm">
                  <p className="text-foreground/90 leading-relaxed">
                    DocWrangler and DocETL are research projects from UC
                    Berkeley's EPIC Data Lab. DocWrangler provides an
                    interactive playground for building data processing
                    pipelines, powered by DocETL - our system that combines a
                    domain-specific language, query optimizer, and execution
                    engine. Learn more at{" "}
                    <a
                      href="https://docetl.org"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline font-medium"
                    >
                      docetl.org
                    </a>
                    .
                  </p>
                  <p className="text-foreground/90 leading-relaxed">
                    DocWrangler's AI Chat and Improve Prompt features use our
                    experimental LLM and log usage data. For privacy, you can
                    use your own API key instead by going to Edit &gt; Edit API
                    keys and enabling &quot;Use personal API key&quot; in the
                    features.
                  </p>
                  <p className="text-foreground/90 leading-relaxed">
                    Want to run DocETL or the playground locally? Check out our{" "}
                    <a
                      href="https://ucbepic.github.io/docetl/playground/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline font-medium"
                    >
                      self-hosted version
                    </a>
                    . For DocETL documentation, visit{" "}
                    <a
                      href="https://ucbepic.github.io/docetl"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline font-medium"
                    >
                      our docs
                    </a>
                    . Have questions? Join our{" "}
                    <a
                      href="https://discord.com/invite/fHp7B2X3xx"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline font-medium"
                    >
                      Discord community
                    </a>
                    .
                  </p>
                </div>
              </PopoverContent>
            </Popover>
          </div>
          <div className="flex items-center gap-2">
            <Scroll className="text-primary" size={20} />
            <h1 className="text-lg font-bold text-primary">DocWrangler</h1>
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
          onDragStart={() => (document.body.style.cursor = "col-resize")}
          onDragEnd={() => (document.body.style.cursor = "default")}
        >
          {showFileExplorer && (
            <ResizablePanel defaultSize={10} minSize={6} className="h-full">
              <ResizablePanelGroup
                direction="vertical"
                className="h-full"
                onDragStart={() => (document.body.style.cursor = "row-resize")}
                onDragEnd={() => (document.body.style.cursor = "default")}
              >
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
            <ResizablePanelGroup
              direction="vertical"
              className="h-full"
              onDragStart={() => (document.body.style.cursor = "row-resize")}
              onDragEnd={() => (document.body.style.cursor = "default")}
            >
              <ResizablePanel
                defaultSize={60}
                minSize={5}
                className="overflow-auto"
              >
                <PerformanceWrapper className="h-full">
                  <PipelineGUI />
                </PerformanceWrapper>
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
                  <PerformanceWrapper className="h-full">
                    <Output />
                  </PerformanceWrapper>
                </ResizablePanel>
              )}
            </ResizablePanelGroup>
          </ResizablePanel>

          {showDatasetView && (
            <>
              <ResizableHandle withHandle className={resizeHandleStyles} />
              <ResizablePanel
                defaultSize={20}
                minSize={10}
                className="h-full overflow-auto"
              >
                <PerformanceWrapper className="h-full">
                  <Suspense
                    fallback={
                      <div className="h-full flex items-center justify-center">
                        <div className="animate-spin h-6 w-6 border-2 border-primary border-r-transparent rounded-full" />
                      </div>
                    }
                  >
                    <DatasetView file={currentFile} />
                  </Suspense>
                </PerformanceWrapper>
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
        <APIKeysDialog
          open={showAPIKeysDialog}
          onOpenChange={setShowAPIKeysDialog}
        />
        <TutorialsDialog
          open={showTutorialsDialog}
          onOpenChange={setShowTutorialsDialog}
          selectedTutorial={selectedTutorial}
          namespace={namespace}
          onFileUpload={(file: File) =>
            setFiles((prevFiles) => [...prevFiles, file])
          }
          setCurrentFile={setCurrentFile}
          setOperations={setOperations}
          setPipelineName={setPipelineName}
          setSampleSize={setSampleSize}
          setDefaultModel={setDefaultModel}
          setFiles={setFiles}
          setSystemPrompt={setSystemPrompt}
          currentFile={currentFile}
          files={files}
        />
      </div>
    </BookmarkProvider>
  );
};

const WebSocketWrapper: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const { namespace } = usePipelineContext();

  return (
    <WebSocketProvider namespace={namespace || ""}>
      {children}
    </WebSocketProvider>
  );
};

const WrappedCodeEditorPipelineApp: React.FC = () => {
  return (
    <ThemeProvider>
      <PipelineProvider>
        <WebSocketWrapper>
          <CodeEditorPipelineApp />
        </WebSocketWrapper>
      </PipelineProvider>
    </ThemeProvider>
  );
};

export default WrappedCodeEditorPipelineApp;
