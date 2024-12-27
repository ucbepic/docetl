import React from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useState, useEffect } from "react";
import { FolderKanban } from "lucide-react";
import { toast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

const adjectives = [
  "swift",
  "bright",
  "clever",
  "gentle",
  "happy",
  "noble",
  "brave",
  "calm",
  "wise",
  "kind",
  "quick",
  "silent",
  "cosmic",
  "azure",
  "golden",
  "silver",
  "crystal",
  "autumn",
  "spring",
  "winter",
];

const nouns = [
  "fox",
  "eagle",
  "wolf",
  "bear",
  "hawk",
  "lion",
  "tiger",
  "deer",
  "owl",
  "dove",
  "raven",
  "phoenix",
  "dragon",
  "falcon",
  "panda",
  "koala",
  "lynx",
  "shark",
  "whale",
  "dolphin",
];

function generateRandomNamespace(): string {
  const adjective = adjectives[Math.floor(Math.random() * adjectives.length)];
  const noun = nouns[Math.floor(Math.random() * nouns.length)];
  const randomStr = Math.random().toString(36).substring(2, 10);
  return `${adjective}-${noun}-${randomStr}`;
}

interface NamespaceDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  currentNamespace: string | null;
  onSave: (namespace: string) => void;
}

export function NamespaceDialog({
  open,
  onOpenChange,
  currentNamespace,
  onSave,
}: NamespaceDialogProps) {
  const [namespace, setNamespace] = useState(currentNamespace || "");
  const [isChecking, setIsChecking] = useState(false);
  const [showWarning, setShowWarning] = useState(false);
  const [shake, setShake] = useState(false);

  useEffect(() => {
    setNamespace(currentNamespace || generateRandomNamespace());
  }, [currentNamespace]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const trimmedNamespace = namespace.trim();

    if (!trimmedNamespace || trimmedNamespace.length === 0) {
      toast({
        title: "Invalid Namespace",
        description:
          "Namespace cannot be empty. Please enter a valid namespace.",
        variant: "destructive",
      });
      setShake(true);
      setTimeout(() => setShake(false), 500);
      return;
    }

    setIsChecking(true);
    try {
      const response = await fetch("/api/checkNamespace", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ namespace: trimmedNamespace }),
      });

      const data = await response.json();

      if (data.exists && !showWarning) {
        setShowWarning(true);
        setShake(true);
        setTimeout(() => setShake(false), 500);
      } else {
        onSave(trimmedNamespace);
        setShowWarning(false);
        setShake(false);
        window.location.reload();
      }
    } catch (error) {
      console.error("Error checking namespace:", error);
      toast({
        title: "Error",
        description: "Failed to check namespace availability",
        variant: "destructive",
      });
    } finally {
      setIsChecking(false);
    }
  };

  const hasNamespaceChanged =
    namespace.trim() !== (currentNamespace || "").trim() &&
    namespace.trim().length > 0;

  const handleOpenChange = (newOpen: boolean) => {
    if (!newOpen && !currentNamespace?.trim()) {
      toast({
        title: "Namespace Required",
        description: "Please set a namespace before continuing.",
        variant: "destructive",
      });
      setShake(true);
      setTimeout(() => setShake(false), 500);
      return;
    }
    onOpenChange(newOpen);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className={cn("sm:max-w-lg", shake && "animate-shake")}>
        <DialogHeader>
          <div className="flex items-center gap-2">
            <FolderKanban className="h-6 w-6 text-primary" />
            <DialogTitle className="text-xl">Set Namespace</DialogTitle>
          </div>
          <DialogDescription className="text-sm text-muted-foreground pt-2">
            Enter a namespace to organize your pipeline configurations. This
            helps keep your work separate from others on the same server.
            {currentNamespace && (
              <div className="mt-2 text-orange-700 dark:text-orange-200 bg-orange-100 dark:bg-orange-950 border border-orange-300 dark:border-orange-800 rounded-md p-2 font-medium">
                Note: Changing the namespace will clear your current workspace.
              </div>
            )}
          </DialogDescription>
        </DialogHeader>
        <div className="py-2">
          <div className="space-y-2">
            <Label htmlFor="namespace" className="text-sm font-medium">
              Namespace <span className="text-red-500">*</span>
            </Label>
            <Input
              id="namespace"
              placeholder="e.g., swift-fox-1234"
              value={namespace}
              onChange={(e) => {
                setNamespace(e.target.value);
                setShowWarning(false);
              }}
              onBlur={(e) => setNamespace(e.target.value.trim())}
              className={cn(
                "w-full",
                isChecking ? "border-red-500" : "",
                !namespace.trim() && "border-red-300"
              )}
              required
            />
            {isChecking && <p className="text-xs text-red-500">Checking...</p>}
            {showWarning && (
              <div className="text-sm text-orange-700 dark:text-orange-200 bg-orange-100 dark:bg-orange-950 border border-orange-300 dark:border-orange-800 rounded-md p-2 font-medium">
                Warning: This namespace already exists. Setting this namespace
                may overwrite another user&apos;s existing caches, but feel free
                to ignore this message if this is your namespace.
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              Use your username or keep the randomly generated namespace
            </p>
          </div>
        </div>
        <DialogFooter className="flex justify-end gap-2">
          <Button
            variant="outline"
            onClick={() => handleOpenChange(false)}
            disabled={!currentNamespace?.trim()}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            className="px-6"
            disabled={isChecking || !hasNamespaceChanged}
            variant={showWarning ? "destructive" : "default"}
          >
            {showWarning ? "Set Namespace Anyway" : "Set Namespace"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
