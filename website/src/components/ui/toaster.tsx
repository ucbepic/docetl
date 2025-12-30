"use client";

import React from "react";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Copy } from "lucide-react";
import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from "@/components/ui/toast";

export function Toaster() {
  const { toasts } = useToast();

  return (
    <ToastProvider>
      {toasts.map(function ({ id, title, description, action, ...props }) {
        const descId = `toast-desc-${id}`;
        return (
          <Toast key={id} {...props}>
            <div className="grid gap-1 flex-1 overflow-hidden">
              <div className="flex items-center gap-2 shrink-0">
                {title && <ToastTitle>{title}</ToastTitle>}
                {description && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-5 w-5 p-0 opacity-70 hover:opacity-100"
                    onClick={() => {
                      const el = document.getElementById(descId);
                      const text = el?.textContent ?? "";
                      if (text) {
                        navigator.clipboard.writeText(text);
                      }
                    }}
                    title="Copy message"
                  >
                    <Copy size={12} />
                  </Button>
                )}
              </div>
              {description && (
                <div className="overflow-y-auto max-h-[60vh]">
                  <ToastDescription id={descId}>{description}</ToastDescription>
                </div>
              )}
            </div>
            {action}
            <ToastClose />
          </Toast>
        );
      })}
      <ToastViewport />
    </ToastProvider>
  );
}
