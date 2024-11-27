// DocumentViewer.tsx
import React, { useEffect, useState } from "react";
import DocViewer, { DocViewerRenderers } from "@cyntler/react-doc-viewer";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import "@cyntler/react-doc-viewer/dist/index.css";

interface DocumentViewerProps {
  isOpen: boolean;
  onClose: () => void;
  filePath: string;
  fileName: string;
}

export const DocumentViewer: React.FC<DocumentViewerProps> = ({
  isOpen,
  onClose,
  filePath,
  fileName,
}) => {
  const [documentUrl, setDocumentUrl] = useState<string>("");

  useEffect(() => {
    if (isOpen && filePath) {
      // Convert the full file path to a URL-safe format
      const encodedPath = encodeURIComponent(filePath);
      const url = `/api/serveDocument/${encodedPath}`;
      setDocumentUrl(url);
    }
  }, [isOpen, filePath]);

  const docs = documentUrl ? [{ uri: documentUrl, fileName: fileName }] : [];

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-[calc(100vh*0.75)] h-[100vh] p-0 overflow-hidden">
        <DialogHeader className="p-4 pb-0">
          <DialogTitle className="pr-8">{fileName}</DialogTitle>
        </DialogHeader>
        <div className="flex-1 p-4 pt-0 overflow-hidden">
          <div className="h-full w-full overflow-hidden">
            {documentUrl && (
              <DocViewer
                pluginRenderers={DocViewerRenderers}
                documents={docs}
                initialActiveDocument={docs[0]}
                style={{
                  height: "100%",
                  width: "100%",
                  maxHeight: "100%",
                  overflow: "auto",
                  backgroundColor: "white",
                }}
                config={{
                  header: {
                    disableHeader: true,
                    disableFileName: true,
                  },
                  pdfZoom: {
                    defaultZoom: 1,
                    zoomJump: 0.2,
                  },
                  pdfVerticalScrollByDefault: true,
                }}
              />
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
