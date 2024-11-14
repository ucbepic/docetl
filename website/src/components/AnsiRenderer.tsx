import React, { useEffect, useRef, useState } from "react";
import Convert from "ansi-to-html";
import { useWebSocket } from "@/contexts/WebSocketContext";
import { useToast } from "@/hooks/use-toast";

const convert = new Convert({
  fg: "#000",
  bg: "#fff",
  newline: true,
  escapeXML: true,
  stream: false,
});

interface AnsiRendererProps {
  text: string;
  readyState: number;
  setTerminalOutput: (text: string) => void;
}

const AnsiRenderer: React.FC<AnsiRendererProps> = ({
  text,
  readyState,
  setTerminalOutput,
}) => {
  const html = convert.toHtml(text);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [userInput, setUserInput] = useState("");
  const { sendMessage } = useWebSocket();
  const { toast } = useToast();

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [text]);

  const handleSendMessage = () => {
    const trimmedInput = userInput.trim();
    if (trimmedInput) {
      sendMessage(trimmedInput);
      toast({
        title: "Terminal input received",
        description: `You sent: ${trimmedInput}`,
      });
      setUserInput("");
    }
  };

  const isWebSocketClosed = readyState === WebSocket.CLOSED;

  return (
    <div
      className={`flex flex-col h-full bg-black text-white font-mono rounded-lg overflow-hidden ${
        isWebSocketClosed ? "opacity-50" : ""
      }`}
    >
      <div ref={scrollRef} className="flex-1 min-h-0 overflow-auto p-4">
        <pre
          className="m-0 whitespace-pre-wrap break-words"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>
      <div className="flex-none p-2 border-t border-gray-700">
        <div className="flex mb-2">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
            className={`flex-grow bg-gray-800 text-white px-2 py-1 rounded-l ${
              isWebSocketClosed ? "cursor-not-allowed" : ""
            }`}
            placeholder={
              isWebSocketClosed
                ? "WebSocket disconnected..."
                : "Type a message..."
            }
            disabled={isWebSocketClosed}
          />
          <button
            onClick={handleSendMessage}
            className={`bg-blue-500 text-white px-4 py-1 rounded-r ${
              isWebSocketClosed ? "cursor-not-allowed opacity-50" : ""
            }`}
            disabled={isWebSocketClosed}
          >
            Send
          </button>
        </div>
        <div className="flex justify-between items-center">
          <div
            className={`text-xs ${
              isWebSocketClosed ? "text-red-500" : "text-gray-500"
            }`}
          >
            WebSocket State:{" "}
            {readyState === WebSocket.CONNECTING
              ? "Connecting"
              : readyState === WebSocket.OPEN
              ? "Open"
              : readyState === WebSocket.CLOSING
              ? "Closing"
              : readyState === WebSocket.CLOSED
              ? "Closed"
              : "Unknown"}
          </div>
          <button
            onClick={() => setTerminalOutput("")}
            className={`text-xs text-gray-500 ${
              isWebSocketClosed ? "cursor-not-allowed opacity-50" : ""
            }`}
            disabled={isWebSocketClosed}
          >
            Clear Output
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnsiRenderer;
