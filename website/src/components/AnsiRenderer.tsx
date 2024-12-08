import React, { useEffect, useRef, useState } from "react";
import Convert from "ansi-to-html";
import { useWebSocket } from "@/contexts/WebSocketContext";

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

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [text]);

  const handleSendMessage = () => {
    const trimmedInput = userInput.trim();
    if (trimmedInput) {
      sendMessage(trimmedInput);
      setTerminalOutput(text + "\n$ " + trimmedInput);
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
        <div className="flex items-center mb-2">
          <span className="text-green-500 mr-2">$</span>
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
            className={`flex-grow bg-transparent text-white outline-none ${
              isWebSocketClosed ? "cursor-not-allowed" : ""
            }`}
            placeholder={isWebSocketClosed ? "WebSocket disconnected..." : ""}
            disabled={isWebSocketClosed}
          />
          {userInput.trim() && !isWebSocketClosed && (
            <button
              onClick={handleSendMessage}
              className="ml-2 px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
            >
              ⏎
            </button>
          )}
        </div>
        <div className="flex justify-between items-center text-xs text-gray-500">
          <div className={isWebSocketClosed ? "text-red-500" : ""}>
            Status:{" "}
            {readyState === WebSocket.CONNECTING
              ? "Connecting"
              : readyState === WebSocket.OPEN
              ? "Connected"
              : readyState === WebSocket.CLOSING
              ? "Closing"
              : readyState === WebSocket.CLOSED
              ? "Disconnected"
              : "Unknown"}
          </div>
          <button
            onClick={() => setTerminalOutput("")}
            className={`hover:text-white transition-colors ${
              isWebSocketClosed ? "cursor-not-allowed opacity-50" : ""
            }`}
            disabled={isWebSocketClosed}
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnsiRenderer;
