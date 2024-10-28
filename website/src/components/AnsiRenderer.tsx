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
    if (userInput.trim()) {
      sendMessage(userInput);
      setUserInput("");
    }
  };

  return (
    <div className="flex flex-col w-full h-[500px] bg-black text-white font-mono rounded-lg overflow-hidden">
      <div
        ref={scrollRef}
        className="flex-grow overflow-auto p-4"
        style={{
          height: "400px", // Reduced height to accommodate input field
          maxHeight: "400px",
        }}
      >
        <pre
          className="m-0 whitespace-pre-wrap break-words"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>
      <div className="p-2 border-t border-gray-700">
        <div className="flex mb-2">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
            className="flex-grow bg-gray-800 text-white px-2 py-1 rounded-l"
            placeholder="Type a message..."
          />
          <button
            onClick={handleSendMessage}
            className="bg-blue-500 text-white px-4 py-1 rounded-r"
          >
            Send
          </button>
        </div>
        <div className="flex justify-between items-center">
          <div className="text-xs text-gray-500">
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
            className="text-xs text-gray-500"
          >
            Clear Output
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnsiRenderer;
