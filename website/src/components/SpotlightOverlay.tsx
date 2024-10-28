import React, { useState, ReactNode, useEffect } from "react";
import {
  Bot,
  MousePointer,
  MessageCircle,
  User,
  RefreshCw,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

import dynamic from "next/dynamic";

const Rnd = dynamic(() => import("react-rnd").then((mod) => mod.Rnd), {
  ssr: false,
});

interface SpotlightOverlayProps {
  children: ReactNode;
}

const SpotlightOverlay: React.FC<SpotlightOverlayProps> = ({ children }) => {
  const [selectedTool, setSelectedTool] = useState<string>("cursor");
  const [showChatBox, setShowChatBox] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatPosition, setChatPosition] = useState({ x: 0, y: 0 });
  const [chatSize, setChatSize] = useState({ width: 400, height: 400 });
  const [chatHistory, setChatHistory] = useState<
    { role: "user" | "assistant"; content: string }[]
  >([]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      setChatPosition({
        x: window.innerWidth / 2 - 200,
        y: window.innerHeight / 2 - 200,
      });
    }
  }, []);

  const handleChatSubmit = () => {
    if (chatInput.trim() === "") return;

    // Add user message to chat history
    const newUserMessage = { role: "user" as const, content: chatInput };
    setChatHistory((prevHistory) => [...prevHistory, newUserMessage]);

    // Here you would typically send the chatInput to your AI service
    console.log("Sending to AI:", { userInput: chatInput });

    // Simulate AI response (replace this with actual AI integration)
    setTimeout(() => {
      const aiResponse = {
        role: "assistant" as const,
        content: `AI response to: ${chatInput}`,
      };
      setChatHistory((prevHistory) => [...prevHistory, aiResponse]);
    }, 1000);

    // Reset chat input after sending
    setChatInput("");
  };

  const handleToolSelect = (tool: string) => {
    setSelectedTool(tool);
    if (tool === "chat") {
      setShowChatBox(true);
    } else {
      setShowChatBox(false);
    }
  };

  const handleClearAll = () => {
    setChatHistory([]);
  };

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      {children}
      <div
        style={{
          position: "absolute",
          top: 10,
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 2000,
        }}
      >
        <div className="bg-white p-1 flex justify-center items-center rounded-full shadow-md border border-transparent bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-padding">
          <div className="bg-white rounded-full p-1 flex justify-center items-center">
            <div className="flex space-x-2">
              <button
                onClick={() => handleToolSelect("cursor")}
                className={`p-2 ${selectedTool === "cursor" ? "bg-blue-500 text-white" : "bg-white"} rounded-full`}
              >
                <MousePointer size={16} />
              </button>
              <button
                onClick={() => handleToolSelect("chat")}
                className={`p-2 ${selectedTool === "chat" ? "bg-blue-500 text-white" : "bg-white"} rounded-full`}
              >
                <MessageCircle size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 999,
          pointerEvents: selectedTool === "cursor" ? "none" : "auto",
        }}
      >
        {/* This div remains to handle the cursor tool */}
      </div>
      {showChatBox && (
        <Rnd
          default={{
            x: chatPosition.x,
            y: chatPosition.y,
            width: chatSize.width,
            height: chatSize.height,
          }}
          minWidth={300}
          minHeight={200}
          bounds="window"
          onDragStop={(e, d) => {
            setChatPosition({ x: d.x, y: d.y });
          }}
          onResizeStop={(e, direction, ref, delta, position) => {
            setChatSize({
              width: ref.offsetWidth,
              height: ref.offsetHeight,
            });
            setChatPosition(position);
          }}
          style={{
            zIndex: 1001,
            position: "fixed",
          }}
        >
          <Card className="w-full h-full shadow-lg overflow-hidden bg-background text-foreground">
            <CardHeader className="cursor-move bg-muted p-3 flex justify-between items-center">
              <div className="flex items-center justify-between w-full">
                <h3 className="text-lg font-semibold flex items-center">
                  <MessageCircle className="w-5 h-5 mr-2" /> AI Assistant
                </h3>
                <button
                  onClick={handleClearAll}
                  className="p-1 rounded-full hover:bg-gray-200 transition-colors duration-200"
                  title="Clear history"
                >
                  <RefreshCw size={16} />
                </button>
              </div>
            </CardHeader>
            <CardContent
              className="flex flex-col p-4"
              style={{ height: "calc(100% - 70px)" }}
            >
              <div className="flex-grow mb-4 overflow-y-auto space-y-3">
                <div className="space-y-3">
                  {chatHistory.map((message, index) => (
                    <div
                      key={index}
                      className={`p-2 rounded-md ${message.role === "user" ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground"}`}
                    >
                      <span className="font-medium">
                        {message.role === "user" ? (
                          <>
                            <User className="inline-block w-4 h-4 mr-1" />
                          </>
                        ) : (
                          <>
                            <Bot className="inline-block w-4 h-4 mr-1" />
                          </>
                        )}
                        :
                      </span>{" "}
                      {message.content}
                    </div>
                  ))}
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Input
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Type your message..."
                  onKeyPress={(e) =>
                    e.key === "Enter" && !e.shiftKey && handleChatSubmit()
                  }
                  className="flex-grow bg-background text-foreground"
                />
                <Button
                  onClick={handleChatSubmit}
                  className="bg-primary text-primary-foreground hover:bg-primary/90"
                >
                  Send
                </Button>
              </div>
            </CardContent>
          </Card>
        </Rnd>
      )}
    </div>
  );
};

export default SpotlightOverlay;
