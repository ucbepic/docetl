import React, {
  createContext,
  useContext,
  useRef,
  useState,
  useCallback,
} from "react";

// Define message types
interface WebSocketMessage {
  type: "output" | "result" | "error" | "optimizer_progress";
  data?: any;
  message?: string;
  status?: string;
  progress?: number;
  should_optimize?: boolean;
  rationale?: string;
  validator_prompt?: string;
}

interface WebSocketContextType {
  lastMessage: WebSocketMessage | null;
  readyState: number;
  connect: () => Promise<void>;
  disconnect: () => void;
  sendMessage: (message: string | Record<string, unknown>) => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

export const WebSocketProvider: React.FC<{
  children: React.ReactNode;
  namespace: string;
}> = ({ children, namespace }) => {
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [readyState, setReadyState] = useState<number>(WebSocket.CLOSED);
  const ws = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    return new Promise<void>((resolve, reject) => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      if (!namespace) {
        reject(new Error("Namespace is required for WebSocket connection"));
        return;
      }

      ws.current = new WebSocket(
        `${
          process.env.NEXT_PUBLIC_BACKEND_HTTPS === "true" ? "wss://" : "ws://"
        }${process.env.NEXT_PUBLIC_BACKEND_HOST}:${
          process.env.NEXT_PUBLIC_BACKEND_PORT
        }/ws/run_pipeline/${namespace}`
      );

      ws.current.onopen = () => {
        setReadyState(WebSocket.OPEN);
        resolve();
      };

      ws.current.onclose = () => {
        setReadyState(WebSocket.CLOSED);
        console.log("WebSocket disconnected");
      };

      ws.current.onerror = (error: Event) => {
        console.error("WebSocket error:", error);
        setLastMessage({
          type: "error",
          data: "A WebSocket error occurred. Make sure websockets is installed and enabled on your server.",
        });
        reject(new Error("WebSocket connection failed"));
      };

      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          if (message.type === "error" && !message.data) {
            setLastMessage({
              type: "error",
              data: message.message || "An unknown error occurred",
            });
          } else {
            setLastMessage(message);
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
          setLastMessage({
            type: "error",
            data: "Failed to parse WebSocket message",
          });
        }
      };
    });
  }, [namespace]);

  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close();
    }
  }, []);

  const sendMessage = useCallback(
    (message: string | Record<string, unknown>) => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify(message));
      } else {
        console.error("WebSocket is not connected");
      }
    },
    []
  );

  const contextValue: WebSocketContextType = {
    lastMessage,
    readyState,
    connect,
    disconnect,
    sendMessage,
  };

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error("useWebSocket must be used within a WebSocketProvider");
  }
  return context;
};
