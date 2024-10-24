import React, { createContext, useContext, useRef, useState, useCallback } from 'react';

interface WebSocketContextType {
  lastMessage: any;
  readyState: number;
  connect: () => Promise<void>;
  disconnect: () => void;
  sendMessage: (message: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [readyState, setReadyState] = useState<number>(WebSocket.CLOSED);
  const ws = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    return new Promise<void>((resolve, reject) => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      ws.current = new WebSocket('ws://localhost:8000/ws/run_pipeline');

      ws.current.onopen = () => {
        setReadyState(WebSocket.OPEN);
        resolve();
      };

      ws.current.onclose = () => {
        setReadyState(WebSocket.CLOSED);
        console.log('WebSocket disconnected');
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };

      ws.current.onmessage = (event) => {
        const message = JSON.parse(event.data);
        setLastMessage(message);
      };
    });
  }, []);

  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close();
    }
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      console.error('WebSocket is not connected');
    }
  }, []);

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
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};