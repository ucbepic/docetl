import { useState, useRef, useCallback } from "react";
import { DecomposeResult } from "@/app/types";

interface DecomposeWebSocketMessage {
  type: "output" | "result" | "error";
  data?: any;
  message?: string;
  traceback?: string;
}

interface UseDecomposeWebSocketProps {
  namespace: string;
  onOutput?: (output: string) => void;
  onComplete?: (result: DecomposeResult) => void;
  onError?: (error: string) => void;
}

export function useDecomposeWebSocket({
  namespace,
  onOutput,
  onComplete,
  onError,
}: UseDecomposeWebSocketProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  const connect = useCallback((): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      if (!namespace) {
        reject(new Error("Namespace is required for WebSocket connection"));
        return;
      }

      const wsUrl = `${
        process.env.NEXT_PUBLIC_BACKEND_HTTPS === "true" ? "wss://" : "ws://"
      }${process.env.NEXT_PUBLIC_BACKEND_HOST}:${
        process.env.NEXT_PUBLIC_BACKEND_PORT
      }/ws/decompose/${namespace}`;

      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        setIsConnected(true);
        resolve();
      };

      ws.current.onclose = () => {
        setIsConnected(false);
        setIsRunning(false);
      };

      ws.current.onerror = (error: Event) => {
        console.error("Decompose WebSocket error:", error);
        onError?.("WebSocket connection failed");
        reject(new Error("WebSocket connection failed"));
      };

      ws.current.onmessage = (event) => {
        try {
          const message: DecomposeWebSocketMessage = JSON.parse(event.data);

          if (message.type === "output") {
            onOutput?.(message.data);
          } else if (message.type === "result") {
            setIsRunning(false);
            // Convert the result data to DecomposeResult format
            const result: DecomposeResult = {
              task_id: "", // Not used in WebSocket mode
              status: "completed",
              decomposed_operations: message.data.decomposed_operations,
              winning_directive: message.data.winning_directive,
              candidates_evaluated: message.data.candidates_evaluated,
              original_outputs: message.data.original_outputs,
              decomposed_outputs: message.data.decomposed_outputs,
              comparison_rationale: message.data.comparison_rationale,
              cost: message.data.cost,
              created_at: new Date().toISOString(),
              completed_at: new Date().toISOString(),
            };
            onComplete?.(result);
            // Close the connection after receiving result
            ws.current?.close();
          } else if (message.type === "error") {
            setIsRunning(false);
            const errorMsg = message.data || message.message || "Unknown error";
            onError?.(errorMsg);
            ws.current?.close();
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
          onError?.("Failed to parse WebSocket message");
        }
      };
    });
  }, [namespace, onOutput, onComplete, onError]);

  const startDecomposition = useCallback(
    async (yamlConfig: string, stepName: string, opName: string) => {
      try {
        await connect();
        setIsRunning(true);

        if (ws.current?.readyState === WebSocket.OPEN) {
          ws.current.send(
            JSON.stringify({
              yaml_config: yamlConfig,
              step_name: stepName,
              op_name: opName,
            })
          );
        }
      } catch (error) {
        setIsRunning(false);
        throw error;
      }
    },
    [connect]
  );

  const cancel = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify("kill"));
    }
  }, []);

  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close();
    }
  }, []);

  return {
    isConnected,
    isRunning,
    startDecomposition,
    cancel,
    disconnect,
  };
}
