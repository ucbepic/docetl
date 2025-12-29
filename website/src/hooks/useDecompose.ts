import { useState, useEffect } from "react";
import axios from "axios";
import { DecomposeResult, DecomposeRequest } from "@/app/types";
import { API_ROUTES } from "@/app/api/constants";

interface UseDecomposeProps {
  onComplete?: (result: DecomposeResult) => void;
  onError?: (error: string) => void;
  pollInterval?: number;
}

export function useDecompose({
  onComplete,
  onError,
  pollInterval = 2000,
}: UseDecomposeProps = {}) {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<DecomposeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const submitTask = async (request: DecomposeRequest) => {
    try {
      setIsLoading(true);
      setError(null);
      setResult(null);

      const response = await axios.post<{ task_id: string }>(
        API_ROUTES.DECOMPOSE.SUBMIT,
        request
      );

      setTaskId(response.data.task_id);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to submit decompose task";
      setError(errorMessage);
      onError?.(errorMessage);
      setIsLoading(false);
    }
  };

  const cancelTask = async () => {
    if (!taskId) return;

    try {
      await axios.post(API_ROUTES.DECOMPOSE.CANCEL(taskId));
      setTaskId(null);
      setIsLoading(false);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to cancel task";
      setError(errorMessage);
      onError?.(errorMessage);
    }
  };

  useEffect(() => {
    if (!taskId) return;

    const pollTask = async () => {
      try {
        const response = await axios.get<DecomposeResult>(
          API_ROUTES.DECOMPOSE.STATUS(taskId)
        );

        setResult(response.data);

        if (
          ["completed", "failed", "cancelled"].includes(response.data.status)
        ) {
          setTaskId(null);
          setIsLoading(false);

          if (response.data.status === "completed") {
            onComplete?.(response.data);
          } else if (response.data.status === "failed" && response.data.error) {
            setError(response.data.error);
            onError?.(response.data.error);
          }
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to fetch task status";
        setError(errorMessage);
        onError?.(errorMessage);
        setTaskId(null);
        setIsLoading(false);
      }
    };

    const interval = setInterval(pollTask, pollInterval);
    return () => clearInterval(interval);
  }, [taskId, onComplete, onError, pollInterval]);

  return {
    submitTask,
    cancelTask,
    result,
    error,
    isLoading,
    isRunning: !!taskId,
  };
}
