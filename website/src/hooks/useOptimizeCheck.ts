import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { OptimizeResult, OptimizeRequest } from "@/app/types";
import { API_ROUTES } from "@/app/api/constants";

interface UseOptimizeCheckProps {
  onComplete?: (result: OptimizeResult) => void;
  onError?: (error: string) => void;
  pollInterval?: number;
}

export function useOptimizeCheck({
  onComplete,
  onError,
  pollInterval = 1000,
}: UseOptimizeCheckProps = {}) {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const submitTask = async (request: OptimizeRequest) => {
    try {
      setIsLoading(true);
      setError(null);
      setResult(null);

      const response = await axios.post<{ task_id: string }>(
        API_ROUTES.OPTIMIZE.SUBMIT,
        request
      );

      setTaskId(response.data.task_id);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to submit task";
      setError(errorMessage);
      onError?.(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const cancelTask = async () => {
    if (!taskId) return;

    try {
      await axios.post(API_ROUTES.OPTIMIZE.CANCEL(taskId));
      setTaskId(null);
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
        const response = await axios.get<OptimizeResult>(
          API_ROUTES.OPTIMIZE.STATUS(taskId)
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
