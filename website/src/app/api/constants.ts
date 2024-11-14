export const API_ROUTES = {
  OPTIMIZE: {
    SUBMIT: "/api/shouldOptimize",
    STATUS: (taskId: string) => `/api/shouldOptimize?taskId=${taskId}`,
    CANCEL: (taskId: string) =>
      `/api/shouldOptimize?taskId=${taskId}&cancel=true`,
  },
} as const;
