export const getBackendUrl = () => {
  const protocol = process.env.NEXT_PUBLIC_BACKEND_HTTPS ? "https" : "http";
  const host = process.env.NEXT_PUBLIC_BACKEND_HOST;
  const port = process.env.NEXT_PUBLIC_BACKEND_PORT;
  return `${protocol}://${host}:${port}`;
};
