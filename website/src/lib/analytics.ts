export const logEvent = (
  action: string,
  category: string,
  label: string,
  value?: number,
): void => {
  window.gtag?.("event", action, {
    event_category: category,
    event_label: label,
    value: value,
  });
};
