// Centralized showcase component styles for consistency and maintainability

export const showcaseStyles = {
  // Card styles - light and clean for screenshots
  card: {
    base: "bg-white border-gray-200 shadow-sm hover:shadow-md transition-shadow",
    selected: "bg-blue-50 border-blue-200",
    hover: "hover:bg-gray-50",
  },
  
  // Text styles with high contrast
  text: {
    title: "text-gray-900 font-semibold",
    subtitle: "text-gray-700",
    muted: "text-gray-600",
    label: "text-gray-800 font-medium",
    value: "text-gray-900",
  },
  
  // Badge color schemes
  badges: {
    // Tone badges
    tone: {
      Defensive: "bg-red-50 text-red-700 border-red-200",
      Aggressive: "bg-orange-50 text-orange-700 border-orange-200",
      Neutral: "bg-gray-50 text-gray-700 border-gray-200",
      Empathetic: "bg-blue-50 text-blue-700 border-blue-200",
      Positive: "bg-green-50 text-green-700 border-green-200",
    },
    
    // Role badges
    role: {
      Member: "bg-purple-50 text-purple-700 border-purple-200",
      Chair: "bg-indigo-50 text-indigo-700 border-indigo-200",
      "Witness-Industry": "bg-amber-50 text-amber-700 border-amber-200",
      "Witness-Physician": "bg-teal-50 text-teal-700 border-teal-200",
      "Witness-Academic": "bg-pink-50 text-pink-700 border-pink-200",
      "Witness-Patient": "bg-cyan-50 text-cyan-700 border-cyan-200",
      "Witness-Government": "bg-slate-50 text-slate-700 border-slate-200",
      "Witness-Other": "bg-gray-50 text-gray-700 border-gray-200",
    },
    
    // Generic badges
    default: "bg-gray-50 text-gray-700 border-gray-200",
    primary: "bg-blue-50 text-blue-700 border-blue-200",
    success: "bg-green-50 text-green-700 border-green-200",
    warning: "bg-yellow-50 text-yellow-700 border-yellow-200",
    danger: "bg-red-50 text-red-700 border-red-200",
  },
  
  // Stats card styles
  stats: {
    card: "bg-white border-gray-200 shadow-sm",
    title: "text-gray-600 text-sm font-medium",
    value: "text-gray-900 text-2xl font-bold",
    subtitle: "text-gray-500 text-xs",
  },
  
  // Table styles
  table: {
    header: "bg-gray-50 text-gray-900 font-medium",
    row: "border-b border-gray-100 hover:bg-gray-50",
    cell: "text-gray-800",
    selected: "bg-blue-50",
  },
  
  // Progress bar styles
  progress: {
    background: "bg-gray-100",
    fill: "bg-blue-500",
  },
  
  // Alert/Info styles
  alert: {
    info: "bg-blue-50 border-blue-200 text-blue-800",
    warning: "bg-yellow-50 border-yellow-200 text-yellow-800",
    success: "bg-green-50 border-green-200 text-green-800",
    error: "bg-red-50 border-red-200 text-red-800",
  },
  
  // Button styles
  button: {
    primary: "bg-blue-600 text-white hover:bg-blue-700",
    secondary: "bg-gray-100 text-gray-700 hover:bg-gray-200",
    ghost: "text-gray-700 hover:bg-gray-100",
  },
  
  // Container styles
  container: {
    main: "bg-gray-50 min-h-screen",
    content: "bg-white rounded-lg shadow-sm p-6",
  },
};

// Helper function to get badge style
export function getBadgeStyle(type: 'tone' | 'role' | 'default', value?: string): string {
  if (type === 'default') return showcaseStyles.badges.default;
  const styles = showcaseStyles.badges[type];
  return (value && styles[value as keyof typeof styles]) || showcaseStyles.badges.default;
}

// Helper function for consistent card styling
export function getCardClassName(isSelected?: boolean, isHoverable?: boolean): string {
  const classes = [showcaseStyles.card.base];
  if (isSelected) classes.push(showcaseStyles.card.selected);
  if (isHoverable) classes.push(showcaseStyles.card.hover);
  return classes.join(' ');
}