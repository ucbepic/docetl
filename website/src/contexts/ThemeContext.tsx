"use client";

import React, { createContext, useContext, useEffect, useState } from "react";

export type Theme =
  | "default"
  | "forest"
  | "majestic"
  | "sunset"
  | "ruby"
  | "monochrome";

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const themes = {
  default: {
    background: "211 40% 99%",
    foreground: "211 5% 0%",
    card: "211 25% 97%",
    "card-foreground": "211 5% 10%",
    popover: "211 40% 99%",
    "popover-foreground": "211 100% 0%",
    primary: "211 100% 50%",
    "primary-foreground": "0 0% 100%",
    secondary: "211 30% 70%",
    "secondary-foreground": "0 0% 0%",
    muted: "173 30% 92%",
    "muted-foreground": "211 5% 35%",
    accent: "173 30% 90%",
    "accent-foreground": "211 5% 10%",
    destructive: "0 100% 30%",
    "destructive-foreground": "211 5% 90%",
    border: "211 30% 50%",
    input: "211 30% 18%",
    ring: "211 100% 50%",
    chart1: "12 76% 61%",
    chart2: "173 58% 39%",
    chart3: "197 37% 24%",
    chart4: "43 74% 66%",
    chart5: "27 87% 67%",
  },
  forest: {
    background: "150 40% 99%",
    foreground: "150 10% 5%",
    card: "150 30% 97%",
    "card-foreground": "150 10% 10%",
    popover: "150 40% 99%",
    "popover-foreground": "150 10% 5%",
    primary: "150 100% 35%",
    "primary-foreground": "0 0% 100%",
    secondary: "150 30% 70%",
    "secondary-foreground": "150 10% 5%",
    muted: "150 20% 95%",
    "muted-foreground": "150 10% 35%",
    accent: "150 20% 90%",
    "accent-foreground": "150 10% 10%",
    destructive: "0 100% 30%",
    "destructive-foreground": "150 10% 90%",
    border: "150 30% 50%",
    input: "150 30% 18%",
    ring: "150 100% 50%",
    chart1: "150 70% 40%",
    chart2: "35 85% 50%",
    chart3: "195 65% 40%",
    chart4: "105 60% 45%",
    chart5: "270 45% 45%",
  },
  majestic: {
    background: "270 30% 99%",
    foreground: "270 10% 5%",
    card: "270 20% 97%",
    "card-foreground": "270 10% 10%",
    popover: "270 30% 99%",
    "popover-foreground": "270 10% 5%",
    primary: "270 100% 50%",
    "primary-foreground": "0 0% 100%",
    secondary: "270 30% 70%",
    "secondary-foreground": "270 10% 5%",
    muted: "270 20% 95%",
    "muted-foreground": "270 10% 35%",
    accent: "270 20% 90%",
    "accent-foreground": "270 10% 10%",
    destructive: "0 100% 30%",
    "destructive-foreground": "270 10% 90%",
    border: "270 30% 50%",
    input: "270 30% 18%",
    ring: "270 100% 50%",
    chart1: "270 70% 60%",
    chart2: "330 65% 55%",
    chart3: "210 60% 50%",
    chart4: "30 70% 55%",
    chart5: "150 55% 45%",
  },
  sunset: {
    background: "30 30% 99%",
    foreground: "30 10% 5%",
    card: "30 20% 97%",
    "card-foreground": "30 10% 10%",
    popover: "30 30% 99%",
    "popover-foreground": "30 10% 5%",
    primary: "30 100% 50%",
    "primary-foreground": "0 0% 100%",
    secondary: "30 30% 70%",
    "secondary-foreground": "30 10% 5%",
    muted: "30 20% 95%",
    "muted-foreground": "30 10% 35%",
    accent: "30 20% 90%",
    "accent-foreground": "30 10% 10%",
    destructive: "0 100% 30%",
    "destructive-foreground": "30 10% 90%",
    border: "30 30% 50%",
    input: "30 30% 18%",
    ring: "30 100% 50%",
    chart1: "30 85% 55%",
    chart2: "45 80% 50%",
    chart3: "15 75% 45%",
    chart4: "60 70% 50%",
    chart5: "0 80% 55%",
  },
  ruby: {
    background: "345 30% 99%",
    foreground: "345 10% 5%",
    card: "345 20% 97%",
    "card-foreground": "345 10% 10%",
    popover: "345 30% 99%",
    "popover-foreground": "345 10% 5%",
    primary: "345 100% 50%",
    "primary-foreground": "0 0% 100%",
    secondary: "345 30% 70%",
    "secondary-foreground": "345 10% 5%",
    muted: "345 20% 95%",
    "muted-foreground": "345 10% 35%",
    accent: "345 20% 90%",
    "accent-foreground": "345 10% 10%",
    destructive: "0 100% 30%",
    "destructive-foreground": "345 10% 90%",
    border: "345 30% 50%",
    input: "345 30% 18%",
    ring: "345 100% 50%",
    chart1: "345 85% 55%",
    chart2: "195 70% 50%",
    chart3: "45 75% 55%",
    chart4: "315 65% 45%",
    chart5: "165 60% 50%",
  },
  monochrome: {
    background: "0 0% 98%",
    foreground: "0 0% 0%",
    card: "0 0% 95%",
    "card-foreground": "0 0% 10%",
    popover: "0 0% 98%",
    "popover-foreground": "0 0% 0%",
    primary: "0 0% 20%",
    "primary-foreground": "0 0% 100%",
    secondary: "0 0% 60%",
    "secondary-foreground": "0 0% 0%",
    muted: "0 0% 90%",
    "muted-foreground": "0 0% 35%",
    accent: "0 0% 85%",
    "accent-foreground": "0 0% 10%",
    destructive: "0 100% 30%",
    "destructive-foreground": "0 0% 90%",
    border: "0 0% 40%",
    input: "0 0% 18%",
    ring: "0 0% 20%",
    chart1: "0 0% 20%",
    chart2: "0 0% 35%",
    chart3: "0 0% 50%",
    chart4: "0 0% 65%",
    chart5: "0 0% 80%",
  },
};

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>("default");

  useEffect(() => {
    const savedTheme = localStorage.getItem("color-theme") as Theme;
    if (savedTheme && themes[savedTheme]) {
      setTheme(savedTheme);
    }
  }, []);

  useEffect(() => {
    const root = window.document.documentElement;
    const themeColors = themes[theme];

    // Apply theme colors
    Object.entries(themeColors).forEach(([key, value]) => {
      root.style.setProperty(`--${key}`, value);
    });

    localStorage.setItem("color-theme", theme);
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
};
