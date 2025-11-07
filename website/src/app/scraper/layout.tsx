import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "DocScraper – AI Web Scraper for DocETL",
  description:
    "Agentic web scraping tool that intelligently collects raw text from articles and blog posts. Build datasets for DocETL pipelines with AI-powered parallel scraping and smart link discovery.",
  keywords: [
    "web scraping",
    "AI scraper",
    "dataset generation",
    "DocETL",
    "agentic scraping",
    "parallel web scraping",
    "document collection",
    "blog scraping",
  ],
  openGraph: {
    title: "DocScraper – AI Web Scraper for DocETL",
    description:
      "Build datasets for DocETL pipelines with AI-powered web scraping. Intelligently navigates websites, extracts raw text, and discovers content across the web.",
    url: "https://www.docetl.org/scraper",
    type: "website",
    images: [
      {
        url: "/docetl-favicon-color.png",
      },
    ],
  },
  icons: {
    icon: "/docetl-favicon-color.png",
    shortcut: "/docetl-favicon-color.png",
    apple: "/docetl-favicon-color.png",
  },
};

export default function ScraperLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}

