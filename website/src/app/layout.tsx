import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { GoogleAnalytics } from "@next/third-parties/google";
import Providers from "./providers";
import React from "react";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: {
    default: "DocETL – AI-Powered Document ETL Platform",
    template: "%s | DocETL",
  },
  description:
    "Open-source toolkit, built by the EPIC Data Lab at UC Berkeley, for creating LLM-powered pipelines that extract, transform, and link knowledge from unstructured documents.",
  keywords: [
    "LLM data extraction",
    "document ETL",
    "AI document processing",
    "unstructured data pipeline",
    "open source AI tooling",
  ],
  openGraph: {
    title: "DocETL – AI-Powered Document ETL Platform",
    description:
      "Build complex document processing pipelines with large language models. Extract structured data, link entities, rank information and more using a single YAML file. Built by the EPIC Data Lab at UC Berkeley.",
    url: "https://www.docetl.org",
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
    other: {
      rel: "icon",
      url: "/docetl-favicon-color.png",
    },
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <Providers>{children}</Providers>
        <GoogleAnalytics gaId="G-M9CR0T6CJ0" />
        <Toaster />
      </body>
    </html>
  );
}
