import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { GoogleAnalytics } from "@next/third-parties/google";
import Providers from "./providers";

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
  title: "DocETL",
  description: "Powering complex document processing pipelines",
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
