import React from "react";
import VisualizationBuilder from "@/components/VisualizationBuilder";

export const dynamic = "force-static";

export default function VisualizationPage(): JSX.Element {
  return (
    <main className="mx-auto max-w-7xl px-4 py-8">
      <VisualizationBuilder />
    </main>
  );
}
