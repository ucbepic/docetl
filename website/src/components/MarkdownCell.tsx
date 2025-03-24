import React, { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import vegaEmbed, { Mode } from "vega-embed";

interface MarkdownCellProps {
  content: string;
}

export const MarkdownCell = React.memo(({ content }: MarkdownCellProps) => {
  return (
    <ReactMarkdown
      components={{
        h1: ({ children }) => (
          <div style={{ fontWeight: "bold", fontSize: "1.5rem" }}>
            {children}
          </div>
        ),
        h2: ({ children }) => (
          <div style={{ fontWeight: "bold", fontSize: "1.25rem" }}>
            {children}
          </div>
        ),
        h3: ({ children }) => (
          <div style={{ fontWeight: "bold", fontSize: "1.1rem" }}>
            {children}
          </div>
        ),
        h4: ({ children }) => (
          <div style={{ fontWeight: "bold" }}>{children}</div>
        ),
        ul: ({ children }) => (
          <ul
            style={{
              listStyleType: "•",
              paddingLeft: "1rem",
              margin: "0.25rem 0",
            }}
          >
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol
            style={{
              listStyleType: "decimal",
              paddingLeft: "1rem",
              margin: "0.25rem 0",
            }}
          >
            {children}
          </ol>
        ),
        li: ({ children }) => (
          <li style={{ marginBottom: "0.125rem" }}>{children}</li>
        ),
        code: ({
          className,
          children,
          inline,
          ...props
        }: {
          className?: string;
          children: React.ReactNode;
          inline?: boolean;
        }) => {

          if (!inline) {
            const match = /language-(\w+)/.exec(className || "");
            if (match) {
              const codeLanguage = match[1];
              if (codeLanguage === "vega" || codeLanguage === "vega-lite") {
                return <VegaVisualizer spec={String(children)} mode={codeLanguage} />;
              }

              return (
                <pre className="bg-slate-100 p-2 rounded">
                  <code className={className} {...props}>
                    {children}
                  </code>
                </pre>
              );
            }
          }

          return (
            <code className="bg-slate-100 px-1 rounded" {...props}>
              {children}
            </code>
          );

        },
        pre: ({ children }) => (
          <pre className="bg-slate-100 p-2 rounded">{children}</pre>
        ),
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-slate-300 pl-4 my-2 italic">
            {children}
          </blockquote>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
});



MarkdownCell.displayName = "MarkdownCell";

interface VegaVisualizerProps {
  /** A vega or vega-lite definition as stringified JSON */
  spec: string;
  /** The mode to use for the Vega visualization */
  mode: Mode;
}

const VegaVisualizer = ({ spec, mode }: VegaVisualizerProps) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      try {
        const parsedSpec = JSON.parse(spec);
        vegaEmbed(containerRef.current, parsedSpec, {
          mode: mode as Mode,
          actions: true,
          ast: true,
        }).catch(console.error);
      } catch (error) {
        console.error("Failed to parse Vega spec:", error);
      }
    }
  }, [spec, mode]);

  return (
    <div className="my-2">
      <div
        ref={containerRef}
        className="border border-slate-200 rounded p-2"
        style={{ minHeight: "200px" }}
      ></div>
    </div>
  );
};
