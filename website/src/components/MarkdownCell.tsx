import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import vegaEmbed, { Mode } from "vega-embed";
import { Button } from "@/components/ui/button";

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

/**
 * Renders Vega or Vega-Lite graphs
 * It can toggle between showing visualization or JSON code.
 */
const VegaVisualizer = ({ spec, mode }: VegaVisualizerProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showCode, setShowCode] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);

  useEffect(() => {
    if (containerRef.current && !showCode) {
      try {
        const parsedSpec = JSON.parse(spec);
        setParseError(null);
        vegaEmbed(containerRef.current, parsedSpec, {
          mode,
          actions: true,
          ast: true,
        }).catch(error => {
          setParseError(error.message);
        });
      } catch (error) {
        setParseError(`${error}`);
      }
    }
  }, [spec, mode, showCode]);

  return (
    <div className="flex flex-col my-2">
      <div className="flex justify-end mb-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowCode(!showCode)}
        >
          {showCode ? "Show Graph" : "Show Code"}
        </Button>
      </div>

      {showCode ? (
        <pre className="bg-slate-100 p-2 rounded">
          <code className={`language-json`}>
            {spec}
          </code>
        </pre>
      ) : (
        <>
          {parseError ? (
            <div>
              <div className="text-red-500 text-sm mt-2 p-2 bg-red-50 rounded " style={{
                width: '100%',
                whiteSpace: "pre-wrap",
                wordWrap: "break-word",
              }}>
                JSON.parse(): {parseError}
              </div>
              <code>
                {spec}
              </code>
            </div>
          ) : (
            <div
              ref={containerRef}
              className="rounded p-2"
              style={{ minHeight: "180px" }}
            />
          )}
        </>
      )}
    </div>
  );
};
