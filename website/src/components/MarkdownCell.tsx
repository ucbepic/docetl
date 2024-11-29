import React from "react";
import ReactMarkdown from "react-markdown";

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
          const match = /language-(\w+)/.exec(className || "");
          return !inline && match ? (
            <pre className="bg-slate-100 p-2 rounded">
              <code className={className} {...props}>
                {children}
              </code>
            </pre>
          ) : (
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
