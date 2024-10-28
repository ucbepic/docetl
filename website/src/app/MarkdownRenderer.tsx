import ReactMarkdown from "react-markdown";

const MarkdownRenderer = ({ content }: { content: any }) => {
  return (
    <div className="prose max-w-none">
      <ReactMarkdown
        components={{
          h1: (props) => <h1 className="text-3xl font-bold mb-4" {...props} />,
          h2: (props) => <h2 className="text-2xl font-bold mb-3" {...props} />,
          h3: (props) => <h3 className="text-xl font-bold mb-2" {...props} />,
          p: (props) => <p className="mb-4" {...props} />,
          ul: (props) => (
            <ul className="list-disc list-inside mb-4" {...props} />
          ),
          ol: (props) => (
            <ol className="list-decimal list-inside mb-4" {...props} />
          ),
          li: (props) => <li className="mb-1" {...props} />,
          a: (props) => (
            <a className="text-blue-500 hover:underline" {...props} />
          ),
          blockquote: (props) => (
            <blockquote
              className="border-l-4 border-gray-300 pl-4 italic mb-4"
              {...props}
            />
          ),
          code: ({ className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || "");
            const language = match ? match[1] : "";
            const isInlineCode =
              !language &&
              typeof children === "string" &&
              !children.includes("\n");

            return isInlineCode ? (
              <code className="bg-gray-100 rounded px-1" {...props}>
                {children}
              </code>
            ) : (
              <pre className="bg-gray-100 rounded p-4 mb-4 overflow-x-auto">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
