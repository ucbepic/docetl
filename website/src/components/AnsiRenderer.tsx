import React, { useEffect, useRef } from 'react';
import Convert from 'ansi-to-html';

const convert = new Convert({
  fg: '#000',
  bg: '#fff',
  newline: true,
  escapeXML: true,
  stream: false
});

interface AnsiRendererProps {
  text: string;
  readyState: number;
  setTerminalOutput: (text: string) => void;
}

const AnsiRenderer: React.FC<AnsiRendererProps> = ({ text, readyState, setTerminalOutput }) => {
  const html = convert.toHtml(text);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [text]);

  return (
    <div className="flex flex-col w-full h-[500px] bg-black text-white font-mono rounded-lg overflow-hidden">
      <div 
        ref={scrollRef}
        className="flex-grow overflow-auto p-4"
        style={{
          height: '450px', // Fixed height for the terminal output
          maxHeight: '450px',
        }}
      >
        <pre
          className="m-0 whitespace-pre-wrap break-words"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>
      <div className="p-2 border-t border-gray-700">
        <div className="text-xs text-gray-500">
          WebSocket State: {readyState === WebSocket.CONNECTING ? "Connecting" :
                            readyState === WebSocket.OPEN ? "Open" :
                            readyState === WebSocket.CLOSING ? "Closing" :
                            readyState === WebSocket.CLOSED ? "Closed" : "Unknown"}
        </div>
        <button onClick={() => setTerminalOutput('')} className="mt-2 text-xs text-gray-500">Clear Output</button>
      </div>
    </div>
  );
};

export default AnsiRenderer;