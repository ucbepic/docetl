import { File } from '@/app/types';
import React from 'react';


const DatasetView: React.FC<{ file: File | null }> = ({ file }) => (
    <div className="h-full p-4 bg-white overflow-x-auto">
      <h2 className="text-lg font-bold mb-2">{file?.name}</h2>
      <pre className="whitespace-pre-wrap">
        {file?.content.split('\n').map((line, index) => (
          <div key={index}>
            <span className="inline-block w-8 text-gray-500 select-none">{index + 1}</span>
            {line}
          </div>
        ))}
      </pre>
    </div>
  );

  export default DatasetView;