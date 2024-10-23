export const saveToFile = async (data: any, defaultFilename: string) => {
  try {
    // Use the showSaveFilePicker API to let user choose save location and filename
    const handle = await window.showSaveFilePicker({
      suggestedName: defaultFilename,
      types: [{
        description: 'DocETL Pipeline File',
        accept: {
          'application/json': ['.dtl']
        }
      }]
    });

    // Create a writable stream and write the data
    const writable = await handle.createWritable();
    await writable.write(JSON.stringify(data, null, 2));
    await writable.close();
  } catch (err) {
    // User cancelled or other error
    console.error('Error saving file:', err);
    throw err;
  }
};

export const loadFromFile = async (): Promise<any> => {
  try {
    // Use the showOpenFilePicker API for a better file selection experience
    const [handle] = await window.showOpenFilePicker({
      types: [{
        description: 'DocETL Pipeline File',
        accept: {
          'application/json': ['.dtl']
        }
      }],
      multiple: false
    });

    const file = await handle.getFile();
    const text = await file.text();
    return JSON.parse(text);
  } catch (err) {
    // User cancelled or other error
    console.error('Error loading file:', err);
    throw err;
  }
};

// Fallback for browsers that don't support the File System Access API
export const saveToFileClassic = async (data: any, defaultFilename: string) => {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = defaultFilename;
  
  // Append to document, click, and cleanup
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export const loadFromFileClassic = async (): Promise<any> => {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.dtl';
    
    input.onchange = async (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) {
        reject(new Error('No file selected'));
        return;
      }

      try {
        const text = await file.text();
        const data = JSON.parse(text);
        resolve(data);
      } catch (error) {
        reject(error);
      }
    };

    input.click();
  });
};
