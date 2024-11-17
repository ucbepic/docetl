from fastapi import APIRouter, UploadFile, File
from typing import List
import tempfile
import os
import aiohttp
from pathlib import Path

router = APIRouter()

MODAL_ENDPOINT = "https://ucbepic--docling-converter-convert-documents.modal.run"

@router.post("/api/convert-documents")
async def convert_documents(files: List[UploadFile] = File(...)):
    # First try Modal endpoint
    try:
        async with aiohttp.ClientSession() as session:
            # Prepare files for multipart upload
            data = aiohttp.FormData()
            for file in files:
                data.add_field('files',
                             await file.read(),
                             filename=file.filename,
                             content_type=file.content_type)
            
            async with session.post(MODAL_ENDPOINT, data=data, timeout=120) as response:
                if response.status == 200:
                    return await response.json()
            
    except Exception as e:
        print(f"Modal endpoint failed: {str(e)}. Falling back to local processing...")
    
    # If Modal fails, fall back to local processing
    from docling.document_converter import DocumentConverter
    doc_converter = DocumentConverter()
    
    # Create a temporary directory to store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to temporary directory
        file_paths = []
        original_filenames = []  # Keep track of original filenames
        txt_files = []  # Track which files are .txt
        for file in files:
            # Reset file position since we might have read it in the Modal attempt
            await file.seek(0)
            file_path = os.path.join(temp_dir, file.filename)
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            file_paths.append(file_path)
            original_filenames.append(file.filename)
            txt_files.append(file.filename.lower().endswith('.txt'))
        
        # Convert all documents
        results = []
        non_txt_paths = [fp for fp, is_txt in zip(file_paths, txt_files) if not is_txt]
        
        # Get docling iterator for non-txt files if there are any
        docling_iter = iter(doc_converter.convert_all(non_txt_paths)) if non_txt_paths else iter([])
        
        # Process all files
        for filename, file_path, is_txt in zip(original_filenames, file_paths, txt_files):
            if is_txt:
                # For txt files, just read the content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                results.append({
                    "filename": filename,
                    "markdown": content
                })
            else:
                # For non-txt files, get next result from docling iterator
                conv_result = next(docling_iter)
                results.append({
                    "filename": filename,
                    "markdown": conv_result.document.export_to_markdown()
                })
        
        return {"documents": results}