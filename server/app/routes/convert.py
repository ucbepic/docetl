from fastapi import APIRouter, UploadFile, File
from typing import List
import tempfile
import os
import aiohttp
from pathlib import Path

router = APIRouter()

MODAL_ENDPOINT = "https://ucbepic--docling-converter-doclingconverter-convert-documents.modal.run"

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
        for file in files:
            # Reset file position since we might have read it in the Modal attempt
            await file.seek(0)
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            file_paths.append(file_path)
            original_filenames.append(file.filename)
        
        # Convert all documents
        results = []
        for filename, conv_result in zip(original_filenames, doc_converter.convert_all(file_paths)):
            results.append({
                "filename": filename,
                "markdown": conv_result.document.export_to_markdown(),
            })
        
        return {"documents": results}