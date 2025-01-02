from fastapi import APIRouter, UploadFile, File, Header
from typing import List, Optional
import tempfile
import os
import aiohttp
from pathlib import Path
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult, DocumentContentFormat
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# Load environment variables
load_dotenv()

# Add Azure credentials
AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

router = APIRouter()

MODAL_ENDPOINT = "https://ucbepic--docling-converter-convert-documents.modal.run"
# MODAL_ENDPOINT = "https://ucbepic--docling-converter-convert-documents-dev.modal.run"

def process_document_with_azure(file_path: str, endpoint: str, key: str) -> str:
    """Process a single document with Azure Document Intelligence"""
    try:
        document_analysis_client = DocumentIntelligenceClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )

        with open(file_path, "rb") as f:
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-layout", AnalyzeDocumentRequest(bytes_source=f.read()), output_content_format=DocumentContentFormat.MARKDOWN,
            )
        result = poller.result()

        return result.content
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return f"Error processing document: {str(e)}"

@router.post("/api/convert-documents")
async def convert_documents(files: List[UploadFile] = File(...), use_docetl_server: str = "false"):
    use_docetl_server = use_docetl_server.lower() == "true" # TODO: make this a boolean
    # Only try Modal endpoint if use_docetl_server is true and there are no txt files
    all_txt_files = all(file.filename.lower().endswith('.txt') or file.filename.lower().endswith('.md') for file in files)
    if use_docetl_server and not all_txt_files:
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
    
    # Process locally if Modal wasn't used or failed

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
            )
        }
    )
    
    # Create a temporary directory to store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to temporary directory
        file_paths = []
        original_filenames = []  # Keep track of original filenames
        txt_files = []  # Track which files are .txt or markdown
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
            txt_files.append(file.filename.lower().endswith('.txt') or file.filename.lower().endswith('.md'))
        
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
    
@router.post("/api/azure-convert-documents")
async def azure_convert_documents(
    files: List[UploadFile] = File(...),
    azure_endpoint: Optional[str] = Header(None),
    azure_key: Optional[str] = Header(None)
):
    if not azure_endpoint or not azure_key:
        return {"error": "Azure credentials are required"}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files and prepare for processing
        file_paths = []
        original_filenames = []
        
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            file_paths.append(file_path)
            original_filenames.append(file.filename)

        # Process documents concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = []
            for file_path in file_paths:
                future = executor.submit(
                    process_document_with_azure,
                    file_path,
                    azure_endpoint,
                    azure_key
                )
                futures.append(future)

            # Collect results as they complete
            results = []
            for future in futures:
                results.append(future.result())

        # Format results to match the existing endpoint's schema
        formatted_results = [
            {
                "filename": filename,
                "markdown": content
            }
            for filename, content in zip(original_filenames, results)
        ]

        return {"documents": formatted_results}
    

