from fastapi import APIRouter, UploadFile, File, Header
from typing import List, Optional
import tempfile
import os
import aiohttp
from pathlib import Path
from urllib.parse import urljoin
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult, DocumentContentFormat
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import base64
import pypdfium2 as pdfium

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

# Maximum page limit for Azure Document Intelligence
MAX_AZURE_PAGE_LIMIT = 200

def get_pdf_page_count(file_path: str) -> int:
    """
    Get the number of pages in a PDF document.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Number of pages in the PDF
    """
    try:
        pdf = pdfium.PdfDocument(file_path)
        return len(pdf)
    except Exception as e:
        print(f"Error counting PDF pages: {str(e)}")
        # Return a large number to trigger the page limit check
        return MAX_AZURE_PAGE_LIMIT + 1

def process_document_with_azure_read(file_path: str, endpoint: str, key: str) -> str:
    """
    Process a single document with Azure Document Intelligence using the prebuilt-read model
    
    Args:
        file_path: Path to the document file
        endpoint: Azure Document Intelligence endpoint
        key: Azure API key
        
    Returns:
        Extracted text content from the document as a string
    """
    try:
        # Initialize the Document Intelligence client
        client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        
        # Read and process the file
        with open(file_path, "rb") as file:
            file_bytes = file.read()
            poller = client.begin_analyze_document(
                "prebuilt-read",
                AnalyzeDocumentRequest(bytes_source=file_bytes))
            result = poller.result()
        
        # Extract text content from all pages
        extracted_text = ""
        for idx, page in enumerate(result.pages):
            # Add page number as a header
            page_number = page.pageNumber if hasattr(page, 'pageNumber') else idx + 1
            extracted_text += f"Page {page_number}\n\n"
            for line in page.lines:
                extracted_text += line.content + "\n"
            extracted_text += "\n"  # Extra line break between pages
                
        return extracted_text.strip()
    
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return f"Error processing document: {str(e)}"

def process_document_with_azure_layout(file_path: str, endpoint: str, key: str) -> str:
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
async def convert_documents(
    files: List[UploadFile] = File(...), 
    use_docetl_server: str = "false",
    custom_docling_url: Optional[str] = Header(None)
):
    use_docetl_server = use_docetl_server.lower() == "true" # TODO: make this a boolean

    
    # If custom Docling URL is provided, forward the request there
    if custom_docling_url:
        try:
            async with aiohttp.ClientSession() as session:
                results = []
                for file in files:
                    # Read file content and encode as base64
                    content = await file.read()
                    base64_content = base64.b64encode(content).decode('utf-8')
                    
                    # Prepare request payload according to Docling server spec
                    payload = {
                        "file_source": {
                            "base64_string": base64_content,
                            "filename": file.filename
                        },
                        "options": {
                            "output_docling_document": False,
                            "output_markdown": True,
                            "output_html": False,
                            "do_ocr": True,
                            "do_table_structure": True,
                            "include_images": False
                        }
                    }
                    
                    async with session.post(
                        urljoin(custom_docling_url, 'convert'),
                        json=payload,
                        timeout=120
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if result["status"] in ("success", '4'):
                                results.append({
                                    "filename": file.filename,
                                    "markdown": result["document"]["markdown"]
                                })
                            else:
                                return {"error": f"Docling server failed to convert {file.filename}: {result.get('errors', [])}"}
                        else:
                            error_msg = await response.text()
                            return {"error": f"Custom Docling server returned error for {file.filename}: {error_msg}"}
                
                return {"documents": results}
            
        except Exception as e:
            print(f"Custom Docling server failed: {str(e)}. Falling back to local processing...")
    
    # Only try Modal endpoint if use_docetl_server is true and there are no txt files
    all_txt_files = all(file.filename.lower().endswith('.txt') or file.filename.lower().endswith('.md') for file in files)
    if use_docetl_server and not all_txt_files:
        try:
            async with aiohttp.ClientSession() as session:
                # Prepare files for multipart upload
                data = aiohttp.FormData()
                for file in files:
                    # Reset file position since we might have read it in the custom Docling attempt
                    await file.seek(0)
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
                content = try_read_file_with_encodings(file_path)
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
    azure_key: Optional[str] = Header(None),
    is_read: str = Header("false")
):
    if not azure_endpoint or not azure_key:
        # Use os.getenv keys
        azure_endpoint = os.getenv("AZURE_DOCUMENTINTELLIGENCE_ENDPOINT")
        azure_key = os.getenv("AZURE_DOCUMENTINTELLIGENCE_API_KEY")
        
        if not azure_endpoint or not azure_key:
            return {"error": "Azure credentials are required"}
        
        # If there are > 50 files, return an error
        if len(files) > 50:
            return {"error": "We will only process up to 50 files; use your own Azure keys to process more."}
    
    is_read = is_read.lower() == "true"

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
            
            # Check page count before adding to processing queue
            if file_path.lower().endswith('.pdf'):
                page_count = get_pdf_page_count(file_path)
                if page_count > MAX_AZURE_PAGE_LIMIT:
                    # Return an error here
                    return {"error": f"Document {file.filename} exceeds maximum page limit of {MAX_AZURE_PAGE_LIMIT} pages. This document has {page_count} pages."}
            
            file_paths.append(file_path)
            original_filenames.append(file.filename)

        # Process documents concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(64, len(file_paths))) as executor:
            futures = []
            for file_path in file_paths:
                future = executor.submit(
                    process_document_with_azure_read if is_read else process_document_with_azure_layout,
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

        # Return results, including information about skipped files
        response = {"documents": formatted_results}

        return response
    
def get_supported_encodings():
    """Get list of supported encodings from environment or use default."""
    encodings_str = os.getenv("TEXT_FILE_ENCODINGS", "utf-8")
    return [enc.strip() for enc in encodings_str.split(",")]

def try_read_file_with_encodings(file_path: str) -> str:
    """Try to read a file with configured encodings."""
    encodings = get_supported_encodings()

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            print(f"Failed to decode {file_path} with encoding {encoding}")
            continue

    # If all encodings fail, try with the most permissive one and replace errors
    with open(file_path, 'r', encoding='latin1', errors='replace') as f:
        return f.read()

