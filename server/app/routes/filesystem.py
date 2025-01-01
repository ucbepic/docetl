from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import os
import yaml
import shutil
from pathlib import Path
from server.app.models import PipelineConfigRequest

router = APIRouter()


def get_home_dir() -> str:
    """Get the home directory from env var or user home"""
    return os.getenv("DOCETL_HOME_DIR", os.path.expanduser("~"))

def get_namespace_dir(namespace: str) -> Path:
    """Get the namespace directory path"""
    home_dir = get_home_dir()
    return Path(home_dir) / ".docetl" / namespace

@router.post("/check-namespace")
async def check_namespace(namespace: str):
    """Check if namespace exists and create if it doesn't"""
    try:
        namespace_dir = get_namespace_dir(namespace)
        exists = namespace_dir.exists()
        
        if not exists:
            namespace_dir.mkdir(parents=True, exist_ok=True)
            
        return {"exists": exists}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check/create namespace: {str(e)}")

@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...), namespace: str = Form(...)):
    """Upload a single file to the namespace files directory"""
    try:
        upload_dir = get_namespace_dir(namespace) / "files"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
            
        return {"path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@router.post("/save-documents")
async def save_documents(files: List[UploadFile] = File(...), namespace: str = Form(...)):
    """Save multiple documents to the namespace documents directory"""
    try:
        uploads_dir = get_namespace_dir(namespace) / "documents"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for file in files:
            # Create safe filename
            safe_name = "".join(c if c.isalnum() or c in ".-" else "_" for c in file.filename)
            file_path = uploads_dir / safe_name
            
            with file_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
                
            saved_files.append({
                "name": file.filename,
                "path": str(file_path)
            })
            
        return {"files": saved_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save documents: {str(e)}")

@router.post("/write-pipeline-config")
async def write_pipeline_config(request: PipelineConfigRequest):
    """Write pipeline configuration YAML file"""
    try:
        home_dir = get_home_dir()
        pipeline_dir = Path(home_dir) / ".docetl" / request.namespace / "pipelines"
        config_dir = pipeline_dir / "configs"
        name_dir = pipeline_dir / request.name / "intermediates"
        
        config_dir.mkdir(parents=True, exist_ok=True)
        name_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = config_dir / f"{request.name}.yaml"
        with file_path.open("w") as f:
            f.write(request.config)
            
        return {
            "filePath": str(file_path),
            "inputPath": request.input_path,
            "outputPath": request.output_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write pipeline configuration: {str(e)}")

@router.get("/read-file")
async def read_file(path: str):
    """Read file contents"""
    try:
        if path.startswith(("http://", "https://")):
            # For HTTP URLs, we'll need to implement request handling
            raise HTTPException(status_code=400, detail="HTTP URLs not supported in this endpoint")
            
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(path)
    except Exception as e:
        print(f"Failed to read file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

@router.get("/read-file-page")
async def read_file_page(path: str, page: int = 0, chunk_size: int = 500000):
    """Read file contents by page"""
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        file_size = file_path.stat().st_size
        start = page * chunk_size
        
        with file_path.open("rb") as f:
            f.seek(start)
            content = f.read(chunk_size).decode("utf-8")
            
        return {
            "content": content,
            "totalSize": file_size,
            "page": page,
            "hasMore": start + len(content) < file_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

@router.get("/serve-document/{path:path}")
async def serve_document(path: str):
    """Serve document files"""
    try:
        # Security check for path traversal
        if ".." in path:
            raise HTTPException(status_code=400, detail="Invalid file path")
            
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(
            path=file_path,
            filename=file_path.name,
            headers={"Cache-Control": "public, max-age=3600"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")

@router.get("/check-file")
async def check_file(path: str):
    """Check if a file exists without reading it"""
    try:
        file_path = Path(path)
        exists = file_path.exists()
        return {"exists": exists}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check file: {str(e)}")
