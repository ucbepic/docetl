import modal
from typing import Dict, Any, List, Optional

APP_NAME = "docetl-experiments"
VOLUME_NAME = "docetl-ro-experiments"
VOLUME_MOUNT_PATH = "/mnt/docetl-ro-experiments"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True) 

# Common image definition for experiments
image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_sync()
    .uv_pip_install("matplotlib", "Levenshtein", "nltk")
    .add_local_python_source("experiments", ignore=lambda p: (
        # Include othersystems only if it contains lotus_evaluation or pz_evaluation files
        ("othersystems" in str(p) and 
         "lotus_evaluation" not in str(p) and 
         "pz_evaluation" not in str(p)) or
        # Exclude large files by size (>300MB)
        (p.is_file() and p.stat().st_size > 300_000_000) or
        # Exclude specific file types
        p.suffix in {'.bin', '.sqlite3'} or
        # Exclude specific directories
        any(part in {'.venv', 'outputs', '__pycache__'} for part in p.parts) or
        # Exclude files starting with dot
        p.name.startswith('.') or
        # Exclude chroma directories
        '.chroma-' in str(p)
    ))
    .add_local_python_source("docetl", ignore=["**/.venv/*"])
)


def create_original_query_result(
    success: bool,
    cost: float,
    output_file_path: Optional[str] = None,
    sample_output: Optional[List[Dict[str, Any]]] = None,
    accuracy: Optional[float] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """Create an original query result dictionary that's serializable for Modal."""
    return {
        "success": success,
        "cost": cost,
        "output_file_path": output_file_path,
        "sample_output": sample_output or [],
        "accuracy": accuracy,
        "error": error
    }