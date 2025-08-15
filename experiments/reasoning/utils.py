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
    .add_local_python_source("experiments", ignore=["**/.venv/*"])
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

