import modal

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

