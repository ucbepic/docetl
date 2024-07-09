# ---
# deploy: true
# cmd: ["modal", "serve", "serve_streamlit.py"]
# ---

import shlex
import subprocess
from pathlib import Path
import modal

# Define container dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "streamlit", "pandas", "numpy", "scikit-learn", "altair"
)

app = modal.App(name="streamlit-chronicling-america", image=image)

# Mount the `st_app_ca.py` script inside the container at a pre-defined path
streamlit_script_local_path = Path(__file__).parent / "st_app_ca.py"
streamlit_script_remote_path = Path("/root/st_app_ca.py")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "st_app_ca.py not found! Place the script with your streamlit app in the same directory."
    )

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)

# Volume for persistent storage
volume = modal.Volume.from_name("chronicling-america-vol")


# Define the function to run the Streamlit server
@app.function(
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount],
    volumes={"/my_vol": volume},
)
@modal.web_server(8000)
def run():
    # Print what is inside the /my_vol directory

    target = shlex.quote(str(streamlit_script_remote_path))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)
