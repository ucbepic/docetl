import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import modal
from server.app.routes import pipeline, convert, filesystem
from dotenv import load_dotenv
import docetl
import server

# Load environment variables
load_dotenv()

# Create FastAPI and Modal apps
app = modal.App(
    "docetl-server", 
    image=modal.Image.debian_slim().add_local_file("pyproject.toml", "/pyproject.toml", copy=True)
    .add_local_file("poetry.lock", "/poetry.lock", copy=True)
    .add_local_file("README.md", "/README.md", copy=True)
    # .add_local_python_source("docetl", "server", copy=True)
    .add_local_dir("docetl", remote_path="/docetl", copy=True)
    .add_local_dir("server", remote_path="/server", copy=True)
    # .poetry_install_from_file("/root/pyproject.toml", force_build=True)
    .pip_install("poetry")
    .pip_install("pyyaml")
    .pip_install("supabase")
    .run_commands(["poetry config virtualenvs.create false", "poetry install --all-extras --no-root && poetry install --all-extras"])
)
vol = modal.Volume.from_name("docetl-backend-api", create_if_missing=True)


@app.function(
    volumes={"/modal": vol},
    secrets=[
        modal.Secret.from_dict({"DOCETL_HOME_DIR": "/modal", "USE_FRONTEND": "true"}),
        modal.Secret.from_name("docetl-secrets"),
        modal.Secret.from_name("supabase-secret"),
    ],
    keep_warm=1,
    timeout=3600
)
@modal.asgi_app()
def main():
    web_app = FastAPI()

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://www.docetl.org", "http://192.168.4.201:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include all routers
    web_app.include_router(pipeline.router)
    web_app.include_router(convert.router)
    web_app.include_router(filesystem.router, prefix="/fs")

    @web_app.get("/")
    async def root():
        return {"message": "DocETL API is running"}

    return web_app

if __name__ == "__main__":
    app.run()