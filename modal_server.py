import os
from fastapi import FastAPI
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
    .add_local_python_source("docetl", "server", copy=True)
    # .poetry_install_from_file("/root/pyproject.toml", force_build=True)
    .pip_install("poetry")
    .run_commands(["poetry config virtualenvs.create false", "poetry install --all-extras --no-root && poetry install --all-extras"])
)
vol = modal.Volume.from_name("docetl-backend-api", create_if_missing=True)


@app.function(
    volumes={"/modal": vol},
    secrets=[
        modal.Secret.from_dict({"DOCETL_HOME_DIR": "/modal", "USE_FRONTEND": "true"}),
        modal.Secret.from_name("docetl-secrets"),
    ],
    keep_warm=1,
)
@modal.asgi_app()
def main():
    web_app = FastAPI()

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