import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.app.routes import pipeline, convert, filesystem
from dotenv import load_dotenv

load_dotenv()

# Read backend configuration from .env
host = os.getenv("BACKEND_HOST", "127.0.0.1")
port = int(os.getenv("BACKEND_PORT", 8000))
reload = os.getenv("BACKEND_RELOAD", "False").lower() == "true"

# Set default allow_origins if BACKEND_ALLOW_ORIGINS is not provided
allow_origins = os.getenv("BACKEND_ALLOW_ORIGINS", "http://localhost:3000").split(",")

app = FastAPI()
os.environ["USE_FRONTEND"] = "true"

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(pipeline.router)
app.include_router(convert.router)
app.include_router(filesystem.router, prefix="/fs")

@app.get("/")
async def root():
    return {"message": "DocETL API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app.main:app", host=host, port=port, reload=reload)
