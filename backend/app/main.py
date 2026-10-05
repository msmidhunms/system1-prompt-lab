"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router as api_router

app = FastAPI(
    title="System 1 Prompt Lab",
    description="API of System 1 Prompt Lab: golden datasets, evaluation, model comparison and prompt optimization for small System 1 classifiers",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "System 1 Prompt Lab API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
