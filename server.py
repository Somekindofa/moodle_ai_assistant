"""
FastAPI backend server for Moodle AI Assistant.

This server acts as a proxy between the JavaScript frontend and the RAG system,
providing streaming responses and document-based or pure generation modes.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from config.settings import setup_logging


# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Moodle AI Assistant Backend Server...")

    # Check for Documents folder on startup
    docs_exist = os.path.exists("Documents") and os.path.isdir("Documents")
    mode = "RAG" if docs_exist else "Generation"
    logger.info(f"Server mode: {mode} (Documents folder exists: {docs_exist})")

    yield

    logger.info("Shutting down Moodle AI Assistant Backend Server...")


# Create FastAPI app
app = FastAPI(
    title="Moodle AI Assistant Backend",
    description="Backend proxy server for Moodle AI Assistant block plugin",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for JavaScript frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse(
        {
            "message": "Moodle AI Assistant Backend Server",
            "version": "1.0.0",
            "docs": "/docs",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
