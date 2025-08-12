"""API routes for the Moodle AI Assistant backend server."""

import os
import asyncio
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from api.models import ChatRequest, SystemStatus, HealthResponse
from pipeline import MoodleAIAssistantPipeline
from config.settings import ConfigurationManager


router = APIRouter()

# Initialize pipeline
config_manager = ConfigurationManager()
pipeline = MoodleAIAssistantPipeline(config_manager)


def check_documents_folder() -> bool:
    """Check if Documents folder exists in the workspace."""
    return os.path.exists("Documents") and os.path.isdir("Documents")


async def generate_sse_response(user_message: str, history: list) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events response for streaming."""
    try:
        async for chunk in pipeline.generate_response(user_message, history):
            # Format as SSE event
            yield f"data: {chunk}\n\n"
        
        # Send end-of-stream marker
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        yield f"data: ERROR: {str(e)}\n\n"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status."""
    docs_exist = check_documents_folder()
    kb_data = pipeline.get_knowledge_base_status()
    vector_count = len(kb_data) if not kb_data.empty else 0
    
    return SystemStatus(
        mode="rag" if docs_exist or vector_count > 0 else "generation",
        documents_folder_exists=docs_exist,
        vector_store_count=vector_count
    )


@router.post("/chat")
async def chat_stream(request: ChatRequest):
    """Main chat endpoint with streaming response."""
    try:
        history = request.history or []
        return EventSourceResponse(
            generate_sse_response(request.message, history),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
