"""API routes for the Moodle AI Assistant backend server."""

import json
import os
import asyncio
from datetime import datetime
from typing import AsyncGenerator
from venv import logger

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.models import ChatMessage, ChatRequest, SystemStatus, HealthResponse
from pipeline import MoodleAIAssistantPipeline
from config.settings import ConfigurationManager


router = APIRouter()

# Initialize pipeline
config_manager = ConfigurationManager()
pipeline = MoodleAIAssistantPipeline(config_manager)


def check_documents_folder() -> bool:
    """Check if Documents folder exists in the workspace."""
    return os.path.exists("Documents") and os.path.isdir("Documents")


async def generate_simplified_stream(user_message: str, history: list[ChatMessage]) -> AsyncGenerator[str, None]:
    """Generate a simpler JSON stream."""
    try:
        logger.info(f"\nReceived user message: {user_message}")
        logger.info(f"Conversation history: {history}\n")
        async for chunk in pipeline.generate_response(user_message, history):
            yield json.dumps({"content": chunk}) + "\n"
        yield json.dumps({"content": "[DONE]"}) + "\n"

    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat())


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status."""
    docs_exist = check_documents_folder()
    kb_data = pipeline.get_knowledge_base_status()
    vector_count = len(kb_data) if not kb_data.empty else 0

    return SystemStatus(
        mode="rag" if docs_exist or vector_count > 0 else "generation",
        documents_folder_exists=docs_exist,
        vector_store_count=vector_count,
    )


@router.post("/chat")
async def chat_stream(request: ChatRequest):
    """Simplified chat endpoint with streaming response."""
    try:
        history = request.history or []
        return StreamingResponse(
            generate_simplified_stream(request.message, history),
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
