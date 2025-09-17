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

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, AnyMessage
from langgraph.types import StreamMode


router = APIRouter()

# Initialize pipeline
config_manager = ConfigurationManager()
pipeline = MoodleAIAssistantPipeline(config_manager)

# Define Json escape
json_escape = "\n"

def check_documents_folder() -> bool:
    """Check if Documents folder exists in the workspace."""
    return os.path.exists("Documents") and os.path.isdir("Documents")


async def generate_simplified_stream(user_messages: str, stream_mode: StreamMode) -> AsyncGenerator[str, None]:
    """Generate a simpler JSON stream."""
    try:
        if stream_mode=="values" or stream_mode=="updates":
            async for (messages, context) in pipeline.generate_response(user_messages, stream_mode=stream_mode):
                serializable_documents = []
                if context:
                    for doc in context:
                        serializable_documents.append({
                            "id": getattr(doc, 'id', None),
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        })

                # Convert messages to serializable format
                serializable_messages = []
                if messages:
                    for msg in messages:
                        serializable_messages.append({
                            "content": getattr(msg, 'content', str(msg)),
                            "type": getattr(msg, 'type', 'unknown'),
                            "id": getattr(msg, 'id', None)
                        })

                yield json.dumps({
                    "content": serializable_messages, 
                    "documents": serializable_documents
                }) + json_escape
            yield json.dumps({"content": "[DONE]"}) + json_escape

        elif stream_mode=="messages":
            async for chunk in pipeline.generate_response(user_messages, stream_mode=stream_mode):
                yield json.dumps({"messages": chunk}) + json_escape
            yield json.dumps({"messages": "[DONE]"}) + json_escape

    except GeneratorExit:
        logger.info("Client disconnected during streaming")
        raise

    except Exception as e:
        yield json.dumps({"error": str(e)}) + json_escape


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
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Simplified chat endpoint with streaming response."""
    try:
        return StreamingResponse(
            generate_simplified_stream(request.message, stream_mode="updates"),
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
