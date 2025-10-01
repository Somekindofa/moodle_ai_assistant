"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

class ChatMessage(BaseModel):
    """Represents a chat message."""
    
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str
    conversation_thread_id: str


class SystemStatus(BaseModel):
    """Response model for system status."""

    mode: str  # "rag" or "generation"
    documents_folder_exists: bool
    vector_store_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    timestamp: str
