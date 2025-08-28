"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatMessage(BaseModel):
    """Represents a chat message."""
    
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str
    history: Optional[List[ChatMessage]] = []


class SystemStatus(BaseModel):
    """Response model for system status."""

    mode: str  # "rag" or "generation"
    documents_folder_exists: bool
    vector_store_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    timestamp: str
