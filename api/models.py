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


class AnnotationSyncRequest(BaseModel):
    """Request model for annotation sync."""
    
    use_extended: bool = True
    clear_existing: bool = False


class AnnotationStats(BaseModel):
    """Statistics about annotations."""
    
    total_annotations: int
    completed_transcriptions: int
    completed_extended: int
    total_videos: int
    videos_with_annotations: int
    vector_store_annotations: int


class VideoMetadata(BaseModel):
    """Video metadata for streaming and display."""
    
    video_id: str
    filename: str
    filepath: str
    start_time: float
    end_time: float
    duration: float
    video_url: str
    annotation_id: Optional[int] = None
    project_name: Optional[str] = None
