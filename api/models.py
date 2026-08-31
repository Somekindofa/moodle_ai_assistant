"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class ChatMessage(BaseModel):
    """Represents a chat message."""
    
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=4000)
    conversation_thread_id: str = Field(..., max_length=255)
    selected_domain: Optional[str] = Field(None, max_length=100)
    course_id: Optional[str] = Field(None, max_length=20)
    is_first_message: bool = False         # True on first message — triggers title generation
    disable_rerank: bool = False           # Ablation flag: skip cross-encoder reranking when True
    user_id: Optional[int] = None          # NEW — validated by chat_proxy.php
    previous_sources: Optional[List[Dict[str, Any]]] = None  # NEW — frontend's prior-turn video cards
    previous_message: Optional[str] = None                   # NEW — frontend's last non-pagination message


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


class AnnotationIngestRequest(BaseModel):
    """Payload for pushing a single completed annotation into the vector store."""

    annotation_id: int
    video_id: int
    transcription: str
    start_time: float
    end_time: float
    video_filename: str
    video_filepath: str
    source_type: str = "local"
    project_name: str = "unknown"
    audio_filepath: str = ""
    allowed_cohort_id: Optional[int] = None          # None = open access
    language: Optional[str] = None                   # ISO 639-1 code detected by Whisper


class CourseModuleIngestRequest(BaseModel):
    """Payload for ingesting a Moodle course module into ChromaDB."""

    course_id: str
    module_id: str
    module_type: str                        # 'page', 'label', 'resource'
    module_name: str
    section_name: str = ""
    content_html: Optional[str] = None     # HTML string for page/label
    content_raw_b64: Optional[str] = None  # base64-encoded file bytes for resource
    file_extension: Optional[str] = None   # 'pdf' or 'docx'


class CourseModuleDeleteRequest(BaseModel):
    """Payload for removing a module's chunks from ChromaDB."""

    course_id: str
    module_id: str


class CourseDeleteRequest(BaseModel):
    """Payload for dropping an entire course collection from ChromaDB."""

    course_id: str


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


class ResyncProjectRequest(BaseModel):
    """Payload for re-tagging a project's ChromaDB documents with a new cohort."""
    project_name: str = Field(..., max_length=255)
    allowed_cohort_id: Optional[int] = None   # None = open access
