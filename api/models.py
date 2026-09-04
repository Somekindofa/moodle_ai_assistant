"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Literal


# Metadata values that mean "no project was recorded", not a real project name.
#
# ``AnnotationIngestRequest.project_name`` below defaults to the literal string
# "unknown", so this placeholder is *stored data* in ChromaDB metadata, not an
# accident of a missing key. Anything that treats ``project_name`` (or the
# ``craft`` tag beside it) as a human-readable label must filter these first:
#
#   - ResyncProjectRequest.reject_placeholder_project refuses them as a resync
#     target, because resyncing "unknown" would delete the whole legacy corpus;
#   - pipeline._build_ambiguous_clarification refuses to name them to a
#     learner, because it once told a real user their question matched
#     "plusieurs sujets du corpus (unknown)".
#
# Kept here, next to the field that produces the placeholder, so both guards
# read from one list instead of each re-spelling "unknown".
PLACEHOLDER_PROJECT_NAMES = frozenset({"unknown", "none", "null", "n/a", "na", "-"})


def is_placeholder_project_name(value: Any) -> bool:
    """True when ``value`` is absent, blank, or a known placeholder.

    Accepts any type (ChromaDB metadata is not schema-checked on read) and
    never raises — a non-string value is stringified before comparison.
    """
    if value is None:
        return True
    cleaned = str(value).strip()
    return not cleaned or cleaned.lower() in PLACEHOLDER_PROJECT_NAMES

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
    craft: Optional[str] = None                      # feeds CRAFT_COHORT_MAP safety net
    task: Optional[str] = None
    annotation_created_at: Optional[str] = None
    annotation_updated_at: Optional[str] = None

    def to_annotation_dict(self) -> Dict[str, Any]:
        """Render this request in the shape AnnotationService expects.

        ``craft`` matters as much as ``allowed_cohort_id`` here: it drives the
        CRAFT_COHORT_MAP safety net, which is what keeps partner content
        restricted when a project's cohort was never set. Dropping it would
        silently index new confidential recordings as open-access.
        """
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "transcription": self.transcription,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "video_filename": self.video_filename,
            "video_filepath": self.video_filepath,
            "source_type": self.source_type,
            "project_name": self.project_name,
            "audio_filepath": self.audio_filepath,
            "allowed_cohort_id": self.allowed_cohort_id,
            "craft": self.craft,
            "language": self.language,
            # AnnotationService indexes these directly rather than via .get(),
            # so they must be present. Omitting annotation_created_at made every
            # real-time ingest raise KeyError and 500.
            "annotation_created_at": self.annotation_created_at or "",
            "annotation_updated_at": self.annotation_updated_at or "",
            "task": self.task or "",
            # Not available yet at transcription time.
            "extended_transcript": None,
        }


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

    @field_validator("project_name")
    @classmethod
    def reject_placeholder_project(cls, value: str) -> str:
        """Refuse placeholder project names ('unknown' and friends) and blanks.

        A resync deletes every document matching the project name before
        re-adding from the live Moodle tables. Annotations indexed before the
        silo write-path worked all carry project_name='unknown', so resyncing
        that placeholder would delete the entire legacy corpus — including
        partner-confidential documents whose access labels would not survive
        the round trip. A placeholder is never a real project.

        Reads its list from PLACEHOLDER_PROJECT_NAMES so this guard and
        pipeline._build_ambiguous_clarification's cannot drift apart.
        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("project_name must not be blank")
        if is_placeholder_project_name(cleaned):
            raise ValueError(
                f"'{cleaned}' is a placeholder for annotations with no project, not a "
                "resync target — resyncing it would delete the legacy corpus. "
                "Use scripts/apply_craft_silo.py to relabel legacy documents."
            )
        return cleaned
