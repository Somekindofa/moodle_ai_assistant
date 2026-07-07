"""API routes for the Moodle AI Assistant backend server."""

import json
import os
from datetime import datetime
from typing import AsyncGenerator, List, Optional
from venv import logger

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from api.models import (
    ChatRequest,
    SystemStatus,
    HealthResponse,
    AnnotationSyncRequest,
    AnnotationStats,
    AnnotationIngestRequest,
    CourseModuleIngestRequest,
    CourseModuleDeleteRequest,
    CourseDeleteRequest,
    ResyncProjectRequest,
)
from pipeline import MoodleAIAssistantPipeline
from config.settings import ConfigurationManager


router = APIRouter()

# Initialize pipeline
config_manager = ConfigurationManager()
pipeline = MoodleAIAssistantPipeline(config_manager)

# Define Json escape
json_escape = "\n"


def check_documents_folder() -> bool:
    """Check if Documents folder exists in the workspace."""
    return os.path.exists("Documents") and os.path.isdir("Documents")


async def generate_simplified_stream(
    user_messages: str,
    conversation_thread_id: str,
    selected_domain: Optional[str] = None,
    course_id: Optional[str] = None,
    is_first_message: bool = False,
    disable_rerank: bool = False,
    user_id: Optional[int] = None,         # NEW
) -> AsyncGenerator[str, None]:
    """Stream the RAG pipeline response as JSON-lines.

    Delegates to pipeline.stream_response() which runs the PRF retrieval
    nodes then streams LLM tokens individually.  The client receives:
      {"event": "conversation_title", "data": "..."}  — only on first message
      {"event": "video_metadata", "data": {...}}       — optional source card
      {"event": "token", "data": "<text>"}             — one per token
      {"event": "documents", "data": [...]}            — document sources
      {"event": "rerank_debug", "data": {...}}         — reranker diagnostics
      {"content": "[DONE]"}                            — terminal marker
    """
    async for line in pipeline.stream_response(
        user_messages,
        conversation_thread_id=conversation_thread_id,
        selected_domain=selected_domain,
        course_id=course_id,
        is_first_message=is_first_message,
        disable_rerank=disable_rerank,
        user_id=user_id,                   # NEW
    ):
        yield line


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
    """Streaming chat — requires a validated user_id from chat_proxy.php."""
    if not request.user_id or request.user_id <= 0:
        raise HTTPException(status_code=403, detail="user_id required")

    return StreamingResponse(
        generate_simplified_stream(
            request.message,
            request.conversation_thread_id,
            request.selected_domain,
            request.course_id,
            request.is_first_message,
            request.disable_rerank,
            user_id=request.user_id,        # NEW
        ),
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"},
    )


@router.post("/ingest-annotation")
async def ingest_annotation(request: AnnotationIngestRequest):
    """Ingest a single completed annotation directly into the vector store.

    Called by the videoelicit backend immediately after a transcription is marked
    'completed', so that new elicitations are searchable in real time without
    waiting for a manual /sync-annotations call.
    """
    try:
        from langchain_core.documents.base import Document

        annotation_dict = {
            "annotation_id":    request.annotation_id,
            "video_id":         request.video_id,
            "transcription":    request.transcription,
            "start_time":       request.start_time,
            "end_time":         request.end_time,
            "video_filename":   request.video_filename,
            "video_filepath":   request.video_filepath,
            "source_type":      request.source_type,
            "project_name":     request.project_name,
            "audio_filepath":   request.audio_filepath,
            "allowed_cohort_id": request.allowed_cohort_id,   # None = open access
            # extended_transcript not available yet at transcription time
            "extended_transcript": None,
        }

        docs = pipeline.annotation_service.annotation_to_documents(
            annotation_dict, use_extended=False
        )

        if not docs:
            return {"status": "skipped", "reason": "no documents produced", "annotation_id": request.annotation_id}

        pipeline.rag_service.add_documents(docs)

        return {
            "status": "ok",
            "documents_added": len(docs),
            "annotation_id": request.annotation_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resync-project-annotations")
async def resync_project_annotations(request: ResyncProjectRequest):
    """Delete and re-ingest all ChromaDB documents for a project with updated cohort metadata.

    Called automatically by the video elicitation backend when an expert
    changes the allowed_cohort_id on an existing project.
    """
    try:
        project_name = request.project_name

        # 1. Fetch annotations from SQLite BEFORE deleting ChromaDB
        annotations = pipeline.annotation_service.get_completed_annotations(
            include_extended=True
        )
        project_annotations = [
            a for a in annotations if (a.get("project_name") or "unknown") == project_name
        ]

        if not project_annotations:
            return {
                "status": "ok",
                "documents_resynced": 0,
                "project_name": project_name,
                "allowed_cohort_id": request.allowed_cohort_id,
            }

        # 2. Delete existing ChromaDB docs for this project (after fetch succeeds)
        existing = pipeline.rag_service.vector_store.get(
            where={"project_name": project_name}
        )
        if existing and existing.get("ids"):
            pipeline.rag_service.vector_store.delete(ids=existing["ids"])
            logger.info(
                f"resync: deleted {len(existing['ids'])} docs for project '{project_name}'"
            )

        # 3. Inject the new cohort_id into each annotation (safe copy, no mutation)
        project_annotations = [
            {**ann, "allowed_cohort_id": request.allowed_cohort_id}
            for ann in project_annotations
        ]

        docs = []
        for ann in project_annotations:
            docs.extend(
                pipeline.annotation_service.annotation_to_documents(ann, use_extended=True)
            )

        if docs:
            pipeline.rag_service.add_documents(docs)

        return {
            "status": "ok",
            "documents_resynced": len(docs),
            "project_name": project_name,
            "allowed_cohort_id": request.allowed_cohort_id,
        }

    except Exception as e:
        logger.error(f"resync-project-annotations failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync-annotations")
async def sync_annotations(request: Optional[AnnotationSyncRequest] = None):
    """Manually trigger annotation sync from SQLite to ChromaDB."""
    try:
        use_extended = request.use_extended if request else True
        clear_existing = request.clear_existing if request else False

        count = pipeline.sync_annotations(
            use_extended=use_extended, clear_existing=clear_existing
        )

        return {
            "status": "success",
            "documents_synced": count,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/annotation-stats", response_model=AnnotationStats)
async def get_annotation_stats():
    """Get statistics about annotations in database and vector store."""
    try:
        stats = pipeline.get_annotation_stats()
        return AnnotationStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Video ID to filepath mapping cache (in production, use Redis or similar)
_video_cache: dict[str, str] = {}


def _register_video_path(video_id: str, filepath: str) -> None:
    """Register a video_id to filepath mapping."""
    _video_cache[video_id] = filepath


def _get_video_path(video_id: str) -> str:
    """Get filepath for a video_id, or search in vector store."""
    import hashlib
    from pathlib import Path

    # Check cache first
    if video_id in _video_cache:
        return _video_cache[video_id]

    # Search vector store for matching video_id
    try:
        vector_data = pipeline.rag_service.get_vector_store_data()

        if not vector_data.get("metadatas"):
            raise HTTPException(status_code=404, detail="No videos in database")

        for metadata in vector_data["metadatas"]:
            if metadata.get("type") == "video_annotation":
                filepath = metadata.get("video_filepath")
                annotation_id = metadata.get("annotation_id")

                if filepath and annotation_id:
                    # Regenerate video_id to match
                    computed_id = hashlib.md5(
                        f"{filepath}_{annotation_id}".encode()
                    ).hexdigest()

                    if computed_id == video_id:
                        _video_cache[video_id] = filepath
                        return filepath

        raise HTTPException(
            status_code=404, detail=f"Video not found for ID: {video_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving video path: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving video")


@router.get("/video/stream/{video_id}")
async def stream_video(video_id: str, request: Request):
    """
    Stream video with HTTP range request support for seeking.

    Supports partial content requests (HTTP 206) which enables:
    - Video seeking in the browser
    - Efficient bandwidth usage
    - Resume capability
    """
    from pathlib import Path
    import os
    import re

    # Security: Validate video_id format (MD5 hash)
    if not re.match(r"^[a-f0-9]{32}$", video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID format")

    # Get video filepath
    video_path = _get_video_path(video_id)

    # Security: Ensure file exists and is readable
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    # Security: Prevent path traversal and enforce allowlisted directories
    _ALLOWED_VIDEO_DIRS = [
        Path("/opt/video_elicitation_annotation_tool").resolve(),
        Path("/var/www/html").resolve(),
        Path("/tmp").resolve(),
    ]
    try:
        video_path_resolved = Path(video_path).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not any(
        str(video_path_resolved).startswith(str(d))
        for d in _ALLOWED_VIDEO_DIRS
    ):
        raise HTTPException(status_code=403, detail="Video path not permitted")

    file_size = os.path.getsize(video_path)

    # Parse Range header
    range_header = request.headers.get("range")

    if not range_header:
        # No range requested - send entire file
        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )

    # Parse range header (format: "bytes=start-end")
    range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)

    if not range_match:
        raise HTTPException(status_code=416, detail="Invalid range header")

    start = int(range_match.group(1))
    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1

    # Validate range
    if start >= file_size or end >= file_size or start > end:
        raise HTTPException(
            status_code=416,
            detail="Range not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    chunk_size = end - start + 1

    # Stream the requested byte range
    def iterfile():
        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = chunk_size

            while remaining > 0:
                read_size = min(8192, remaining)  # 8KB chunks
                data = f.read(read_size)

                if not data:
                    break

                remaining -= len(data)
                yield data

    return StreamingResponse(
        iterfile(),
        status_code=206,  # Partial Content
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
        },
    )


# ─────────────────────────────────────────────────────────────────
# Course content ingestion endpoints
# ─────────────────────────────────────────────────────────────────

@router.post("/ingest-course-module")
async def ingest_course_module(request: CourseModuleIngestRequest):
    """Ingest a Moodle course module (page/label/resource) into its per-course ChromaDB collection.

    Called by the Moodle plugin's event observer immediately after a teacher
    creates or updates a course module.  Chunking and embedding happen here.
    """
    try:
        count = pipeline.course_rag_service.ingest_module(
            course_id=request.course_id,
            module_id=request.module_id,
            module_type=request.module_type,
            module_name=request.module_name,
            section_name=request.section_name,
            content_html=request.content_html,
            content_raw_b64=request.content_raw_b64,
            file_extension=request.file_extension,
        )
        return {
            "status": "ok",
            "chunks_indexed": count,
            "collection": f"course_{request.course_id}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-course-module")
async def delete_course_module(request: CourseModuleDeleteRequest):
    """Remove all chunks belonging to a course module from ChromaDB."""
    try:
        deleted = pipeline.course_rag_service.delete_module(
            course_id=request.course_id,
            module_id=request.module_id,
        )
        return {"status": "ok", "chunks_deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-course")
async def delete_course(request: CourseDeleteRequest):
    """Drop the entire ChromaDB collection for a course."""
    try:
        pipeline.course_rag_service.delete_collection(course_id=request.course_id)
        return {"status": "ok", "collection_deleted": f"course_{request.course_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/annotations-dashboard")
async def get_annotations_dashboard():
    """Return all video elicitation annotations with video and user metadata for the dashboard."""
    import pymysql
    import pymysql.cursors

    conn = pymysql.connect(
        host="localhost",
        user="moodleuser",
        password=os.getenv("MOODLE_DB_PASSWORD", ""),
        database="moodle",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    a.id,
                    a.craft,
                    a.task,
                    a.starttime,
                    a.endtime,
                    a.transcription,
                    a.transcriptionstatus,
                    a.reviewstatus,
                    a.judgestatus,
                    a.taggingstatus,
                    a.issalient,
                    a.tags,
                    a.feedbackchoices,
                    a.timecreated,
                    v.filename  AS video_filename,
                    v.source_type,
                    u.username,
                    u.firstname,
                    u.lastname
                FROM mdl_local_videoelicit_annotations a
                LEFT JOIN mdl_local_videoelicit_videos v ON v.id = a.videoid
                LEFT JOIN mdl_user u ON u.id = a.userid
                ORDER BY a.timecreated DESC
            """)
            rows = cursor.fetchall()

        for row in rows:
            # Parse JSON fields stored as longtext
            for field in ("tags", "feedbackchoices"):
                raw = row.get(field)
                if raw:
                    try:
                        row[field] = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        row[field] = []
                else:
                    row[field] = []
            # Human-readable timestamp
            row["timecreated"] = datetime.fromtimestamp(row["timecreated"]).isoformat() if row["timecreated"] else None

        return {"annotations": rows, "total": len(rows)}
    finally:
        conn.close()


@router.get("/course-status/{course_id}")
async def get_course_status(course_id: str):
    """Return indexing statistics for a course collection."""
    try:
        return pipeline.course_rag_service.get_course_status(course_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
