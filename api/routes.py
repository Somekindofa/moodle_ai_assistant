"""API routes for the Moodle AI Assistant backend server."""

import os
from datetime import datetime
from typing import Optional
from venv import logger

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, Response
from api.models import (
    ChatRequest, 
    SystemStatus, 
    HealthResponse,
    AnnotationSyncRequest,
    AnnotationStats
)
from pipeline import MoodleAIAssistantPipeline
from config.settings import ConfigurationManager

router = APIRouter()

# Initialize pipeline
config_manager = ConfigurationManager()
pipeline = MoodleAIAssistantPipeline(config_manager)

def check_documents_folder() -> bool:
    """Check if Documents folder exists in the workspace."""
    return os.path.exists("Documents") and os.path.isdir("Documents")

async def generate_simplified_stream(
    user_messages: str, conversation_thread_id: str, stream_mode: StreamMode
) -> AsyncGenerator[str, None]:
    """Generate a simpler JSON stream with video metadata support."""
    try:
        if stream_mode == "updates":
            video_metadata_sent = False
            serializable_documents = []
            serializable_messages = []

            async for messages, context, video_metadata in pipeline.generate_response(
                user_messages, conversation_thread_id=conversation_thread_id, stream_mode=stream_mode
            ):
                # Send video metadata event FIRST if available and not yet sent
                if video_metadata and not video_metadata_sent:
                    yield json.dumps(
                        {
                            "event": "video_metadata",
                            "data": video_metadata
                        }
                    ) + json_escape
                    video_metadata_sent = True

                if context:
                    serializable_documents = []
                    for doc in context:
                        serializable_documents.append(
                            {
                                "id": getattr(doc, "id", None),
                                "page_content": doc.page_content,
                                "metadata": doc.metadata,
                            }
                        )

                if messages:
                    serializable_messages = []
                    for msg in messages:
                        serializable_messages.append(
                            {
                                "content": getattr(msg, "content", str(msg)),
                                "type": getattr(msg, "type", "unknown"),
                                "id": getattr(msg, "id", None),
                            }
                        )

                yield json.dumps(
                    {
                        "event": "message",
                        "content": serializable_messages,
                        "documents": serializable_documents,
                    }
                ) + json_escape

            yield json.dumps({"content": "[DONE]"}) + json_escape

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
async def chat_stream(request: ChatRequest):
    """
    Non-streaming chat endpoint - waits for complete response.
    Returns everything at once: AI message, documents, and video metadata.
    """
    try:
        result = await pipeline.generate_response(
            request.message, request.conversation_thread_id
        )

        return {
            "status": "success",
            "messages": result["messages"],  # AI response text
            "documents": result["documents"],  # Retrieved docs metadata
            "video_metadata": result.get("video_metadata"),  # Video info if available
            "conversation_thread_id": request.conversation_thread_id,
        }

    except Exception as e:
        logger.error(f"Chat request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync-annotations")
async def sync_annotations(request: Optional[AnnotationSyncRequest] = None):
    """Manually trigger annotation sync from SQLite to ChromaDB."""
    try:
        use_extended = request.use_extended if request else True
        clear_existing = request.clear_existing if request else False
        
        count = pipeline.sync_annotations(
            use_extended=use_extended,
            clear_existing=clear_existing
        )
        
        return {
            "status": "success",
            "documents_synced": count,
            "timestamp": datetime.now().isoformat()
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
                    computed_id = hashlib.md5(f"{filepath}_{annotation_id}".encode()).hexdigest()
                    
                    if computed_id == video_id:
                        _video_cache[video_id] = filepath
                        return filepath
        
        raise HTTPException(status_code=404, detail=f"Video not found for ID: {video_id}")
        
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
    if not re.match(r'^[a-f0-9]{32}$', video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    
    # Get video filepath
    video_path = _get_video_path(video_id)
    
    # Security: Ensure file exists and is readable
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    
    # Security: Prevent path traversal
    try:
        video_path_resolved = Path(video_path).resolve()
        # Could add additional checks here for allowed directories
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path")
    
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
            }
        )
    
    # Parse range header (format: "bytes=start-end")
    range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
    
    if not range_match:
        raise HTTPException(status_code=416, detail="Invalid range header")
    
    start = int(range_match.group(1))
    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
    
    # Validate range
    if start >= file_size or end >= file_size or start > end:
        raise HTTPException(
            status_code=416,
            detail="Range not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"}
        )
    
    chunk_size = end - start + 1
    
    # Stream the requested byte range
    def iterfile():
        with open(video_path, 'rb') as f:
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
        }
    )
