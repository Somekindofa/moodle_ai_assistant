# Multimodal RAG Implementation Guide

## Overview

This implementation enables your Moodle AI Assistant to consume video annotation data from the SQLite database (`annotations.db`) and embed it into the ChromaDB vector store for semantic search and retrieval.

## Architecture

### Key Components

1. **AnnotationService** (`services/annotation_service.py`)
   - Reads from SQLite database at `chroma_langchain_db/elicitations_db/annotations.db`
   - Fetches completed annotations with video metadata
   - Converts annotations to LangChain Document objects
   - Provides statistics about annotation data

2. **RAGService Extensions** (`services/rag_service.py`)
   - New methods for syncing annotations to ChromaDB
   - Support for full sync and incremental sync
   - Ability to clear and rebuild annotation documents
   - Document counting for monitoring

3. **Pipeline Integration** (`pipeline.py`)
   - Auto-sync annotations on startup
   - Manual sync trigger methods
   - Combined statistics from database and vector store

4. **API Endpoints** (`api/routes.py`)
   - `POST /sync-annotations` - Manually trigger sync
   - `GET /annotation-stats` - Get comprehensive statistics

## Document Structure

Each annotation generates up to 2 documents in the vector store:

### Raw Transcription Document
```python
{
    "page_content": "raw transcription text",
    "metadata": {
        "annotation_id": 123,
        "video_id": 456,
        "video_filename": "tutorial.mp4",
        "video_filepath": "/path/to/video.mp4",
        "start_time": 10.5,
        "end_time": 45.2,
        "duration": 34.7,
        "audio_filepath": "/path/to/audio.wav",
        "source_type": "uploaded",
        "project_name": "woodworking",
        "annotation_created_at": "2025-10-20T10:30:00",
        "type": "video_annotation",
        "transcript_type": "raw",
        "source": "tutorial.mp4#123_raw"
    }
}
```

### Extended Transcript Document (LLM-enhanced)
```python
{
    "page_content": "enhanced transcript text",
    "metadata": {
        # ... same metadata as raw ...
        "transcript_type": "extended",
        "source": "tutorial.mp4#123_extended"
    }
}
```

## Usage

### Automatic Sync on Startup

The pipeline automatically syncs annotations when the server starts:

```python
# In pipeline.py
def _auto_sync_annotations(self):
    stats = self.annotation_service.get_annotation_stats()
    if stats.get("completed_extended", 0) > 0:
        count = self.rag_service.sync_annotations_to_vector_store(
            use_extended=True,
            clear_existing=False
        )
```

### Manual Sync via API

**Full Sync (keeps existing documents):**
```bash
curl -X POST http://localhost:8000/api/sync-annotations \
  -H "Content-Type: application/json" \
  -d '{"use_extended": true, "clear_existing": false}'
```

**Rebuild from Scratch:**
```bash
curl -X POST http://localhost:8000/api/sync-annotations \
  -H "Content-Type: application/json" \
  -d '{"use_extended": true, "clear_existing": true}'
```

**Get Statistics:**
```bash
curl http://localhost:8000/api/annotation-stats
```

Response:
```json
{
  "total_annotations": 50,
  "completed_transcriptions": 45,
  "completed_extended": 40,
  "total_videos": 10,
  "videos_with_annotations": 8,
  "vector_store_annotations": 80
}
```

### Programmatic Usage

```python
from pipeline import MoodleAIAssistantPipeline

pipeline = MoodleAIAssistantPipeline()

# Manual sync
count = pipeline.sync_annotations(
    use_extended=True,
    clear_existing=False
)
print(f"Synced {count} documents")

# Get stats
stats = pipeline.get_annotation_stats()
print(f"Vector store has {stats['vector_store_annotations']} annotation documents")
```

### Incremental Sync

For periodic updates (e.g., every 5 minutes):

```python
from datetime import datetime, timedelta

# Get annotations from last 5 minutes
last_sync = datetime.now() - timedelta(minutes=5)
count = pipeline.rag_service.sync_new_annotations(
    since_timestamp=last_sync,
    use_extended=True
)
```

## Retrieval Behavior

When a user queries the system:

1. **Semantic Search**: Query embeddings are matched against both raw and extended transcripts
2. **MMR Selection**: Uses Max Marginal Relevance to diversify results (k=15 by default)
3. **Metadata Preserved**: Retrieved documents include all video metadata
4. **Video Segment Linking**: Frontend can use `video_filepath`, `start_time`, `end_time` to display or jump to video segments

### Example Retrieval

```python
# User asks: "How do I cut dovetail joints?"
# System retrieves annotation documents with metadata:
{
    "page_content": "To cut dovetail joints, start by marking...",
    "metadata": {
        "video_filename": "woodworking_basics.mp4",
        "start_time": 125.5,
        "end_time": 180.3,
        "transcript_type": "extended"
    }
}

# Frontend can then:
# 1. Display the text context
# 2. Show video player at 02:05 (125.5s)
# 3. Highlight that this is from an extended transcript
```

## Database Schema Reference

The implementation expects this SQLite schema:

### annotations table
- `id` - Annotation ID
- `video_id` - Foreign key to videos
- `start_time` - Segment start (seconds)
- `end_time` - Segment end (seconds)
- `transcription` - Raw transcript
- `transcription_status` - "completed" for sync
- `extended_transcript` - LLM-enhanced version
- `extended_transcript_status` - "completed" for sync
- `audio_filepath` - Path to audio file
- `created_at` / `updated_at` - Timestamps

### videos table
- `id` - Video ID
- `filename` - Video filename
- `filepath` - Full path to video
- `duration` - Total video duration
- `source_type` - "uploaded", "local", "gdrive"
- `batch_position` - Position in batch
- `project_id` - Foreign key to projects

### projects table
- `id` - Project ID
- `name` - Project name
- `description` - Project description

## Configuration

Database path is configurable in `AnnotationService` initialization:

```python
# Default location
annotation_service = AnnotationService(
    config_manager,
    db_path="chroma_langchain_db/elicitations_db/annotations.db"
)
```

## Migration from Old Documents Folder

### Before (Old Approach)
```
documents/
├── GBL_Doc1.txt
├── GBL_Doc2.txt
└── ...
```

### After (New Approach)
```
chroma_langchain_db/
└── elicitations_db/
    └── annotations.db  # Source of truth
```

**To migrate:**
1. Ensure `annotations.db` has completed annotations
2. Run sync: `POST /sync-annotations` with `clear_existing: true`
3. Delete `documents/` folder (or keep for reference)
4. Vector store now contains only annotation-based documents

## Monitoring and Debugging

### Check Sync Status
```python
import logging
logging.basicConfig(level=logging.INFO)

# Watch logs during startup:
# "Retrieved X completed annotations"
# "Synced Y annotation documents to vector store"
```

### Verify Vector Store
```python
# Count annotation documents
count = pipeline.rag_service.get_annotation_documents_count()
print(f"Annotation documents in vector store: {count}")

# Inspect vector store
vector_data = pipeline.rag_service.get_vector_store_data()
for metadata in vector_data['metadatas']:
    if metadata.get('type') == 'video_annotation':
        print(f"Annotation {metadata['annotation_id']}: {metadata['source']}")
```

### Clear Annotation Documents Only
```python
# Remove only annotation documents, keep other documents
pipeline.rag_service._clear_annotation_documents()
```

## Next Steps

### Periodic Sync
Implement a background task to sync periodically:

```python
import asyncio

async def periodic_sync():
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        count = pipeline.sync_annotations()
        logger.info(f"Periodic sync: {count} documents updated")

# Add to FastAPI startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_sync())
```

### File System Watchdog
Monitor database file changes:

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AnnotationDBHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('annotations.db'):
            logger.info("Database updated, triggering sync...")
            pipeline.sync_annotations()
```

### Feedback Integration
Use the `feedback` field to weight or filter results:

```python
# In RAGService.similarity_search()
# Filter to only positive feedback annotations
results = self.vector_store.max_marginal_relevance_search(
    query, 
    k=k,
    filter={"type": "video_annotation", "feedback": 1}
)
```

## Troubleshooting

### No annotations synced
- Check database path is correct
- Verify annotations have `transcription_status = 'completed'`
- Check `extended_transcript_status = 'completed'` if using extended

### Database not found
- Ensure database exists at `chroma_langchain_db/elicitations_db/annotations.db`
- Check file permissions
- Verify relative path from server root

### Vector store errors
- Check ChromaDB persistence directory exists
- Verify embeddings model is accessible
- Check disk space for vector store growth

## Benefits Over Old Approach

1. **Single Source of Truth**: Database is canonical, vector store is derived
2. **Rich Metadata**: Video segments linked to timestamps for playback
3. **Incremental Updates**: Only sync new annotations, not full rebuild
4. **Dual Transcripts**: Both raw and enhanced available for retrieval
5. **Scalable**: Database can grow without file system clutter
6. **Queryable**: Can filter/join annotations with video/project data
7. **Atomic Updates**: Database transactions ensure consistency

## Summary

This implementation transforms your RAG system from static document loading to dynamic video annotation retrieval. The SQLite database provides structured metadata about video segments, while ChromaDB enables semantic search over the transcript content. Together, they enable multimodal RAG where text queries can retrieve specific video segments for playback or reference.
