# Moodle AI Assistant - AI Agent Instructions

## Project Overview

A RAG-powered FastAPI backend that streams LLM responses to a Moodle JavaScript frontend. Architecture follows a **modular service-oriented pattern** with dependency injection.

**Core Flow**: JavaScript Frontend → FastAPI (SSE streaming) → `MoodleAIAssistantPipeline` → LangGraph state machine → RAG/LLM services → Fireworks.ai LLM

## Critical Architecture Patterns

### Service Dependency Chain (Initialization Order)
```python
# services/rag_service.py - RAG operations (retrieve, enhance_query, retrieve_final, generate)
# services/graph_service.py - LangGraph workflow builder (depends on RAGService)
# services/document_service.py - Document loading/splitting
# services/annotation_service.py - Video annotation database access
# pipeline.py - MoodleAIAssistantPipeline (orchestrates all)
```

Key: **Services initialize in dependency order**. `AnnotationService` and `RAGService` are injected into constructors.

### Multi-Step Conversation Graph (NOT simple retrieve→generate)
**Current sequence**: `retrieve` → `enhance_query` → `retrieve_final` → `generate`
- Each step receives/modifies `ConversationState` (messages, context, video_metadata, enhanced_query)
- All 4 RAG methods defined in `rag_service.py` lines 201-417
- Graph compiled with `MemorySaver` checkpointer (thread_id: hardcoded `"abc123"`)
- Nodes created dynamically via `_create_runnable()` wrapper to ensure unique names

### Document Loading & Persistence
- **Auto-load on startup**: `documents/` folder (lowercase!) triggers RAG mode
- **Supported types**: `.pdf`, `.txt`, `.md` (defined in `AppConfig.supported_file_types`)
- **Splitter**: `CharacterTextSplitter` (NOT recursive) - chunk splitting controlled in settings
- **Vector store**: ChromaDB persists to `./chroma_langchain_db/` with collection name configurable
- **Similarity search**: Uses MMR (max marginal relevance) with k=15 (see `RAGConfig.similarity_search_k`)

### Annotation Sync Feature
- **SQLite database**: `chroma_langchain_db/elicitations_db/annotations.db` (if exists)
- **Auto-sync on startup**: `pipeline._auto_sync_annotations()` syncs completed annotations to vector store
- **Video metadata streaming**: Annotations include `video_id`, `start_time`, `end_time` for synchronized playback
- **API endpoint**: `POST /api/annotations/sync` with options for extended transcripts

### Prompt Template & Language
- **Custom prompt** in `rag_service.py` lines 49-54: French-language apprenticeship learning context
- **Expected variables**: `{history}`, `{context}`, `{query}` (NOT `{question}`)
- **Model**: Fireworks API `llama-v3p3-70b-instruct` with temp=1.0, max_tokens=1024

## Development Workflows

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt  # or conda env create -f environment.yml

# Set environment variables
export FIREWORKS_API_KEY=<your_key>
export LANGCHAIN_API_KEY=<optional_for_tracing>

# Run server
python main.py  # Starts on http://127.0.0.1:8000
# API docs: http://127.0.0.1:8000/docs
```

### Testing & Debugging
- **Health check**: `GET /api/health` returns `{status, timestamp}`
- **System status**: `GET /api/status` returns `{mode, documents_folder_exists, vector_store_count}`
- **Streaming chat**: `POST /api/chat` with `{message, conversation_thread_id}` returns SSE stream
- **Annotation stats**: `GET /api/annotations/stats` (if database exists)
- **Logs**: Console output via `setup_logging()` in `config/settings.py` (format: timestamp, level, module, message)

### Adding Documents at Runtime
```python
pipeline.load_documents(file_paths)  # Returns DataFrame with metadata
pipeline.clear_knowledge_base()       # Remove all vectors
```

### Critical File Editing Points
- **Adding RAG workflow steps**: Implement new method in `rag_service.py`, add to functions list in `pipeline._build_conversation_graph()`
- **Modifying prompt**: Edit template in `rag_service.py` __init__ (ensure `{history}`, `{context}`, `{query}` keys)
- **Changing embeddings model**: Edit `RAGConfig.embedding_model` in `config/settings.py`
- **Configuring LLM**: Edit `RAGConfig.llm_model_url`, temperatures, penalties in `config/settings.py`

## SSE Streaming Response Contract

**Endpoint**: `POST /api/chat`  
**Request**: `{"message": "user query", "conversation_thread_id": "unique_id"}`  
**Response**: SSE stream of JSON lines (delimited by `\n`):
```json
{"event": "video_metadata", "data": {video_id, filename, filepath, start_time, end_time, ...}}
{"content": [{content: "...", type: "ai|human", id: "..."}, ...], "documents": [{id, page_content, metadata}, ...]}
{"content": "[DONE]"}
```

**Key**: Client must listen for `"[DONE]"` signal; stream mode hardcoded to `"updates"`.

## Project-Specific Gotchas

1. **Folder naming**: Code checks `documents/` lowercase. `Documents/` with capital D won't auto-load.
2. **Thread ID hardcoded**: Test ID `"abc123"` in `pipeline.py` line 22 — needs parameterization for multi-user.
3. **CORS wildcard**: `server.py` allows `http://localhost:8080` only; add Moodle domain before production.
4. **Graph function requirements**: Functions passed to `build_conversation_graph()` MUST exist in `RAGService` or ValueError raised.
5. **Annotation database optional**: If `/elicitations_db/annotations.db` missing, sync features silently skip (no error).
6. **Error handling silent**: Services log errors but may continue with partial state; check console logs.

## File Reference Map

| File | Purpose | Key Classes/Functions |
|------|---------|-----|
| `config/settings.py` | Config management, logging | `RAGConfig`, `AppConfig`, `ConfigurationManager`, `setup_logging()` |
| `services/rag_service.py` | RAG operations | `RAGService` with `retrieve()`, `enhance_query()`, `retrieve_final()`, `generate()` |
| `services/graph_service.py` | LangGraph compilation | `ConversationGraphService.build_conversation_graph()` |
| `services/document_service.py` | Document loading | `DocumentProcessingService.load_and_split_documents()` |
| `services/annotation_service.py` | Annotation DB access | `AnnotationService.get_completed_annotations()` |
| `pipeline.py` | Service orchestration | `MoodleAIAssistantPipeline` (entry point) |
| `api/routes.py` | FastAPI endpoints | `POST /api/chat`, `/api/status`, `/api/annotations/sync` |
| `core/types.py` | Shared types | `ConversationState` (LangGraph state schema) |

## External Dependencies

- **LLM**: Fireworks.ai (llama-v3p3-70b-instruct)
- **Embeddings**: HuggingFace (sentence-transformers/all-mpnet-base-v2)
- **Vector DB**: ChromaDB (local persistence)
- **Tracing** (optional): LangSmith (requires LANGCHAIN_API_KEY)
