# Moodle AI Assistant - AI Agent Instructions

## Project Overview

A RAG-powered FastAPI backend that streams LLM responses to a Moodle JavaScript frontend. Architecture follows a **modular service-oriented pattern** with dependency injection.

**Core Flow**: JavaScript Frontend → FastAPI (streaming SSE) → Pipeline → RAG Service → LangGraph → Fireworks.ai LLM

## Architecture Components

### Service Layer (`services/`)
- **`rag_service.py`**: ChromaDB vector store + HuggingFace embeddings + Fireworks LLM. Key: `similarity_search()` uses MMR (max marginal relevance), not cosine similarity
- **`graph_service.py`**: LangGraph conversation flow. Builds state graphs with `retrieve → generate` sequence using `RunnableLambda` wrappers
- **`document_service.py`**: Loads PDFs/TXT/MD using LangChain loaders with `CharacterTextSplitter`
- **`langchain_service.py`**: LangChain initialization and prompt templates
- **`annotation_service.py`**: Manages annotation database and syncs completed annotations to vector store

### Orchestration
- **`pipeline.py`**: `MoodleAIAssistantPipeline` coordinates all services. Auto-loads documents from `documents/` folder and auto-syncs annotations on startup
- **`server.py`**: FastAPI app with CORS middleware. SSE streaming via `api/routes.py`

### Configuration
- **`config/settings.py`**: Centralized config using dataclasses (`RAGConfig`, `AppConfig`). Loads `.env` with `python-dotenv`
- Required env vars: `FIREWORKS_API_KEY`, `LANGCHAIN_API_KEY`

## Key Patterns & Conventions

### 1. Dual Operating Modes
System auto-detects mode based on vector store state:
- **RAG Mode**: When `documents/` folder exists or vectors loaded
- **Generation Mode**: Direct LLM queries without retrieval
Check `api/routes.py::check_documents_folder()` and `/api/status` endpoint

### 2. Streaming Response Architecture
- Uses **Server-Sent Events (SSE)**, NOT WebSocket
- Stream mode hardcoded to `"updates"` in `api/routes.py::generate_simplified_stream()`
- Pipeline yields `(messages, context, video_metadata)` tuples; API serializes to JSON with `\n` delimiter
- Video metadata sent as first event: `{"event": "video_metadata", "data": video_metadata}`
- Client must watch for `{"content": "[DONE]"}` termination signal

### 3. LangGraph State Management
- State type: `ConversationState` TypedDict in `core/types.py`
- Graph uses `MemorySaver` checkpointer with thread ID `"abc123"` (hardcoded in `pipeline.py`)
- Nodes created dynamically via `_create_runnable()` wrapper ensuring unique names

### 4. Document Processing
- Auto-loads on startup from `documents/` folder (note lowercase, not `Documents/`)
- Supported types: `.pdf`, `.txt`, `.md` (defined in `config/settings.py`)
- Uses `CharacterTextSplitter` by default (NOT recursive splitter)
- Vector store persists to `./chroma_langchain_db/`

### 5. Prompt Template Pattern
Custom template in `rag_service.py` focuses on **French apprenticeship learning** in glassblowing:
```python
"Vous aidez des apprentis dans les arts et l'artisanat à apprendre comment effectuer des techniques..."
```
Falls back to simple template if hub pull fails. **Important**: Template expects `{history}`, `{context}`, `{query}` variables. Always respond in French.

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

### Testing/Debugging
- Basic test suite exists in `tests/` folder with integration tests (e.g., `test_pipeline_integration.py`)
- Check `/api/health` and `/api/status` endpoints for system state
- Use `/docs` Swagger UI for endpoint testing
- Logs to console via `setup_logging()` in `config/settings.py`

### Adding Documents
1. Place files in `documents/` folder (create if missing)
2. Restart server for auto-load OR
3. Use `pipeline.load_documents()` method programmatically

### Dependency Management
- **Development**: Uses conda environment (`environment.yml`)
- **Production**: Uses pip (`requirements.txt`)
- Key versions: LangChain 0.3.x, LangGraph 0.4.x, FastAPI 0.115.x

## Integration Points

### Frontend Contract
**POST /api/chat**:
- Request: `{"message": "user query"}`
- Response: SSE stream with JSON objects containing:
  - `content`: Array of message objects with `{content, type, id}`
  - `documents`: Array of retrieved context docs with `{id, page_content, metadata}`
  - Final message: `{"content": "[DONE]"}`

### External Dependencies
- **Fireworks.ai**: LLM provider (llama-v3p1-70b-instruct)
- **HuggingFace**: Embeddings (sentence-transformers/all-mpnet-base-v2)
- **ChromaDB**: Local vector store persistence
- **LangSmith**: Optional tracing (requires LANGCHAIN_API_KEY)

## Common Pitfalls

1. **Documents folder naming inconsistency**: `server.py` and `api/routes.py` check `"Documents"` (uppercase), but `pipeline.py` uses `"documents"` (lowercase). Use lowercase as per convention.
2. **Thread ID**: Currently hardcoded to `"abc123"` - not production-ready for multi-user scenarios
3. **Stream mode**: Only `"updates"` mode is implemented; other modes will raise ValueError
4. **CORS**: Set to wildcard `"*"` - restrict for production
5. **Similarity search**: Uses MMR with k=15 (configurable in `RAGConfig`), not simple cosine
6. **Error handling**: Services log errors but may silently fail; check logs for processing issues
7. **Language**: All responses must be in French per prompt template

## File Organization

```
config/       - Configuration, logging, env management
services/     - Business logic (RAG, documents, graph, annotations)
api/          - FastAPI routes and Pydantic models
core/         - Shared types
documents/    - Document source folder (auto-loaded)
chroma_langchain_db/  - Persistent vector store
tests/        - Integration tests
```

Legacy code preserved in `app_legacy.py` for reference.
