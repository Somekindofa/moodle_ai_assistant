# Moodle AI Assistant – Copilot Instructions

## Big picture
- FastAPI backend with modular services and LangGraph orchestration. Core flow: FastAPI → `MoodleAIAssistantPipeline` → LangGraph (`retrieve` → `enhance_query` → `retrieve_final` → `generate`) → Fireworks LLM via LangChain.
- RAG uses ChromaDB + HuggingFace embeddings. `RAGService.similarity_search()` uses MMR, not cosine.

## Key components (files to read first)
- Orchestration: `MoodleAIAssistantPipeline` in [pipeline.py](../pipeline.py)
- HTTP API: routes and streaming video endpoint in [api/routes.py](../api/routes.py)
- RAG + prompt template: [services/rag_service.py](../services/rag_service.py)
- Graph assembly: [services/graph_service.py](../services/graph_service.py)
- Shared state: `ConversationState` in [core/types.py](../core/types.py)

## Project-specific conventions
- Document auto-load from lowercase `documents/` only; `check_documents_folder()` should match this.
- Query enhancement is enabled (initial retrieval → enhanced query → final retrieval). Avoid reintroducing HyDE nodes unless explicitly requested.
- Prompt template expects `{history}`, `{context}`, `{query}` and responds in French.
- Vector store persists at `./chroma_langchain_db/`.

## API behavior and integration points
- POST `/api/chat` is non-streaming; returns `messages`, `documents`, and optional `video_metadata`.
- GET `/video/stream/{video_id}` supports HTTP range requests for video seeking.
- `/api/status` indicates RAG vs generation mode based on vector store/documents state.

## External dependencies
- Fireworks.ai LLM (model URL in `RAGConfig`), HuggingFace embeddings, ChromaDB.
- Required env vars: `FIREWORKS_API_KEY`; `LANGCHAIN_API_KEY` optional for tracing.

## Developer workflows
- Install deps with `pip install -r requirements.txt` or conda env from `environment.yml`.
- Run server via `python main.py` (FastAPI on `http://127.0.0.1:8000`).
- Debug via `/docs`, `/api/health`, `/api/status`.

## Pitfalls
- Keep `documents/` lowercase consistent across services.
- MMR retrieval default `k=15` lives in `RAGConfig`.
- Some routes rely on `StreamingResponse` for video only; chat is non-streaming.
