# Moodle AI Assistant – Copilot Instructions

## Big picture
- **Architecture**: FastAPI backend → `MoodleAIAssistantPipeline` → LangGraph with HyDE (Hypothetical Document Embeddings)
- **Core flow**: User query → `generate_hypothetical_document` (LLM creates synthetic expert elicitation) → `retrieve_with_hyde` (search vector DB using synthetic text for better semantic matching) → `generate` (LLM response with context) → FastAPI returns messages + documents + optional video metadata
- **Why HyDE**: Novice queries ("how do I hold my blowpipe?") don't match expert elicitation transcripts linguistically. HyDE generates matching-style synthetic text to bridge this gap.
- **RAG stack**: ChromaDB vector store + HuggingFace multilingual embeddings (MMR search, k=5) + Fireworks LLM

## Key components (read first for understanding flows)
1. **Orchestration**: `MoodleAIAssistantPipeline` in [pipeline.py](../pipeline.py) - initializes services, auto-loads documents, builds/runs LangGraph
2. **LangGraph assembly**: `ConversationGraphService.build_conversation_graph()` in [services/graph_service.py](../services/graph_service.py) - dynamically wires nodes
3. **HyDE generation**: `RAGService.generate_hypothetical_document()` in [services/rag_service.py](../services/rag_service.py) - creates expert-style text from vague queries
4. **HyDE retrieval**: `RAGService.retrieve_with_hyde()` in [services/rag_service.py](../services/rag_service.py) - single-pass MMR search + video metadata extraction
5. **HTTP API**: `ChatRequest → /api/chat` endpoint in [api/routes.py](../api/routes.py) - streaming JSON lines response
6. **State management**: `ConversationState` in [core/types.py](../core/types.py) - tracks messages, context (Documents), hypothetical_document, video_metadata, selected_domain

## Project-specific conventions
- **Document loading**: Auto-loads from `documents/` (lowercase, hardcoded in pipeline.py line 39) on startup if folder exists
- **HyDE prompts**: Must be French (project-specific). See example in `generate_hypothetical_document()` requesting "positionnement précis des mains", "angles d'outils", "sensations physiques"
- **Prompt template**: Expects `{history}`, `{context}`, `{query}` variables (defined in RAGService.__init__); responds in French
- **Vector store location**: `./chroma_langchain_db/` (relative path, persists collection between runs)
- **Retrieval k values**: `retrieve_with_hyde` uses k=5 (NOT k=15) for single-pass precision; legacy methods had k=15 for two-pass
- **Video metadata**: Extracted in `retrieve_with_hyde` via `_extract_video_metadata()` using MD5 hash of filepath+annotation_id

## API behavior
- **POST `/api/chat`**: Takes `ChatRequest(message, conversation_thread_id, selected_domain)`, streams JSON lines: video_metadata event, message event, then `[DONE]`
- **POST `/ingest-annotation`**: Takes `AnnotationIngestRequest`, ingests a single completed annotation directly into the vector store for real-time searchability
- **GET `/video/stream/{video_id}`**: Supports HTTP `Range` headers for seeking; returns 206 Partial Content; validates video_id is MD5 hash format
- **GET `/api/status`**: Returns `mode` ("rag" if docs exist OR vector_store has docs, else "generation")

## Domain selection
- `selected_domain` is an optional field on `ChatRequest` (e.g. "Soufflerie de verre", "Scellerie nautique", "Ganterie")
- When set, it flows through `ConversationState.selected_domain` and is used in `route_query()`, `generate()`, and `direct_generate()` to focus the LLM on a specific craft domain
- Domain hint injected as: `"Vous vous concentrez particulièrement sur le domaine : {domain}."`

## External dependencies
- **LLM**: Fireworks.ai (model `accounts/fireworks/models/qwen3-8b` in RAGConfig)
- **Embeddings**: HuggingFace `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (supports French)
- **Vector DB**: Chroma with persistent SQLite backend at `./chroma_langchain_db/`
- **Required env vars**: `FIREWORKS_API_KEY` (must set); `LANGCHAIN_API_KEY` optional for LangChain tracing
- **Annotation service**: SQLite database (optional) for syncing video transcripts via `sync_annotations_to_vector_store()`

## Developer workflows
- **Setup**: `pip install -r requirements.txt` OR `conda env create -f environment.yml`
- **Run**: `python main.py` (starts FastAPI server on `http://127.0.0.1:8000`)
- **Debug**: Visit `http://127.0.0.1:8000/docs` (auto-generated OpenAPI), check `/api/health`, `/api/status`
- **Test HyDE**: Call `/api/chat` with message like "how do I position my hands?" — see HyDE preview in logs, verify context retrieved

## Common patterns
- **Service initialization order** (pipeline.py __init__): LangChainService → AnnotationService → RAGService → DocumentProcessingService → ConversationGraphService
- **Error handling**: Log with `logger.error()` + traceback (see pipeline.py), raise with meaningful context
- **Document metadata**: `metadata` dict includes `source` (filename), `type` ("video_annotation" or "text"), `page_content` (text chunk)
- **LangGraph nodes**: Add via `ConversationGraphService.build_conversation_graph(functions=[...])` with method names from RAGService

## Legacy code to avoid
- `retrieve()`, `enhance_query()`, `retrieve_final()` methods in RAGService (marked [LEGACY] in comments) — old two-pass retrieval approach replaced by HyDE
- Do NOT use query enhancement for new features; use HyDE generation instead

## Pitfalls
- Document auto-load checks `documents/` (lowercase), not `Documents/` — inconsistent casing breaks loading
- `RAGConfig.similarity_search_k=15` is NOT used by HyDE (uses k=5 hardcoded in `retrieve_with_hyde()`)
- LLM initialization fails silently if `FIREWORKS_API_KEY` not set — check logs carefully
- Video streaming (range requests) only works if file exists on disk; metadata must have valid `video_filepath`
