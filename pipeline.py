"""Main application pipeline orchestrating all services."""

import logging
from langchain_core.documents.base import Document
from langchain_core.messages import AnyMessage
import pandas as pd
from typing import List, Dict, Any, Literal, Optional
from langsmith import traceable

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from config.settings import ConfigurationManager
from core.types import ConversationState
from services.langchain_service import LangChainService
from services.rag_service import RAGService
from services.course_rag_service import CourseRAGService
from services.document_service import DocumentProcessingService
from services.graph_service import ConversationGraphService
from services.annotation_service import AnnotationService
from services.silo_service import SiloService


logger = logging.getLogger(__name__)
test_thread_id = "abc123"
test_config = RunnableConfig({"configurable": {"thread_id": test_thread_id}})
StreamMode = Literal["values", "updates"]


class MoodleAIAssistantPipeline:
    """Main pipeline orchestrating the Moodle AI Assistant services."""

    # Maximum number of candidates passed to the cross-encoder reranker.
    # Keeping this low is critical on 2-core CPUs where bge-reranker-v2-m3
    # takes ~5 s per document pair.
    MAX_RERANK_CANDIDATES = 5

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()

        # Initialize services in dependency order
        self.langchain_service = LangChainService(self.config_manager)
        self.annotation_service = AnnotationService(self.config_manager)
        # RAGService is initialised first so we can share its embeddings model
        self.rag_service = RAGService(
            self.config_manager,
            annotation_service=self.annotation_service,
        )
        # CourseRAGService reuses the same HuggingFace embeddings instance
        self.course_rag_service = CourseRAGService(
            embeddings=self.rag_service.embeddings,
            persist_directory=self.rag_service.config.persist_directory,
        )
        # Inject course_rag_service into rag_service for PRF dual-collection retrieval
        self.rag_service.course_rag_service = self.course_rag_service

        self.document_service = DocumentProcessingService(self.config_manager)
        self.graph_service = ConversationGraphService(self.rag_service)
        self.silo_service = SiloService()

        # Auto-load documents from Documents folder if it exists
        self._auto_load_documents()

        # Auto-sync annotations from database
        self._auto_sync_annotations()

        # Build and compile conversation graph
        self.conversation_graph = self._build_conversation_graph()

        logger.info("Moodle AI Assistant Pipeline initialized successfully")

    def _auto_load_documents(self):
        """Automatically load documents from Documents folder if it exists."""
        import os
        import glob

        documents_folder = "documents"
        if os.path.exists(documents_folder) and os.path.isdir(documents_folder):
            logger.info("Documents folder found - loading documents...")

            # Find all supported files in documents folder
            supported_extensions = self.config_manager.get_config().supported_file_types
            all_files = []

            for ext in supported_extensions:
                pattern = os.path.join(documents_folder, "**", f"*{ext}")
                files = glob.glob(pattern, recursive=True)
                normalized_files = [os.path.normpath(f) for f in files]
                all_files.extend(normalized_files)

            if all_files:
                logger.info(
                    f"Found {len(all_files)} supported files in documents folder"
                )
                self.load_documents(all_files)
            else:
                logger.info("No supported files found in documents folder")
        else:
            logger.info("No documents folder found - will use pure generation mode")

    def _auto_sync_annotations(self) -> None:
        """Automatically sync annotations from database on startup."""
        try:
            stats = self.annotation_service.get_annotation_stats()
            logger.info(f"Annotation database stats: {stats}")

            if stats.get("completed_extended", 0) > 0:
                count = self.rag_service.sync_annotations_to_vector_store(
                    use_extended=False,  # Use raw transcripts only
                    clear_existing=False
                )
                logger.info(f"Auto-synced {count} annotation documents on startup")
            else:
                logger.info("No completed annotations found for auto-sync")

        except Exception as e:
            logger.warning(f"Auto-sync of annotations failed: {str(e)}")

    def _build_conversation_graph(self) -> CompiledStateGraph:
        """Build and compile the PRF conversation graph with cross-encoder reranking.

        Pipeline:
          retrieve_initial → refine_query_prf → retrieve_final_dual → rerank → generate

        retrieve_initial and retrieve_final_dual cast a wide net from both the
        video annotation collection and per-course collections.  rerank filters
        and re-orders the candidate set using a local multilingual cross-encoder,
        replacing the old L2 distance guardrail with a principled relevance score.
        """
        try:
            return self.graph_service.build_conversation_graph(
                functions=[
                    "retrieve_initial",
                    "refine_query_prf",
                    "retrieve_final_dual",
                    "rerank",
                    "generate",
                ]
            ).compile_graph()
        except Exception as e:
            logger.error(f"Failed to build conversation graph: {str(e)}")
            raise

    def load_documents(self, file_paths: List[str]) -> pd.DataFrame:
        """Load and process documents into the knowledge base."""
        try:
            if not file_paths:
                raise ValueError("No files selected for loading")

            # Filter supported files
            supported_files = self.document_service.filter_supported_files(file_paths)
            if not supported_files:
                logger.warning("No supported files found in the provided list")
                return pd.DataFrame()

            # Load and split documents
            documents = self.document_service.load_and_split_documents(supported_files)

            if documents:
                # Add to RAG service
                self.rag_service.add_documents(documents)

                # Return updated knowledge base view
                return self._create_knowledge_base_dataframe()
            else:
                logger.warning("No documents were successfully processed")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            return pd.DataFrame()

    def clear_knowledge_base(self) -> None:
        """Clear all documents from the knowledge base."""
        try:
            self.rag_service.remove_documents("all")
            logger.info("Knowledge base cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear knowledge base: {str(e)}")

    def _create_knowledge_base_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame representation of the current knowledge base."""
        try:
            vector_data = self.rag_service.get_vector_store_data()

            if not vector_data.get("ids"):
                return pd.DataFrame(columns=["ID", "Title", "Source"])

            df = pd.DataFrame(
                {
                    "ID": vector_data["ids"],
                    "Title": [
                        metadata.get("title", "N/A")
                        for metadata in vector_data.get("metadatas", [])
                    ],
                    "Source": [
                        metadata.get("source", "N/A")
                        for metadata in vector_data.get("metadatas", [])
                    ],
                }
            )
            return df

        except Exception as e:
            logger.error(f"Failed to create knowledge base DataFrame: {str(e)}")
            return pd.DataFrame(columns=["ID", "Title", "Source"])

    async def generate_response(
        self,
        message: str,
        conversation_thread_id: str,
        stream_mode: StreamMode,
        selected_domain: Optional[str] = None,
        course_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate response using the PRF retrieval pipeline.

        Retrieves from video annotation collection and (when course_id is given)
        from the per-course ChromaDB collection.
        """
        try:
            config = RunnableConfig({"configurable": {"thread_id": conversation_thread_id}})

            logger.info(f"Starting generation for message: {message[:20]}...")

            # using ainvoke instead of astream - runs graph to completion
            final_state = await self.conversation_graph.ainvoke(
                {
                    "messages": [message],
                    "selected_domain": selected_domain,
                    "course_id": course_id,
                },
                config=config
            )
            logger.info(f"Graph execution complete. Final state keys: \n{final_state.keys()}")

            # Extract AI message from final state
            messages = final_state.get("messages", [])
            ai_message = None

            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "ai":
                    ai_message = msg.content
                    break

            if not ai_message:
                logger.warning("Could not generate a response.")
                ai_message = "Could not generate a response."

            context_docs: List[Document] = final_state.get("context", [])
            document_sources = []

            _seen_module_ids: set = set()
            for doc in context_docs:
                mid = doc.metadata.get("module_id", "")
                if mid:
                    if mid in _seen_module_ids:
                        continue
                    _seen_module_ids.add(mid)
                source_info = {
                    "source": doc.metadata.get("source", "nan"),
                    "type": doc.metadata.get("type", "nan"),
                    "page_content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "module_id":   mid,
                    "module_type": doc.metadata.get("module_type", ""),
                    "module_name": doc.metadata.get("module_name", ""),
                    "course_id":   doc.metadata.get("course_id", ""),
                    "heading_path": doc.metadata.get("heading_path", ""),
                    "section_name": doc.metadata.get("section_name", ""),
                }
                document_sources.append(source_info)
            logger.info(f"Extracted {len(document_sources)} document sources")

            video_metadata = final_state.get("video_metadata")

            if video_metadata:
                logger.info(
                    f"Video metadata found: {video_metadata.get('filename', 'unknown_video')}"
                )

            return {
                "messages": ai_message,
                "documents": document_sources,
                "video_metadata": video_metadata
            }

        except Exception as e:
            import traceback
            logger.error(f"Batch generation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _generate_conversation_title(self, message: str) -> str:
        """Call the LLM with max_tokens=120 to produce a short conversation title.

        Uses .bind() to override max_tokens without mutating the shared LLM instance.
        Falls back to a truncated version of the message if the LLM call fails.
        """
        try:
            prompt = (
                "Génère un titre court (5 à 8 mots) qui résume la question suivante "
                "d'un apprenti. Réponds UNIQUEMENT avec le titre, sans guillemets ni "
                f"ponctuation finale.\n\nQuestion : {message}"
            )
            title_llm = self.rag_service.llm.bind(max_tokens=120)
            response = await title_llm.ainvoke(prompt)
            title = response.content.strip().strip('"').strip("'")
            return title if title else message[:60]
        except Exception as e:
            logger.warning(f"Title generation failed: {e}")
            return message[:60]

    async def _classify_in_domain(self, message: str) -> bool:
        """Return True if message is in-domain (craft trades / apprenticeship).

        Uses a minimal LLM call (max_tokens=10, temperature=0) to classify.
        Fails open on any exception so legitimate questions are never blocked.
        Note: a crafted message can manipulate the classifier to return True (fail-open path);
        the main RAG system-prompt guardrail remains the last line of defence for such cases.
        """
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            classifier_llm = self.rag_service.llm.bind(max_tokens=10, temperature=0)
            system = SystemMessage(content=(
                "Tu es un classifieur de sujets. Réponds par un seul mot : "
                "OUI si la question concerne les arts et métiers, l'apprentissage, "
                "les techniques artisanales (soufflage de verre, ganterie, menuiserie, "
                "sellerie, etc.). Réponds NON pour tout le reste (politique, actualité, "
                "géographie, célébrités, etc.). Réponds uniquement OUI ou NON."
            ))
            human = HumanMessage(content=message)
            response = await classifier_llm.ainvoke([system, human])
            import re
            first_word = re.sub(r"[^A-Z]", "", response.content.strip().upper().split()[0]) if response.content.strip() else ""
            return first_word in {"OUI", "YES"}
        except Exception as e:
            logger.warning(f"Input classifier failed (fail-open): {e}")
            return True

    @traceable(name="stream_response", run_type="chain")
    async def stream_response(
        self,
        message: str,
        conversation_thread_id: str,
        selected_domain: Optional[str] = None,
        course_id: Optional[str] = None,
        is_first_message: bool = False,
        disable_rerank: bool = False,
        user_id: Optional[int] = None,     # NEW
    ):
        """Async generator that streams the full response as JSON-lines events.

        Runs the three PRF retrieval nodes synchronously (they are CPU-bound
        vector-search + one LLM call each), then streams the final generation
        token-by-token so the client sees output immediately.

        Yields JSON-line strings:
          {"event": "status", "data": "..."}               — pipeline step hints
          {"event": "conversation_title", "data": "..."}  — only when is_first_message=True
          {"event": "video_metadata", "data": {...}}       — optional, before tokens
          {"event": "token", "data": "<text>"}             — one per LLM token
          {"event": "documents", "data": [...]}            — after all tokens
          {"content": "[DONE]"}                            — terminal marker
        """
        import asyncio
        import json

        try:
            from langchain_core.messages import HumanMessage

            # Resolve silo scope — inner catch yields error JSON-line and returns early on DB failure
            user_cohort_ids: list[int] = []
            enrolled_course_ids: list[str] = []
            if user_id and user_id > 0:
                try:
                    user_cohort_ids = self.silo_service.get_allowed_cohorts(user_id)
                    enrolled_course_ids = self.silo_service.get_enrolled_course_ids(user_id)
                except Exception as e:
                    logger.error(f"SiloService failed for user {user_id}: {e}")
                    yield json.dumps({"event": "error", "data": "Service temporarily unavailable"}) + "\n"
                    yield json.dumps({"content": "[DONE]"}) + "\n"
                    return

            # --- Pre-LLM topic classifier ---
            is_in_domain = await self._classify_in_domain(message)
            if not is_in_domain:
                yield json.dumps({"event": "status", "data": "Vérification de la question…"}) + "\n"
                yield json.dumps({"event": "token", "data": (
                    "Je n'ai pas trouvé d'information pertinente dans le corpus "
                    "pour répondre à cette question. Veuillez poser une question "
                    "sur les arts et métiers ou consulter votre formateur."
                )}) + "\n"
                yield json.dumps({"content": "[DONE]"}) + "\n"
                return

            # Generate and emit the conversation title before the PRF pipeline
            # so the client can update the sidebar title immediately.
            if is_first_message:
                title = await self._generate_conversation_title(message)
                yield json.dumps({"event": "conversation_title", "data": title}) + "\n"

            # Build the initial state dict (MessagesState expects message objects)
            state: Dict[str, Any] = {
                "messages": [HumanMessage(content=message)],
                "selected_domain": selected_domain,
                "course_id": course_id,
                "context": [],
                "video_metadata": None,
                "refined_query": None,
                "hypothetical_document": None,
                "enhanced_query": None,
                "query_variants": [],
                "route": None,
                "user_cohort_ids": user_cohort_ids,           # NEW
                "enrolled_course_ids": enrolled_course_ids,   # NEW
            }

            # --- PRF step 1: initial retrieval ---
            yield json.dumps({"event": "status", "data": "Recherche dans la base de connaissances…"}) + "\n"
            result = await asyncio.to_thread(self.rag_service.retrieve_initial, state)
            state.update(result)

            # --- PRF step 2: corpus-grounded query refinement ---
            yield json.dumps({"event": "status", "data": "Reformulation de la question…"}) + "\n"
            result = await asyncio.to_thread(self.rag_service.refine_query_prf, state)
            state.update(result)

            # --- PRF step 3: final retrieval with refined query ---
            yield json.dumps({"event": "status", "data": "Récupération des sources pertinentes…"}) + "\n"
            result = await asyncio.to_thread(self.rag_service.retrieve_final_dual, state)
            state.update(result)

            # --- PRF step 4: cross-encoder reranking and relevance filtering ---
            if not disable_rerank:
                yield json.dumps({"event": "status", "data": "Classement des résultats…"}) + "\n"
                # Cap candidates before reranking — bge-reranker-v2-m3 takes
                # ~5 s per pair on a 2-core CPU, so keep the list tight.
                ctx = state.get("context", [])
                if len(ctx) > self.MAX_RERANK_CANDIDATES:
                    state["context"] = ctx[: self.MAX_RERANK_CANDIDATES]
                # Run synchronous cross-encoder inference in a thread so the
                # event loop remains responsive during the ~20-30 s prediction.
                result = await asyncio.to_thread(self.rag_service.rerank, state)
                state.update(result)

            # Re-emit video metadata after reranking (top doc may have changed).
            if state.get("video_metadata"):
                yield json.dumps({"event": "video_metadata", "data": state["video_metadata"]}) + "\n"

            # --- Stream the generate step token by token ---
            yield json.dumps({"event": "status", "data": "Génération de la réponse…"}) + "\n"
            async for token in self.rag_service.stream_generate(state):
                yield json.dumps({"event": "token", "data": token}) + "\n"

            # --- Emit document sources after generation completes ---
            context_docs: List[Document] = state.get("context", [])
            _seen_module_ids: set = set()
            document_sources = []
            for doc in context_docs:
                mid = doc.metadata.get("module_id", "")
                # Deduplicate course-content docs by module_id: multiple chunks
                # from the same Moodle module all resolve to the same URL on
                # click, so only the first chunk is needed as a representative.
                if mid:
                    if mid in _seen_module_ids:
                        continue
                    _seen_module_ids.add(mid)
                document_sources.append({
                    "source": doc.metadata.get("source", "nan"),
                    "type": doc.metadata.get("type", "nan"),
                    "page_content_preview": (
                        doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content
                    ),
                    "module_id":    mid,
                    "module_type":  doc.metadata.get("module_type", ""),
                    "module_name":  doc.metadata.get("module_name", ""),
                    "course_id":    doc.metadata.get("course_id", ""),
                    "heading_path": doc.metadata.get("heading_path", ""),
                    "section_name": doc.metadata.get("section_name", ""),
                })
            yield json.dumps({"event": "documents", "data": document_sources}) + "\n"

            yield json.dumps({"content": "[DONE]"}) + "\n"

        except GeneratorExit:
            logger.info("Client disconnected during streaming")
            raise
        except Exception as e:
            import traceback
            logger.error(f"stream_response failed: {e}")
            logger.error(traceback.format_exc())
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    def get_current_directory(self) -> str:
        """Get current working directory."""
        return self.rag_service.get_current_directory()

    def get_knowledge_base_status(self) -> pd.DataFrame:
        """Get current knowledge base status as DataFrame."""
        return self._create_knowledge_base_dataframe()

    def sync_annotations(
        self,
        use_extended: bool = False,  # Changed default to False - use raw transcripts
        clear_existing: bool = False
    ) -> int:
        """Manually trigger annotation sync."""
        return self.rag_service.sync_annotations_to_vector_store(
            use_extended=use_extended,
            clear_existing=clear_existing
        )

    def get_annotation_stats(self) -> Dict[str, Any]:
        """Get annotation statistics from both database and vector store."""
        db_stats = self.annotation_service.get_annotation_stats()
        vector_count = self.rag_service.get_annotation_documents_count()

        return {
            **db_stats,
            "vector_store_annotations": vector_count
        }
