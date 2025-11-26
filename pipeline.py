"""Main application pipeline orchestrating all services."""

import logging
from langchain_core.documents.base import Document
import pandas as pd
from typing import List, Dict, Any, Optional

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from config.settings import ConfigurationManager
from core.types import ConversationState
from services.langchain_service import LangChainService
from services.rag_service import RAGService
from services.document_service import DocumentProcessingService
from services.graph_service import ConversationGraphService
from services.annotation_service import AnnotationService


logger = logging.getLogger(__name__)
test_thread_id = "abc123"
test_config = RunnableConfig({"configurable": {"thread_id": test_thread_id}})


class MoodleAIAssistantPipeline:
    """Main pipeline orchestrating the Moodle AI Assistant services."""

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()

        # Initialize services in dependency order
        self.langchain_service = LangChainService(self.config_manager)
        self.annotation_service = AnnotationService(self.config_manager)
        self.rag_service = RAGService(
            self.config_manager,
            annotation_service=self.annotation_service
        )
        self.document_service = DocumentProcessingService(self.config_manager)
        self.graph_service = ConversationGraphService(self.rag_service)

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
        """Build and compile the conversation graph with query enhancement."""
        try:
            # Updated sequence: retrieve -> enhance_query -> retrieve_final -> generate
            return self.graph_service.build_conversation_graph(
                functions=["retrieve", "enhance_query", "retrieve_final", "generate"]
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
        self, message: str, conversation_thread_id: str
    ) -> Dict[str, Any]:
        """
        Generate complete response without streaming.
        Waits for entire graph to finish, then returns everything at once.
        
        Returns:
        ```
            {
                "message": "AI response text",
                "documents": [...],  # List of retrieved document sources
                "video_metadata": {...} or None
            }
        ```
        """
        try:
            config = RunnableConfig(
                {"configurable": {"thread_id": conversation_thread_id}}
            )

            logger.info(f"Starting generation for message: {message[:20]}...")

            # using ainvoke instead of astream - runs graph to completion
            final_state: ConversationState = await self.conversation_graph.ainvoke(
                {"messages": [message]},
                config=config
            )
            logger.info(f"Graph execution complete. Finale state keys: \n{finale_state.keys()}")

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

            for doc in context_docs:
                source_info = {
                    "source": doc.metadata.get("source", "nan"),
                    "type": doc.metadata.get("type", "nan"),
                    "page_content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
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
