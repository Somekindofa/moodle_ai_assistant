"""Main application pipeline orchestrating all services."""

import logging
from langchain_core.documents.base import Document
import pandas as pd
from typing import List, Dict, Any, AsyncGenerator, Optional, Union, overload, Literal

from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AnyMessage

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StreamMode

from config.settings import ConfigurationManager
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
        self,
        message: str,
        stream_mode: StreamMode,
    ) -> AsyncGenerator[tuple[List[AnyMessage], List[Document], Optional[Dict[str, Any]]], None]:
        """Generate streaming response for user query with optional history and video metadata."""
        try:
            if stream_mode == "updates":
                accumulated_context = []  # Initialize to avoid unbound variable
                accumulated_video_metadata = None  # Track video metadata
                
                async for update in self.conversation_graph.astream(
                    {"messages": [message]}, stream_mode=stream_mode, config=test_config
                ):
                    for node_name, node_output in update.items():
                        # Initial retrieval node
                        if (
                            node_name == "retrieve_runnable"
                            and "context" in node_output
                        ):
                            logger.info(f"Initial retrieval: {len(node_output.get('context', []))} docs")
                        
                        # Query enhancement node
                        elif (
                            node_name == "enhance_query_runnable"
                            and "enhanced_query" in node_output
                        ):
                            logger.info(f"Enhanced query: {node_output.get('enhanced_query', 'N/A')}")
                        
                        # Final retrieval node (with video metadata)
                        elif (
                            node_name == "retrieve_final_runnable"
                            and "context" in node_output
                        ):
                            accumulated_context: List[Document] = node_output.get("context", [])
                            accumulated_video_metadata = node_output.get("video_metadata", None)
                            logger.info(f"Final retrieval: {len(accumulated_context)} docs, video_metadata: {accumulated_video_metadata is not None}")

                        # Generation node
                        elif (
                            node_name == "generate_runnable"
                            and "messages" in node_output
                        ):
                            messages: List[AnyMessage] = node_output.get("messages", [])
                            if messages and accumulated_context:
                                yield (messages, accumulated_context, accumulated_video_metadata)

            else:
                logger.warning(
                    f"Unsupported stream_mode '{stream_mode}'. "
                    f"Currently supported modes: 'messages', 'values'. "
                    f"Please update the pipeline to handle this mode or use a supported one."
                )
                raise ValueError(
                    f"Pipeline configuration error: stream_mode '{stream_mode}' is not implemented. "
                    f"Supported modes: ['messages', 'values']"
                )

        except Exception as e:
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            error_message = AIMessage(content=f"Error generating response: {str(e)}")
            yield ([error_message], [], None)

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
