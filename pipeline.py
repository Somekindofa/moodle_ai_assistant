"""Main application pipeline orchestrating all services."""

import logging
import pandas as pd
from typing import List, Dict, Any, AsyncGenerator, Optional
from api.models import ChatMessage

from langchain.schema import HumanMessage, AIMessage
from langgraph.types import StreamMode

from config.settings import ConfigurationManager
from services.langchain_service import LangChainService
from services.rag_service import RAGService
from services.document_service import DocumentProcessingService
from services.graph_service import ConversationGraphService


logger = logging.getLogger(__name__)


class MoodleAIAssistantPipeline:
    """Main pipeline orchestrating the Moodle AI Assistant services."""

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()

        # Initialize services in dependency order
        self.langchain_service = LangChainService(self.config_manager)
        self.rag_service = RAGService(
            self.config_manager, self.langchain_service.get_prompt_template()
        )
        self.document_service = DocumentProcessingService(self.config_manager)
        self.graph_service = ConversationGraphService(self.rag_service)

        # Auto-load documents from Documents folder if it exists
        self._auto_load_documents()

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

    def _build_conversation_graph(self):
        """Build and compile the conversation graph."""
        try:
            return self.graph_service.build_conversation_graph().compile_graph()
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
        user_query: str,
        history: List[ChatMessage],
        stream_mode: StreamMode = "messages",
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response for user query with optional history."""
        try:
            # Convert history to LangChain format
            history_lc = []
            for msg in history:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    role = msg.role
                    content = msg.content
                elif isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content")
                else:
                    logger.warning(f"Skipping unsupported message format: {type(msg)}")
                    continue

                if content is None:
                    content = ""
                elif not isinstance(content, str):
                    content = str(content)

                if role == "user":
                    history_lc.append(HumanMessage(content=content))
                elif role == "assistant":
                    history_lc.append(AIMessage(content=content))

            history_lc.append(HumanMessage(content=user_query))

            # Stream response from conversation graph
            logger.info(f"Generating response for query: {user_query} with history of {len(history_lc)} messages:\n {history_lc}")
            async for chunk, _ in self.conversation_graph.astream(
                {"question": user_query, "history": history}, stream_mode=stream_mode
            ):
                chunk_content = getattr(chunk, "content", str(chunk))
                if chunk_content:
                    yield chunk_content

        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            yield f"Error generating response: {str(e)}"

    def get_current_directory(self) -> str:
        """Get current working directory."""
        return self.rag_service.get_current_directory()

    def get_knowledge_base_status(self) -> pd.DataFrame:
        """Get current knowledge base status as DataFrame."""
        return self._create_knowledge_base_dataframe()
