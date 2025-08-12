"""RAG service for document retrieval and generation."""

import os
import logging
from typing import List, Dict, Any, Union, Optional
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document

from config.settings import ConfigurationManager
from core.types import ConversationState


logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG (Retrieval Augmented Generation) operations."""

    def __init__(self, config_manager: ConfigurationManager, prompt_template=None):
        self.config_manager = config_manager
        self.config = config_manager.get_config().rag
        self.prompt_template = prompt_template

        # Initialize components
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._initialize_vector_store()
        self.llm = self._initialize_llm()

        logger.info(
            f"RAG service initialized with collection '{self.config.collection_name}'"
        )

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize HuggingFace embeddings."""
        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
            logger.info(
                f"Embeddings initialized with model: {self.config.embedding_model}"
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    def _initialize_vector_store(self) -> Chroma:
        """Initialize Chroma vector store."""
        try:
            vector_store = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.config.persist_directory,
            )
            logger.info(f"Vector store initialized at: {self.config.persist_directory}")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def _initialize_llm(self):
        """Initialize chat model."""
        try:
            llm = init_chat_model(
                self.config.llm_model_url, model_provider=self.config.llm_provider
            )
            logger.info(f"LLM initialized: {self.config.llm_model_url}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            return None

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def remove_documents(
        self, file_paths: Union[List[str], Literal["all"]] = "all"
    ) -> None:
        """Remove documents from the vector store."""
        try:
            if file_paths == "all":
                self.vector_store.reset_collection()
                logger.info("Cleared entire vector store collection")
                return

            ids_to_remove = []
            for file_path in file_paths:
                results = self.vector_store.get(where={"source": file_path})
                if results and "ids" in results:
                    ids_to_remove.extend(results["ids"])

            if ids_to_remove:
                self.vector_store.delete(ids=ids_to_remove)
                logger.info(f"Removed {len(ids_to_remove)} documents from vector store")
            else:
                logger.info("No documents found to remove for the specified file paths")

        except Exception as e:
            logger.error(f"Failed to remove documents: {str(e)}")
            raise

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform similarity search with the given query."""
        try:
            k = k or self.config.similarity_search_k
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []

    def retrieve(self, state: ConversationState) -> Dict[str, Any]:
        """Retrieve relevant documents for a given state."""
        # Check if we have any documents in the vector store
        vector_data = self.get_vector_store_data()
        has_documents = bool(vector_data.get("ids"))
        
        if has_documents:
            retrieved_docs = self.similarity_search(state["question"])
            return {"context": retrieved_docs}
        else:
            # No documents available - return empty context for pure generation
            logger.info("No documents in vector store - switching to pure generation mode")
            return {"context": []}

    def generate(self, state: ConversationState) -> Dict[str, Any]:
        """Generate response using retrieved context or pure generation."""
        if not self.llm:
            raise ValueError("No LLM available. Please check LLM initialization.")

        try:
            # Check if we have context (documents)
            has_context = bool(state.get("context"))
            
            if has_context and self.prompt_template:
                # RAG mode: use context with prompt template
                docs_content = "\n\n".join(doc.page_content for doc in state["context"])
                message = self.prompt_template.invoke(
                    {"question": state["question"], "context": docs_content}
                )
            else:
                # Pure generation mode: direct question to LLM
                logger.info("Using pure generation mode - no context available")
                from langchain.schema import HumanMessage
                message = [HumanMessage(content=state["question"])]

            # Generate response
            response = self.llm.invoke(message)

            # Update conversation history
            current_history = state.get("history", [])
            updated_history = current_history + [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": response.content},
            ]

            return {"answer": response.content, "history": updated_history}

        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise

    def get_vector_store_data(self) -> Dict[str, Any]:
        """Get current vector store data."""
        try:
            return self.vector_store.get()
        except Exception as e:
            logger.error(f"Failed to get vector store data: {str(e)}")
            return {"ids": [], "metadatas": []}

    def get_current_directory(self) -> str:
        """Get current working directory."""
        return os.getcwd()
