"""RAG service for document retrieval and generation."""

import os
import logging
from typing import List, Dict, Any, Union, Optional
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain import hub

from config.settings import ConfigurationManager
from core.types import ConversationState


logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG (Retrieval Augmented Generation) operations."""

    def __init__(self, config_manager: ConfigurationManager, use_hub_template: bool = False):
        self.config_manager = config_manager
        self.config = self.config_manager.get_config().rag
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._initialize_vector_store()
        self.llm = self._initialize_llm()
        if use_hub_template:
            self.prompt_template = self._load_prompt_template()
        else:
            self.prompt_template = PromptTemplate(
                input_variables=["history", "context", "question"],
                template="You are helping apprentices in arts and crafts to learn how to perform techniques and gain skills and insight into different domains. "\
                "\n\nYou will use the following discussion history with your apprentice here " \
                "\n\n<history>{history}</history>\n\n and this context " \
                "\n\n<context>{context}</context>\n\n to answer the following query " \
                "\n\n<query>{query}</query>." \
                "When you do not have enough context to help you answer the query, just respond 'I do not know'",
            )

        logger.info(
            f"RAG service initialized with collection '{self.config.collection_name}'"
        )

    def _load_prompt_template(self) -> Optional[PromptTemplate]:
        """Load prompt template from LangChain Hub."""
        raise NotImplementedError
        try:
            self.prompt_template = hub.pull(self.config.prompt_url, include_model=True)
            logger.info(f"Prompt template loaded from {self.config.prompt_url}")
            return self.prompt_template

        except Exception as e:
            logger.error(f"Failed to load prompt template: {str(e)}")
            self.prompt_template = PromptTemplate.from_template(
                "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            )
            logger.info("Using fallback prompt template")
            return self.prompt_template

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
                self.config.llm_model_url,
                model_provider=self.config.llm_provider,
                streaming=True,
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

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """Searches the vector store for documents similar to the provided `query` string.
        Args:
            query (str): The search query string to find similar documents for.
            k (Optional[int], optional): Maximum number of results to return.
            score_thresh (float, optional): Minimum decay score threshold for filtering results.
                Defaults to 0.6.
            decay_factor (float, optional): Factor used in exponential decay score calculation.
                Defaults to 2.
        Returns:
            List[tuple[Document, float]]: List of tuples containing Document objects and
                their original similarity scores, filtered by decay threshold and deduplicated
                by source.
        Raises:
            Exception: Logs error and returns empty list if similarity search fails.
        Note:
            - Decay score is calculated as: 100 * exp(score/decay_factor)
            - Documents are deduplicated based on their metadata source field
            - Original scores are returned, not decay scores
        """
        try:
            k = k or self.config.similarity_search_k
            seen_docs = set()
            unique_results = []
            results = self.vector_store.max_marginal_relevance_search(query, k=k)
            for doc in results:
                logger.info(f"Document {doc.metadata.get("source", "")}")
                doc_content = str(doc.metadata.get("source"))
                if doc_content not in seen_docs:
                    seen_docs.add(doc_content)
                    unique_results.append(doc)

            logger.info(f"Similarity search returned {len(unique_results)} results")
            return unique_results

        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []

    def retrieve(self, state: ConversationState) -> Dict[str, Any]:
        """Retrieve relevant documents for a given state."""
        # Check if we have any documents in the vector store
        vector_data = self.get_vector_store_data()
        has_documents = bool(vector_data.get("ids"))

        if has_documents:
            retrieved_docs = self.similarity_search(
                str(state.get("messages")[-1].content)
            )
            if not retrieved_docs:
                logger.info("No relevant documents found for the query")
                return {"context": []}
            else:
                logger.info(f"Retrieved {len(retrieved_docs)} documents for the query")
                return {"context": retrieved_docs}
        else:
            logger.info(
                "No documents in vector store - switching to pure generation mode"
            )
            return {"context": []}

    def generate(self, state: ConversationState) -> Dict[str, List[BaseMessage]]:
        """Generate response using retrieved context or pure generation."""
        if not self.llm:
            raise ValueError("No LLM available. Please check LLM initialization.")

        try:
            logger.info(f"DEBUG - State at node generate: {state}")
            context_docs = state.get("context", [])
            filled_prompt = None

            context_texts = (
                "\n\n".join([doc.page_content for doc in context_docs])
                if context_docs
                else "No relevant documents found."
            )

            if self.prompt_template:
                filled_prompt = self.prompt_template.invoke(
                    {   
                        "history": state.get("messages")[:-1],
                        "query": str(state.get("messages")[-1].content),
                        "context": context_texts,
                    }
                )

            if filled_prompt:
                response = self.llm.invoke(filled_prompt)
                return {"messages": [response]}
            else:
                # Fallback: direct LLM invocation without prompt template
                logger.warning(
                    "No prompt template available, using direct LLM invocation"
                )
                fallback_prompt = f"Question: {str(state.get('messages'))}\nContext: {context_texts}\nAnswer:"
                response = self.llm.invoke(fallback_prompt)
                return {"messages": [response]}

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
