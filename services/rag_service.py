"""RAG service for document retrieval and generation."""

import os
import logging
from typing import List, Dict, Any, Union, Optional
from typing_extensions import Literal
from datetime import datetime
import json

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

    def __init__(
        self,
        config_manager: ConfigurationManager,
        use_hub_template: bool = False,
        annotation_service: Optional[Any] = None,
    ):
        self.config_manager = config_manager
        self.config = self.config_manager.get_config().rag
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._initialize_vector_store()
        self.llm = self._initialize_llm()
        self.annotation_service = annotation_service  # Optional dependency

        if use_hub_template:
            self.prompt_template = self._load_prompt_template()
        else:
            self.prompt_template = PromptTemplate(
                input_variables=["history", "context", "question"],
                template="Vous aidez des apprentis dans les arts et l'artisanat à apprendre comment effectuer des techniques et acquérir des compétences et des connaissances dans le domaine de la soufflerie de verre. "
                "\n\nVous utiliserez l'historique de discussion suivant avec votre apprenti ici "
                "\n\n<history>\n{history}\n</history>\n\n et ce contexte "
                "\n\n<context>\n{context}\n</context>\n\n pour répondre à la requête suivante "
                "\n\n<query>\n{query}\n</query>."
                "\n\nFournissez une réponse en français détaillée et instructive sur la manière de se positionner, les outils que l'on utilise, les erreurs communes."
                "\n\nSi le contexte ne contient pas d'informations pertinentes, répondez honnêtement que vous ne savez pas."
                "\n\nRépondez toujours en français."
                "\n\nUtilise le markdown pour formater ta réponse, en utilisant des listes à puces, des tableaux et des sections si nécessaire.",
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

    def vs_add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def vs_remove_documents(
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
        """Searches the vector store for documents similar to the provided `query` string.
        Args:
            query (str): The search query string to find similar documents for.
            k (Optional[int], optional): Maximum number of results to return.
            score_thresh (float, optional): Minimum decay score threshold for filtering results.
                Defaults to 0.6.
            decay_factor (float, optional): Factor used in exponential decay score calculation.
                Defaults to 2.
        Returns:
            List[Document]: List of Document objects filtered by decay threshold and deduplicated
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
                doc_content = str(doc.metadata.get("source"))
                if doc_content not in seen_docs:
                    seen_docs.add(doc_content)
                    unique_results.append(doc)

            logger.info(f"Similarity search returned {len(unique_results)} results")
            return unique_results

        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []

    def hyde_generate(self, state: ConversationState):
        """Generate a Hypothetical document using HyDE method for subsequent similarity search"""
        hyde_prompt: str = """You create """
    
    def retrieve(self, state: ConversationState) -> Dict[str, Any]:
        """Retrieve relevant documents for a given state (initial retrieval for query enhancement)."""
        # Check if we have any documents in the vector store
        vector_data = self.get_vector_store_data()
        has_documents = bool(vector_data.get("ids"))

        if has_documents:
            retrieved_docs = self.similarity_search(
                str(state.get("messages")[-1].content)
            )
            if not retrieved_docs:
                logger.info("No relevant documents found for the query")
                return {"context": [], "video_metadata": None}
            else:
                logger.info(
                    f"Retrieved {len(retrieved_docs)} documents for initial retrieval"
                )

                # Don't extract video metadata yet - that happens in final retrieval
                return {"context": retrieved_docs, "video_metadata": None}
        else:
            logger.info(
                "No documents in vector store - switching to pure generation mode"
            )
            return {"context": [], "video_metadata": None}



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
                        "history": [
                            f"{msg.type}: {msg.content}"
                            for msg in state.get("messages", [])[:-1]
                        ],
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

    def sync_annotations_to_vector_store(
        self,
        use_extended: bool = False,  # Changed default to False - use raw transcripts
        clear_existing: bool = False,
    ) -> int:
        """
        Sync completed annotations from SQLite to ChromaDB.

        Args:
            use_extended: Whether to include extended transcripts (default: False, uses raw transcripts)
            clear_existing: Whether to clear existing annotation documents first

        Returns:
            Number of documents added to vector store
        """
        if not self.annotation_service:
            logger.error("No annotation service available for syncing")
            return 0

        try:
            # Optionally clear existing annotation documents
            if clear_existing:
                self._clear_annotation_documents()

            # Fetch completed annotations
            annotations = self.annotation_service.get_completed_annotations(
                include_extended=use_extended
            )

            if not annotations:
                logger.info("No completed annotations to sync")
                return 0

            # Convert to documents
            all_documents = []
            for annotation in annotations:
                docs = self.annotation_service.annotation_to_documents(
                    annotation, use_extended=use_extended
                )
                all_documents.extend(docs)

            # Add to vector store
            if all_documents:
                self.vs_add_documents(all_documents)
                logger.info(
                    f"Synced {len(all_documents)} annotation documents to vector store"
                )
                return len(all_documents)

            return 0

        except Exception as e:
            logger.error(f"Failed to sync annotations: {str(e)}")
            return 0

    def sync_new_annotations(
        self,
        since_timestamp: datetime,
        use_extended: bool = False,  # Changed default to False - use raw transcripts
    ) -> int:
        """
        Sync only new/updated annotations since a timestamp.

        Args:
            since_timestamp: Only sync annotations updated after this time
            use_extended: Whether to include extended transcripts (default: False, uses raw transcripts)

        Returns:
            Number of documents added
        """
        if not self.annotation_service:
            logger.error("No annotation service available for syncing")
            return 0

        try:
            annotations = self.annotation_service.get_annotations_since(
                since_timestamp, include_extended=use_extended
            )

            if not annotations:
                logger.info(f"No new annotations since {since_timestamp}")
                return 0

            all_documents = []
            for annotation in annotations:
                docs = self.annotation_service.annotation_to_documents(
                    annotation, use_extended=use_extended
                )
                all_documents.extend(docs)

            if all_documents:
                self.vs_add_documents(all_documents)
                logger.info(f"Synced {len(all_documents)} new annotation documents")
                return len(all_documents)

            return 0

        except Exception as e:
            logger.error(f"Failed to sync new annotations: {str(e)}")
            return 0

    def _clear_annotation_documents(self) -> None:
        """Remove all annotation-type documents from vector store."""
        try:
            results = self.vector_store.get(where={"type": "video_annotation"})
            if results and "ids" in results and results["ids"]:
                self.vector_store.delete(ids=results["ids"])
                logger.info(
                    f"Cleared {len(results['ids'])} annotation documents from vector store"
                )
        except Exception as e:
            logger.error(f"Failed to clear annotation documents: {str(e)}")

    def get_annotation_documents_count(self) -> int:
        """Get count of annotation documents in vector store."""
        try:
            results = self.vector_store.get(where={"type": "video_annotation"})
            count = len(results.get("ids", []))
            logger.info(f"Found {count} annotation documents in vector store")
            return count
        except Exception as e:
            logger.error(f"Failed to count annotation documents: {str(e)}")
            return 0

    def _extract_video_metadata(
        self, documents: List[Document]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract video metadata from retrieved documents.

        Looks for video annotation documents and extracts video playback information.
        Returns metadata for the first video annotation found.

        Args:
            documents: List of retrieved documents (just Document objects, not tuples)

        Returns:
            Dictionary with video metadata or None if no video annotations found
        """
        import hashlib

        for doc in documents:
            metadata = doc.metadata

            # Check if this is a video annotation document
            if metadata.get("type") == "video_annotation":
                video_filepath = metadata.get("video_filepath")

                if not video_filepath:
                    continue

                # Generate secure video_id from filepath and annotation_id
                video_id_source = (
                    f"{video_filepath}_{metadata.get('annotation_id', '')}"
                )
                video_id = hashlib.md5(video_id_source.encode()).hexdigest()

                video_metadata = {
                    "video_id": video_id,
                    "filename": metadata.get("video_filename", "unknown.mp4"),
                    "filepath": video_filepath,
                    "start_time": float(metadata.get("start_time", 0)),
                    "end_time": float(metadata.get("end_time", 0)),
                    "duration": float(metadata.get("duration", 0)),
                    "video_url": f"/api/video/stream/{video_id}",
                    "annotation_id": metadata.get("annotation_id"),
                    "project_name": metadata.get("project_name"),
                }

                logger.info(
                    f"Extracted video metadata for {video_metadata['filename']}"
                )
                return video_metadata

        logger.info("No video annotations found in retrieved documents")
        return None
