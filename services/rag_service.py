"""RAG service for document retrieval and generation."""

import os
import logging
from typing import List, Dict, Any, Union, Optional
from annotated_types import doc
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
                input_variables=["history", "context", "query", "specific_domain"],
                template="Vous aidez des apprentis dans les arts et l'artisanat à apprendre comment effectuer des techniques et acquérir des compétences et des connaissances dans différents domaines."
                "{specific_domain}"
                "\n\nVous utiliserez l'historique de discussion suivant avec votre apprenti ici "
                "\n\n<history>\n{history}\n</history>\n\n et ce contexte "
                "\n\n<context>\n{context}\n</context>\n\n pour répondre à la requête suivante "
                "\n\n<query>\n{query}\n</query>.",
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
                logger.info(f"Document {doc.metadata.get('source', '')}")
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
        """Retrieve relevant documents for a given state (initial retrieval for query enhancement)."""
        # Check if we have any documents in the vector store
        vector_data = self.get_vector_store_data()
        has_documents = bool(vector_data.get("ids"))
        logger.info(f"State at retrieve: {state}")

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

    def enhance_query(self, state: ConversationState) -> Dict[str, Any]:
        """
        Enhance user query using LLM based on initially retrieved documents.

        This node receives:
        - Original user query from messages[-1]
        - Top 3 retrieved documents from context

        It uses the LLM to enhance the query by incorporating relevant aspects
        from the retrieved documents while preserving the original query's intent.

        Returns:
        - enhanced_query: The improved query string for final retrieval
        """
        if not self.llm:
            logger.warning(
                "No LLM available for query enhancement, using original query"
            )
            return {"enhanced_query": str(state.get("messages")[-1].content)}

        try:
            original_query = str(state.get("messages")[-1].content)
            context_docs = state.get("context", [])

            # If no context was retrieved, skip enhancement
            if not context_docs:
                logger.info("No context available, skipping query enhancement")
                return {"enhanced_query": original_query}

            # Prepare context snippets from retrieved documents
            context_snippets = []
            for i, doc in enumerate(context_docs[:3], 1):  # Use top 3 docs
                # Extract key information from metadata
                doc_type = doc.metadata.get("transcript_type", "text")
                source = doc.metadata.get("source", "unknown")

                # Truncate content to avoid overwhelming the LLM
                content_preview = (
                    doc.page_content[:300] + "..."
                    if len(doc.page_content) > 300
                    else doc.page_content
                )

                context_snippets.append(
                    f"Document {i} ({doc_type} from {source}):\n{content_preview}"
                )

            context_text = "\n\n".join(context_snippets)

            # Create enhancement prompt
            enhancement_prompt = f"""You are a query enhancement assistant. Your task is to improve a user's query by incorporating relevant aspects from retrieved documents while preserving the original query's meaning and intent.

Original User Query:
{original_query}

Retrieved Context Snippets (for reference only):
{context_text}

Instructions:
1. Analyze the original query and identify its core intent
2. Review the retrieved context snippets for relevant terminology, concepts, or domain-specific language
3. Enhance the query by:
   - Adding relevant technical terms or domain vocabulary from the context
   - Incorporating synonyms or related concepts that appear in the retrieved documents
   - Maintaining the original query's semantic meaning and user intent
4. Keep the enhanced query concise (1-3 sentences maximum)
5. Do NOT change the fundamental question being asked
6. Do NOT simply copy text from the retrieved documents

Enhanced Query (respond with ONLY the enhanced query, no explanations):"""

            # Invoke LLM for query enhancement
            response = self.llm.invoke(enhancement_prompt)
            # Handle response content which can be string or list
            if isinstance(response.content, str):
                enhanced_query = response.content.strip()
            elif isinstance(response.content, list):
                # Join list items if content is a list
                enhanced_query = " ".join(
                    [str(item) for item in response.content]
                ).strip()
            else:
                enhanced_query = str(response.content).strip()

            logger.info(f"Original query: '{original_query}'")
            logger.info(f"Enhanced query: '{enhanced_query}'")

            return {"enhanced_query": enhanced_query}

        except Exception as e:
            logger.error(f"Error during query enhancement: {str(e)}")
            # Fallback to original query on error
            return {"enhanced_query": str(state.get("messages")[-1].content)}

    def retrieve_final(self, state: ConversationState) -> Dict[str, Any]:
        """
        Perform final retrieval using the enhanced query.

        This retrieves the single most relevant document using the enhanced query
        and extracts video metadata if applicable.

        Returns:
        - context: List containing the top retrieved document
        - video_metadata: Video playback information if available
        """
        # Check if we have any documents in the vector store
        vector_data = self.get_vector_store_data()
        has_documents = bool(vector_data.get("ids"))

        if not has_documents:
            logger.info("No documents in vector store - skipping final retrieval")
            return {"context": [], "video_metadata": None}

        try:
            # Get enhanced query from state
            enhanced_query = state.get("enhanced_query")

            # If no enhanced query, fall back to original
            if not enhanced_query:
                enhanced_query = str(state.get("messages")[-1].content)
                logger.warning("No enhanced query found, using original query")

            # Perform retrieval with enhanced query
            # Use k=1 to get only the most relevant document
            retrieved_docs = self.similarity_search(enhanced_query, k=15)

            if not retrieved_docs:
                logger.info("No relevant documents found with enhanced query")
                return {"context": [], "video_metadata": None}

            # Take only the top result
            top_doc = [retrieved_docs[0]]
            logger.info(
                f"Final retrieval selected top document: {top_doc[0].metadata.get('source', 'unknown')}"
            )

            # Extract video metadata from the top document
            video_metadata = self._extract_video_metadata(top_doc)

            return {"context": top_doc, "video_metadata": video_metadata}

        except Exception as e:
            logger.error(f"Error during final retrieval: {str(e)}")
            return {"context": [], "video_metadata": None}

    def generate(self, state: ConversationState) -> Dict[str, List[BaseMessage]]:
        """Generate response using retrieved context or pure generation."""
        if not self.llm:
            raise ValueError("No LLM available. Please check LLM initialization.")

        try:
            logger.info(
                f"Generating response for state with {len(state.get('messages', []))} messages"
            )
            context_docs = state.get("context", [])

            context_data = (
                "\n\n".join([doc.page_content for doc in context_docs])
                if context_docs
                else "No relevant document found in the knowledge base."
            )

            if self.prompt_template:
                domain = state.get("selected_domain")
                domain_line = (
                    f"\n\nVous vous concentrez particulièrement sur le domaine : {domain}."
                    if domain else ""
                )
                filled_prompt = self.prompt_template.invoke(
                    {
                        "history": [
                            f"{msg.type}: {msg.content}"
                            for msg in state.get("messages", [])[:-1]
                        ],
                        "query": str(state.get("messages")[-1].content),
                        "context": context_docs,
                        "specific_domain": domain_line,
                    }
                )
            else:
                logger.warning("No prompt template available, using fallback.")
                filled_prompt = f"""
                Question: {str(state.get('messages'))}
                \nContext: {context_data}
                \nAnswer:
                """

            response = self.llm.invoke(filled_prompt)

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
                self.add_documents(all_documents)
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
                self.add_documents(all_documents)
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

    def route_query(self, state: ConversationState) -> Dict[str, Any]:
        """Decide whether to retrieve from the knowledge base or answer directly.

        Routing rules (checked in order):
        1. Vector store empty → llm_only  (no content to retrieve)
        2. No LLM available  → rag        (fallback: let retrieval try anyway)
        3. LLM classifies the message     → rag | llm_only
        """
        # Rule 1 — empty store: skip retrieval entirely
        vector_data = self.get_vector_store_data()
        if not vector_data.get("ids"):
            logger.info("Vector store is empty — routing to direct LLM")
            return {"route": "llm_only"}

        if not self.llm:
            logger.warning("No LLM available for routing — defaulting to rag")
            return {"route": "rag"}

        query = str(state.get("messages")[-1].content)
        domain = state.get("selected_domain")
        domain_context = (
            f'  The user is currently focused on the craft domain: "{domain}".\n'
            if domain else ""
        )

        prompt = (
            "You are a routing classifier for a vocational-training assistant.\n"
            "Classify the following message as EXACTLY one of:\n"
            '  "rag"      — the question is about craft techniques, gestures, tools, materials,\n'
            "               training procedures, videos, or any domain-specific knowledge that\n"
            "               may be found in the knowledge base.\n"
            '  "llm_only" — the message is a greeting, chitchat, a general-knowledge question,\n'
            "               or anything that does NOT require retrieved training content.\n"
            + domain_context + "\n"
            f'Message: "{query}"\n\n'
            "Reply with exactly one word: rag or llm_only"
        )

        try:
            response = self.llm.invoke(prompt)
            raw = response.content.strip().lower()

            if "llm_only" in raw:
                route = "llm_only"
            elif "rag" in raw:
                route = "rag"
            else:
                logger.warning(f"Ambiguous routing response '{raw}' — defaulting to rag")
                route = "rag"

            logger.info(f"Router: '{query[:80]}' → {route}")
            return {"route": route}

        except Exception as e:
            logger.error(f"Routing failed ({e}) — defaulting to rag")
            return {"route": "rag"}

    def direct_generate(self, state: ConversationState) -> Dict[str, List[BaseMessage]]:
        """Generate a response directly from LLM weights, without retrieval."""
        if not self.llm:
            raise ValueError("No LLM available. Please check LLM initialization.")

        try:
            history_lines = [
                f"{msg.type}: {msg.content}"
                for msg in state.get("messages", [])[:-1]
            ]
            history_text = "\n".join(history_lines) if history_lines else "(début de conversation)"
            query = str(state.get("messages")[-1].content)

            domain = state.get("selected_domain")
            domain_line = (
                f"\nVous vous concentrez particulièrement sur le domaine : {domain}."
                if domain else ""
            )
            direct_prompt = (
                "Vous êtes un assistant pédagogique pour des apprentis dans les arts et l'artisanat "
                "(soufflage de verre, menuiserie, travail du cuir, assemblage, etc.)."
                + domain_line + "\n\n"
                f"Historique de conversation :\n{history_text}\n\n"
                f"Message de l'apprenti : {query}\n\n"
                "Répondez de manière concise et bienveillante."
            )

            response = self.llm.invoke(direct_prompt)
            logger.info("Direct generation complete (no retrieval)")
            return {"messages": [response]}

        except Exception as e:
            logger.error(f"Direct generation failed: {e}")
            raise

    def multi_query(self, state: ConversationState) -> Dict[str, Any]:
        """Generate multiple query variants for broader retrieval."""
        if not self.llm:
            return {"query_variants": [str(state.get("messages")[-1].content)]}

        original_query = str(state.get("messages")[-1].content)
        prompt = f"Generate 3 alternative phrasings of this apprenticeship query for better search in videos and lessons: '{original_query}'. Focus on synonyms and related techniques. Respond with comma-separated variants only."
        response = self.llm.invoke(prompt)
        variants = [v.strip() for v in response.content.split(",") if v.strip()]
        variants = variants[:3]  # Limit to 3
        if not variants:
            variants = [original_query]
        logger.info(f"Generated query variants: {variants}")
        return {"query_variants": variants}

    def retrieve_combined(self, state: ConversationState) -> Dict[str, Any]:
        """Retrieve and combine docs from all query variants."""
        variants = state.get("query_variants", [str(state.get("messages")[-1].content)])
        all_docs = []
        seen_sources = set()
        for query in variants:
            docs = self.similarity_search(query, k=10)  # Smaller k for limited data
            for doc in docs:
                source = doc.metadata.get("source")
                if source not in seen_sources:
                    all_docs.append(doc)
                    seen_sources.add(source)
        logger.info(f"Combined {len(all_docs)} unique docs from variants")
        return {"context": all_docs[:30]}  # Candidate pool

    def rerank(self, state: ConversationState) -> Dict[str, Any]:
        """Re-rank context docs for relevance."""
        if not self.llm:
            top_docs = state.get("context", [])[:3]
            video_metadata = self._extract_video_metadata(top_docs)
            return {"context": top_docs, "video_metadata": video_metadata}

        original_query = str(state.get("messages")[-1].content)
        docs = state.get("context", [])
        if len(docs) <= 3:
            top_docs = docs
        else:
            # Simple re-rank: Score top 10 with LLM
            scored = []
            for doc in docs[:10]:
                prompt = f"Rate how relevant this content is to the query '{original_query}' (1-5, where 5 is highly relevant): {doc.page_content[:150]}"
                try:
                    score = int(self.llm.invoke(prompt).content.strip())
                except:
                    score = 1
                scored.append((doc, score))
            top_docs = [
                doc for doc, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:3]
            ]

        video_metadata = self._extract_video_metadata(top_docs)
        logger.info(f"Re-ranked to top 3 docs")
        return {"context": top_docs, "video_metadata": video_metadata}
