"""RAG service for document retrieval and generation."""

import os
import logging
from typing import List, Dict, Any, Union, Optional
from langsmith import traceable
from typing_extensions import Literal
from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain import hub
from sentence_transformers import CrossEncoder

from config.settings import ConfigurationManager
from services.reranker_service import InfomaniakReranker
from core.types import ConversationState


logger = logging.getLogger(__name__)


def build_cohort_filter(user_cohort_ids: list) -> dict:
    """Build a ChromaDB `where` filter enforcing cohort-level access.

    Documents pass if they are open-access (cohort_id == -1, open_access == True)
    OR if their cohort_id is in the user's allowed cohort list.
    """
    if not user_cohort_ids:
        return {"open_access": True}
    return {
        "$or": [
            {"cohort_id": {"$in": list(user_cohort_ids)}},
            {"open_access": True},
        ]
    }


class RAGService:
    """Service for RAG (Retrieval Augmented Generation) operations."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        use_hub_template: bool = False,
        annotation_service: Optional[Any] = None,
        course_rag_service: Optional[Any] = None,
    ):
        self.config_manager = config_manager
        self.config = self.config_manager.get_config().rag
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._initialize_vector_store()
        self.llm = self._initialize_llm()
        self.cross_encoder = self._initialize_cross_encoder()
        self.annotation_service = annotation_service  # Optional dependency
        self.course_rag_service = course_rag_service  # Optional — per-course collections

        if use_hub_template:
            self.prompt_template = self._load_prompt_template()
            self.system_prompt = None
            self.user_template = None
        else:
            self.prompt_template = None  # unused — replaced by system_prompt + user_template
            self.system_prompt = (
                "Vous êtes un assistant pédagogique expert qui aide des apprentis dans les arts et métiers "
                "(soufflage de verre, ganterie, menuiserie, sellerie, etc.) à maîtriser les techniques et les "
                "connaissances de leur domaine.\n\n"
                "RÈGLES ABSOLUES — respectez-les impérativement :\n"
                "- Répondez TOUJOURS en français correct et soigné, sans fautes d'orthographe ni de grammaire.\n"
                "- N'utilisez JAMAIS d'emojis.\n"
                "- Ne produisez JAMAIS de balises <think> ni de raisonnement interne visible.\n"
                "- N'inventez JAMAIS d'URLs, de liens, de références bibliographiques ou de citations.\n"
                "- Basez-vous EXCLUSIVEMENT sur le contexte documentaire fourni. "
                "Si le contexte est insuffisant ou ne traite pas de la question posée, répondez UNIQUEMENT : "
                "\"Je n'ai pas trouvé d'information pertinente dans le corpus pour répondre à cette question. "
                "Veuillez reformuler ou consulter votre formateur.\" "
                "Ne complétez JAMAIS par des connaissances extérieures au contexte fourni.\n\n"
                "STRUCTURE DE LA RÉPONSE — adaptez-la à la nature de la question :\n"
                "- Pour une question factuelle simple (température, durée, proportion, définition…), "
                "répondez directement et précisément sans imposer de sections superflues.\n"
                "- Pour une question procédurale ou gestuelle (comment réaliser une action), "
                "structurez la réponse avec des sections pertinentes : étapes clés, erreurs fréquentes et corrections.\n"
                "- Dans tous les cas, utilisez le markdown (titres, listes à puces ou numérotées, tableaux) "
                "uniquement lorsqu'il améliore la lisibilité.\n\n"
                "SECTION OBLIGATOIRE EN FIN DE RÉPONSE :\n"
                "Ajoutez toujours une section \"**Pour aller plus loin**\" avec exactement trois questions de suivi "
                "nommées A, B et C. "
                "A et B approfondissent le sujet de la réponse. "
                "C explore un aspect connexe différent pour élargir la culture de l'apprenti.\n"
                "Format :\n"
                "**A.** [question A]\n"
                "**B.** [question B]\n"
                "**C.** [question C — sujet connexe]\n\n"
                "L'apprenti peut répondre avec une seule lettre (A, B ou C) pour développer la question correspondante."
            )
            self.user_template = (
                "Historique de la conversation :\n<history>\n{history}\n</history>\n\n"
                "Contexte documentaire récupéré :\n<context>\n{context}\n</context>\n\n"
                "Requête de l'apprenti :\n<query>\n{query}\n</query>"
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

    def _build_messages(
        self,
        state: ConversationState,
        context_data: str,
    ) -> List:
        """Build a [SystemMessage, HumanMessage] list for the LLM.

        Uses the split system_prompt / user_template when available (Infomaniak
        path), falling back to the legacy PromptTemplate for hub-loaded templates.
        """
        domain = state.get("selected_domain")
        domain_suffix = (
            f"\n\nVous vous concentrez particulièrement sur le domaine : {domain}."
            if domain else ""
        )
        query = str(state.get("messages")[-1].content) + domain_suffix
        history_lines = [
            f"{msg.type}: {msg.content}"
            for msg in state.get("messages", [])[:-1]
        ]
        history_text = "\n".join(history_lines) if history_lines else "(début de conversation)"

        if self.system_prompt and self.user_template:
            user_text = self.user_template.format(
                history=history_text,
                context=context_data,
                query=query,
            )
            return [SystemMessage(content=self.system_prompt), HumanMessage(content=user_text)]

        # Fallback: legacy hub template returns a StringPromptValue
        if self.prompt_template:
            filled = self.prompt_template.invoke(
                {"history": history_lines, "context": context_data, "query": query}
            )
            return [HumanMessage(content=filled.text if hasattr(filled, "text") else str(filled))]

        return [HumanMessage(content=f"Context: {context_data}\n\nQuestion: {query}\n\nAnswer:")]

    def _initialize_embeddings(self) -> OpenAIEmbeddings:
        """Initialize Infomaniak embeddings (OpenAI-compatible endpoint)."""
        try:
            api_key = self.config_manager.get_env_var("INFOMANIAK_API_KEY")
            product_id = self.config_manager.get_env_var("INFOMANIAK_PRODUCT_ID")
            base_url = f"https://api.infomaniak.com/2/ai/{product_id}/openai/v1"
            embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=api_key,
                openai_api_base=base_url,
            )
            logger.info(
                f"Embeddings initialized with model: {self.config.embedding_model} "
                f"via Infomaniak (product_id={product_id})"
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
        """Initialize Infomaniak chat model (OpenAI-compatible endpoint)."""
        try:
            api_key = self.config_manager.get_env_var("INFOMANIAK_API_KEY")
            product_id = self.config_manager.get_env_var("INFOMANIAK_PRODUCT_ID")
            base_url = f"https://api.infomaniak.com/2/ai/{product_id}/openai/v1"
            llm = ChatOpenAI(
                model=self.config.llm_model,
                openai_api_key=api_key,
                openai_api_base=base_url,
                streaming=True,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                top_p=self.config.llm_top_p,
                frequency_penalty=self.config.llm_frequency_penalty,
                presence_penalty=self.config.llm_presence_penalty,
                # Prevent the model from calling web search or other tools
                model_kwargs={"tool_choice": "none"},
            )
            logger.info(
                f"LLM initialized: {self.config.llm_model} "
                f"via Infomaniak (product_id={product_id})"
            )
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            logger.error("Check that INFOMANIAK_API_KEY and INFOMANIAK_PRODUCT_ID are set in your .env file")
            raise RuntimeError(
                f"LLM initialization failed: {str(e)}. "
                "Please ensure INFOMANIAK_API_KEY and INFOMANIAK_PRODUCT_ID are properly configured in your .env file."
            )

    # Model name for the multilingual cross-encoder reranker.
    # bge-reranker-v2-m3 is a multilingual model with calibrated scores where
    # 0.0 is a meaningful relevance boundary, unlike mmarco models whose raw
    # logits are systematically negative on non-web-document corpora.
    CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"

    # Minimum cross-encoder relevance score.  BGE reranker outputs scores in a
    # range where 0.0 separates relevant from non-relevant, so this threshold
    # can be used at face value without corpus-specific calibration.
    RERANK_SCORE_THRESHOLD: float = 0.0

    def _initialize_cross_encoder(self):
        """Load local cross-encoder or skip if remote reranker is configured."""
        if self.config_manager.get_config().rag.use_remote_reranker:
            logger.info("Remote reranker configured — skipping local cross-encoder load")
            return None

        try:
            model = CrossEncoder(
                self.CROSS_ENCODER_MODEL,
                device="cpu",
                trust_remote_code=True,
            )
            # Sanity-check: a model that returns all-zero scores for any input
            # has not initialised correctly and would pass every doc through.
            test_scores = model.predict([("test query", "test document")])
            if float(test_scores[0]) == 0.0:
                raise RuntimeError(
                    f"Cross-encoder {self.CROSS_ENCODER_MODEL} returned a zero "
                    "score on a sanity-check pair — classification head likely "
                    "uninitialised. Check model name and sentence-transformers version."
                )
            logger.info(f"Cross-encoder reranker loaded: {self.CROSS_ENCODER_MODEL}")
            return model
        except Exception as e:
            logger.error(f"Failed to load cross-encoder ({e})")
            raise

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
        k: Optional[int] = None,
        cohort_filter: Optional[dict] = None,
    ) -> List[Document]:
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

            kwargs = {}
            if cohort_filter is not None:
                kwargs["filter"] = cohort_filter

            results = self.vector_store.max_marginal_relevance_search(query, k=k, **kwargs)
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

    def generate_hypothetical_document(
        self, state: ConversationState
    ) -> Dict[str, Any]:
        """
        Generate a hypothetical expert elicitation using HyDE approach.

        Takes vague user query like "how do I hold my blowpipe" and generates
        a synthetic expert-style elicitation that would answer it. This synthetic
        document is then used for embedding similarity search.

        Returns:
        - hypothetical_document: Generated expert-style explanation
        """
        if not self.llm:
            logger.warning("No LLM available for HyDE generation")
            return {"hypothetical_document": None}

        try:
            original_query = str(state.get("messages")[-1].content)

            # Create HyDE prompt tailored to expert elicitations in arts and crafts
            hyde_prompt = f"""Tu es un expert artisan fournissant une élicitation verbale détaillée de ta technique pendant que tu la démontres. Un apprenti te demande : "{original_query}"

Génère une explication détaillée à la première personne de la technique comme si tu verbalisais tes mouvements pendant une démonstration. Inclus :
- Le positionnement précis des mains et la description de la prise
- Les angles et orientations des outils
- Le timing et le rythme des mouvements
- Les sensations physiques et les retours que tu ressens
- Les erreurs courantes et les corrections
- La terminologie technique utilisée dans le métier

Écris 2-3 paragraphes dans le style d'un expert qui pense à voix haute pendant une démonstration. Sois spécifique et technique.

Élicitation d'expert :"""

            # Generate hypothetical document
            response = self.llm.invoke(hyde_prompt)

            if isinstance(response.content, str):
                hypothetical_doc = response.content.strip()
            elif isinstance(response.content, list):
                hypothetical_doc = " ".join(
                    [str(item) for item in response.content]
                ).strip()
            else:
                hypothetical_doc = str(response.content).strip()

            logger.info(
                f"HyDE generated document length: {len(hypothetical_doc)} chars"
            )
            logger.info(f"HyDE preview: {hypothetical_doc[:200]}...")

            return {"hypothetical_document": hypothetical_doc}

        except Exception as e:
            logger.error(f"Error during HyDE generation: {str(e)}")
            return {"hypothetical_document": None}

    def retrieve_with_hyde(self, state: ConversationState) -> Dict[str, Any]:
        """
        Retrieve using HyDE-generated document instead of original query.

        Uses the synthetic expert elicitation for embedding similarity,
        which should better match actual expert transcript language.
        """
        vector_data = self.get_vector_store_data()
        has_documents = bool(vector_data.get("ids"))

        if not has_documents:
            logger.info(
                "No documents in vector store - switching to pure generation mode"
            )
            return {"context": [], "video_metadata": None}

        try:
            # Use hypothetical document if available, else fall back to original query
            hyde_doc = state.get("hypothetical_document")

            if hyde_doc:
                search_query = hyde_doc
                logger.info("Using HyDE-generated document for retrieval")
            else:
                search_query = str(state.get("messages")[-1].content)
                logger.warning("No HyDE document available, using original query")

            # Single retrieval pass with appropriate k
            user_cohort_ids = state.get("user_cohort_ids") or []
            cohort_filter = build_cohort_filter(user_cohort_ids) if user_cohort_ids is not None else None
            retrieved_docs = self.similarity_search(search_query, k=5, cohort_filter=cohort_filter)

            if not retrieved_docs:
                logger.info("No relevant documents found")
                return {"context": [], "video_metadata": None}

            # Extract video metadata from top result
            video_metadata = self._extract_video_metadata(retrieved_docs[:1])

            logger.info(f"Retrieved {len(retrieved_docs)} documents using HyDE")

            return {"context": retrieved_docs, "video_metadata": video_metadata}

        except Exception as e:
            logger.error(f"Error during HyDE retrieval: {str(e)}")
            return {"context": [], "video_metadata": None}

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
                else "Aucun document pertinent trouvé dans la base de connaissances."
            )

            messages = self._build_messages(state, context_data)
            response = self.llm.invoke(messages)
            return {"messages": [response]}

        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise

    @traceable(name="stream_generate", run_type="llm")
    async def stream_generate(self, state: ConversationState):
        """Async generator that streams LLM tokens for the generate step.

        When no documents were retrieved, emits a hard-coded refusal instead of
        calling the LLM — this prevents the model from hallucinating answers
        from its parametric weights.  The LLM is only invoked when there is
        actual corpus context to ground the response.
        """
        if not self.llm:
            raise ValueError("No LLM available. Please check LLM initialization.")

        context_docs = state.get("context", [])

        if not context_docs:
            logger.info("stream_generate: no context — returning deterministic refusal")
            yield (
                "Je ne dispose pas de ressources suffisantes dans la base documentaire "
                "pour répondre à cette question de manière fiable. "
                "Veuillez consulter votre formateur ou vérifier que les contenus du cours "
                "ont bien été intégrés dans le système."
            )
            return

        context_data = "\n\n".join([doc.page_content for doc in context_docs])
        messages = self._build_messages(state, context_data)

        in_think_block = False
        async for chunk in self.llm.astream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if not token:
                continue
            # Strip <think>...</think> reasoning blocks that some models emit
            if "<think>" in token:
                in_think_block = True
            if in_think_block:
                if "</think>" in token:
                    in_think_block = False
                    token = token.split("</think>", 1)[-1]
                else:
                    continue
            if token:
                yield token

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
        user_cohort_ids = state.get("user_cohort_ids") or []
        cohort_filter = build_cohort_filter(user_cohort_ids) if user_cohort_ids is not None else None
        all_docs = []
        seen_sources = set()
        for query in variants:
            docs = self.similarity_search(query, k=10, cohort_filter=cohort_filter)  # Smaller k for limited data
            for doc in docs:
                source = doc.metadata.get("source")
                if source not in seen_sources:
                    all_docs.append(doc)
                    seen_sources.add(source)
        logger.info(f"Combined {len(all_docs)} unique docs from variants")
        return {"context": all_docs[:30]}  # Candidate pool

    @traceable(name="rerank", run_type="chain")
    def rerank(self, state: ConversationState) -> Dict[str, Any]:
        """Rerank retrieved docs by relevance — local cross-encoder or remote API.

        When use_remote_reranker=True, delegates to InfomaniakReranker (HTTP API).
        When False, uses the local multilingual cross-encoder (no API call).
        Docs below threshold are dropped; empty context triggers the deterministic
        refusal in stream_generate / generate.
        """
        query = str(state.get("messages")[-1].content)
        docs = state.get("context", [])

        if not docs:
            return {"context": [], "video_metadata": None}

        rag_cfg = self.config_manager.get_config().rag

        if rag_cfg.use_remote_reranker:
            api_key = self.config_manager.get_env_var("INFOMANIAK_API_KEY")
            product_id = self.config_manager.get_env_var("INFOMANIAK_PRODUCT_ID")
            remote = InfomaniakReranker(
                api_key=api_key,
                product_id=product_id,
                model=rag_cfg.reranker_model,
                threshold=rag_cfg.remote_reranker_score_threshold,
            )
            passing = remote.rerank(query, docs)
            logger.info(
                f"rerank (remote): {len(docs)} candidates → {len(passing)} passed "
                f"threshold={rag_cfg.remote_reranker_score_threshold}"
            )
            video_metadata = self._extract_video_metadata(passing)
            return {
                "context": passing,
                "video_metadata": video_metadata,
                "rerank_debug": {
                    "disabled": False,
                    "backend": "remote",
                    "model": rag_cfg.reranker_model,
                    "candidates_in": len(docs),
                    "passing_out": len(passing),
                    "threshold": rag_cfg.remote_reranker_score_threshold,
                },
            }

        # Local cross-encoder path
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.cross_encoder.predict(pairs)

        scored_docs = sorted(
            zip(scores, docs), key=lambda x: x[0], reverse=True
        )

        passing = [
            doc for score, doc in scored_docs
            if score >= self.RERANK_SCORE_THRESHOLD
        ]

        top_score = float(scores.max())
        all_scores_sorted = sorted([round(float(s), 4) for s in scores.tolist()], reverse=True)

        logger.info(
            f"rerank (local): {len(docs)} candidates → {len(passing)} passed threshold "
            f"(top score={top_score:.2f}, threshold={self.RERANK_SCORE_THRESHOLD})"
        )

        video_metadata = self._extract_video_metadata(passing)
        return {
            "context": passing,
            "video_metadata": video_metadata,
            "rerank_debug": {
                "disabled": False,
                "backend": "local",
                "candidates_in": len(docs),
                "passing_out": len(passing),
                "threshold": self.RERANK_SCORE_THRESHOLD,
                "top_score": round(top_score, 4),
                "scores": all_scores_sorted,
            },
        }

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

    # ============================================================================
    # PRF PIPELINE — corpus-grounded query refinement
    # Replaces HyDE as the active retrieval strategy.
    # Graph: retrieve_initial → refine_query_prf → retrieve_final_dual → generate
    # ============================================================================

    def _merge_dedup(
        self, a: List[Document], b: List[Document]
    ) -> List[Document]:
        """Merge two document lists, deduplicating by metadata.source."""
        seen: set = set()
        merged: List[Document] = []
        for doc in a + b:
            src = doc.metadata.get("source", "")
            if src not in seen:
                seen.add(src)
                merged.append(doc)
        return merged

    # Maximum number of docs fed into the LLM context across all sources.
    # Keeps the total prompt well within the Apertus-70B 16 384-token window
    # (system prompt ≈ 450 tok + 8 chunks × 400 tok = 3 650 tok + 1 200 tok output
    # = 4 850 tok, leaving a comfortable margin).
    MAX_CONTEXT_DOCS = 8

    @traceable(name="retrieve_initial", run_type="retriever")
    def retrieve_initial(self, state: ConversationState) -> Dict[str, Any]:
        """PRF step 1 — first-pass retrieval with the raw user query.

        Queries both the video annotation collection and the per-course collection
        (if course_id is present in state).  Results are stored in state["context"]
        for the subsequent PRF reformulation step.  Relevance filtering is handled
        downstream by the cross-encoder rerank node — this step casts a wide net.
        """
        vector_data = self.get_vector_store_data()
        has_annotation_docs = bool(vector_data.get("ids"))

        query = str(state.get("messages")[-1].content)
        course_id = state.get("course_id")

        results: List[Document] = []

        # 1. Video annotations collection.
        user_cohort_ids = state.get("user_cohort_ids") or []
        cohort_filter = build_cohort_filter(user_cohort_ids) if user_cohort_ids is not None else None
        if has_annotation_docs:
            annotation_results = self.similarity_search(query, k=5, cohort_filter=cohort_filter)
            results.extend(annotation_results)
        else:
            logger.info("Annotation collection empty — skipping annotation retrieval")

        # 2. Course collections — priority course gets k=6, all others k=1.
        if self.course_rag_service:
            course_results = self.course_rag_service.similarity_search_all_courses(
                query,
                k_per_course=1,
                priority_course_id=course_id,
            )
            results = self._merge_dedup(results, course_results)
        elif course_id:
            logger.warning("course_id provided but course_rag_service not injected")

        if not results:
            logger.info("retrieve_initial: no documents found")
            return {"context": [], "video_metadata": None}

        results = results[: self.MAX_CONTEXT_DOCS]
        video_metadata = self._extract_video_metadata(results)
        logger.info(f"retrieve_initial: {len(results)} docs retrieved")
        return {"context": results, "video_metadata": video_metadata}

    @traceable(name="refine_query_prf", run_type="chain")
    def refine_query_prf(self, state: ConversationState) -> Dict[str, Any]:
        """PRF step 2 — corpus-grounded query reformulation.

        Uses the top-3 retrieved documents from the first pass to reformulate
        the original query using vocabulary from the actual corpus, not LLM
        parametric knowledge.  Falls back to original query if no context or
        no LLM is available.
        """
        original_query = str(state.get("messages")[-1].content)
        context_docs = state.get("context", [])

        if not context_docs or not self.llm:
            logger.info("refine_query_prf: no context or LLM — using original query")
            return {"refined_query": original_query}

        # Take up to top-3 docs for the reformulation prompt
        snippets = []
        for i, doc in enumerate(context_docs[:3], 1):
            doc_type = doc.metadata.get("module_type") or doc.metadata.get("transcript_type", "text")
            preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            snippets.append(f"[Document {i} — {doc_type}]\n{preview}")

        context_text = "\n\n".join(snippets)

        # Key distinction from plain enhance_query: we explicitly instruct the LLM
        # to use vocabulary FROM the corpus, not invent expert-sounding terms.
        prf_prompt = (
            "Tu es un assistant de reformulation de requête pour un système de recherche documentaire.\n\n"
            "Requête originale de l'apprenti :\n"
            f'"{original_query}"\n\n'
            "Documents récupérés (utilise leur vocabulaire technique, ne l'invente pas) :\n"
            f"{context_text}\n\n"
            "Instructions :\n"
            "1. Identifie les termes techniques et le vocabulaire du domaine présents dans les documents ci-dessus.\n"
            "2. Reformule la requête de l'apprenti en incorporant ces termes techniques issus du corpus.\n"
            "3. Préserve l'intention originale de la question.\n"
            "4. Réponds avec UNIQUEMENT la requête reformulée, sans explication (1-2 phrases maximum).\n\n"
            "Requête reformulée :"
        )

        try:
            response = self.llm.invoke(prf_prompt)
            if isinstance(response.content, str):
                refined = response.content.strip()
            elif isinstance(response.content, list):
                refined = " ".join(str(item) for item in response.content).strip()
            else:
                refined = str(response.content).strip()

            logger.info(f"PRF: '{original_query}' → '{refined}'")
            return {"refined_query": refined}

        except Exception as e:
            logger.error(f"refine_query_prf failed: {e} — using original query")
            return {"refined_query": original_query}

    @traceable(name="retrieve_final_dual", run_type="retriever")
    def retrieve_final_dual(self, state: ConversationState) -> Dict[str, Any]:
        """PRF step 3 — second-pass retrieval using the refined query.

        Queries both collections again with the PRF-improved query and replaces
        state["context"] with the final candidate set.  Relevance filtering is
        handled downstream by the cross-encoder rerank node.
        """
        refined_query = state.get("refined_query") or str(state.get("messages")[-1].content)
        course_id = state.get("course_id")

        vector_data = self.get_vector_store_data()
        has_annotation_docs = bool(vector_data.get("ids"))

        annotation_results: List[Document] = []
        user_cohort_ids = state.get("user_cohort_ids") or []
        cohort_filter = build_cohort_filter(user_cohort_ids) if user_cohort_ids is not None else None

        # 1. Video annotations.
        if has_annotation_docs:
            annotation_results = self.similarity_search(refined_query, k=5, cohort_filter=cohort_filter)

        # 2. Course collections.
        course_results: List[Document] = []
        if self.course_rag_service:
            course_results = self.course_rag_service.similarity_search_all_courses(
                refined_query,
                k_per_course=1,
                priority_course_id=course_id,
            )

        results = self._merge_dedup(annotation_results, course_results)

        if not results:
            logger.info("retrieve_final_dual: no documents found with refined query")
            return {"context": [], "video_metadata": None}

        results = results[: self.MAX_CONTEXT_DOCS]
        video_metadata = self._extract_video_metadata(results)
        logger.info(
            f"retrieve_final_dual: {len(results)} candidates "
            f"(annotations={len(annotation_results)}, course={len(course_results)})"
        )
        return {"context": results, "video_metadata": video_metadata}

    # ============================================================================
    # LEGACY METHODS (kept for reference - can be removed after testing HyDE)
    # ============================================================================

    def retrieve(self, state: ConversationState) -> Dict[str, Any]:
        """[LEGACY] Retrieve relevant documents for a given state (initial retrieval for query enhancement)."""
        # Check if we have any documents in the vector store
        vector_data = self.get_vector_store_data()
        has_documents = bool(vector_data.get("ids"))
        logger.info(f"State at retrieve: {state}")

        if has_documents:
            user_cohort_ids = state.get("user_cohort_ids") or []
            cohort_filter = build_cohort_filter(user_cohort_ids) if user_cohort_ids is not None else None
            retrieved_docs = self.similarity_search(
                str(state.get("messages")[-1].content), cohort_filter=cohort_filter
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
        [LEGACY] Enhance user query using LLM based on initially retrieved documents.

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
        [LEGACY] Perform final retrieval using the enhanced query.

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
            user_cohort_ids = state.get("user_cohort_ids") or []
            cohort_filter = build_cohort_filter(user_cohort_ids) if user_cohort_ids is not None else None
            retrieved_docs = self.similarity_search(enhanced_query, k=15, cohort_filter=cohort_filter)

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
