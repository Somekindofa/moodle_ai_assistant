"""RAG service for document retrieval and generation."""

import hashlib
import os
import re
import base64
import logging
from typing import List, Dict, Any, Tuple, Union, Optional
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
from services import translation_service
from core.types import ConversationState


logger = logging.getLogger(__name__)


def merge_dedup_interleaved(a: list, b: list) -> List:
    """Merge two retrieval result lists round-robin, deduplicating by source.

    Interleaved rather than concatenated because the merged list is truncated
    downstream (MAX_CONTEXT_DOCS here, MAX_RERANK_CANDIDATES in the pipeline).
    Concatenating put every annotation ahead of every course chunk, so a cut at
    N could — and did — discard all course content before the reranker scored
    any of it: course material never lost on relevance, it just never entered
    the contest. Round-robin makes truncation cost both sources equally.

    Leftovers from the longer list are appended once the shorter one runs out.
    """
    seen: set = set()
    merged: List = []

    for index in range(max(len(a), len(b))):
        for source_list in (a, b):
            if index >= len(source_list):
                continue
            doc = source_list[index]
            key = doc.metadata.get("source", "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)

    return merged


def stable_document_id(document) -> str:
    """A deterministic id for a document, so re-ingesting replaces rather than appends.

    Chroma upserts when given ids, so a stable id makes ingestion idempotent.
    Without one, langchain_chroma generates a fresh UUID per call and the
    startup annotation sync appends a full copy of the corpus every restart.

    Two keying strategies, because the store holds two shapes of document:

    * **Video annotations** key on ``source`` (``file.mp4#<id>_raw``), which is
      unique per annotation. Deliberately *not* content-based: re-running
      translation rewrites the text, and that must update the existing document
      rather than create a second copy of the same clip.
    * **Everything else** (file chunks from app.py) keys on source + content,
      because one file is split into many chunks that all share a ``source``;
      keying on source alone would collapse a document into its last chunk.
    """
    metadata = getattr(document, "metadata", None) or {}
    source = metadata.get("source") or ""

    if metadata.get("type") == "video_annotation" and source:
        return f"annotation::{source}"

    digest = hashlib.sha256(
        f"{source}|{document.page_content}".encode("utf-8")
    ).hexdigest()
    return f"sha256::{digest}"


def stable_document_ids(documents: list) -> List[str]:
    """Stable ids for a batch of documents. See stable_document_id."""
    return [stable_document_id(doc) for doc in documents]


def build_cohort_filter(user_cohort_ids: list, craft: Optional[str] = None) -> dict:
    """Build a ChromaDB `where` filter enforcing cohort-level access.

    Documents pass if they are open-access (cohort_id == -1, open_access == True)
    OR if their cohort_id is in the user's allowed cohort list.

    If `craft` is given (the student picked a domain focus button), it's ANDed
    on top as an additional requirement — see DOMAIN_MAP.
    """
    if not user_cohort_ids:
        base = {"open_access": True}
    else:
        base = {
            "$or": [
                {"cohort_id": {"$in": list(user_cohort_ids)}},
                {"open_access": True},
            ]
        }
    if craft:
        return {"$and": [base, {"craft": craft}]}
    return base


# Maps a domain-focus button's label (sent by the frontend as `selected_domain`,
# see plugin/templates/chat_interface.mustache) to the two identifiers needed to
# narrow retrieval: the Moodle course category (for course_content) and the
# annotation `craft` tag (for video_annotation). A domain is only added here
# once its category/craft actually exist and hold content — never the other
# way around, so a stale/unmapped label just falls through to unfiltered
# retrieval (see retrieve_initial/retrieve_final_dual) instead of dead-ending.
DOMAIN_MAP: Dict[str, Dict[str, Any]] = {
    "Soufflerie de verre": {"category_id": 25, "craft": "glassblowing"},
    "Ganterie": {"category_id": 34, "craft": "glovemaking"},
}


# ── Deterministic user-facing messages, keyed by ISO 639-1 language code ────
#
# These are the three things a learner can be shown *instead of* an answer:
# the pre-LLM topic classifier's rejection, assess_relevance's INSUFFICIENT
# refusal, and its AMBIGUOUS clarifying question. None of them goes through
# the LLM, so none of them was following the learner's language — the answer
# path already does (see _build_messages and query_language), which meant a
# Greek learner got Greek when retrieval worked and French exactly when it
# did not, i.e. at the moment they most needed to understand the reply.
#
# A static table rather than an LLM translation call per refusal, because:
#   - a refusal is already the slow, disappointing path, and a network round
#     trip is the wrong thing to add to it;
#   - translation can fail (see translation_service.translate_to_french's
#     error path, which returns None), and a failed translation of a refusal
#     would degrade to no message at all — the one output that must never be
#     missing;
#   - the strings are fixed and few. There is nothing here to generate.
#
# Coverage is deliberately limited to the languages this deployment is known
# to serve today: French (UI, corpus, prompts), plus English and Greek —
# both attested in eval/fixtures/xling_annotations_seed.json /
# xling_course_chunks_seed.json and in the GR-Glassblowing course. Any other
# detected language falls back to English (the learner demonstrably did not
# write French, so English is the better guess) and logs a warning naming the
# ISO code, so a genuine coverage gap surfaces in the logs and can be closed
# with data instead of guesswork. Add a language here only once something
# actually asks for it.
USER_MESSAGE_FALLBACK_LANG = "en"

USER_MESSAGES: Dict[str, Dict[str, str]] = {
    # assess_relevance == INSUFFICIENT. The French entry is also
    # RAGService.INSUFFICIENT_CONTEXT_MESSAGE and is interpolated verbatim
    # into the system prompt — see that attribute before editing it.
    "insufficient_context": {
        "fr": (
            "Je n'ai pas trouvé d'information pertinente dans le corpus pour répondre à cette question. "
            "Veuillez reformuler ou consulter votre formateur."
        ),
        "en": (
            "I could not find any relevant information in the corpus to answer this question. "
            "Please rephrase it, or ask your trainer."
        ),
        "el": (
            "Δεν βρήκα σχετικές πληροφορίες στο σώμα κειμένων για να απαντήσω σε αυτή την ερώτηση. "
            "Παρακαλώ αναδιατυπώστε την ή απευθυνθείτε στον εκπαιδευτή σας."
        ),
    },
    # Pre-LLM topic classifier said the question is out of domain.
    "off_topic": {
        "fr": (
            "Je n'ai pas trouvé d'information pertinente dans le corpus "
            "pour répondre à cette question. Veuillez poser une question "
            "sur les arts et métiers ou consulter votre formateur."
        ),
        "en": (
            "I could not find any relevant information in the corpus to answer "
            "this question. Please ask a question about the crafts and trades, "
            "or ask your trainer."
        ),
        "el": (
            "Δεν βρήκα σχετικές πληροφορίες στο σώμα κειμένων για να απαντήσω "
            "σε αυτή την ερώτηση. Παρακαλώ κάντε μια ερώτηση σχετική με τις "
            "τέχνες και τα επαγγέλματα ή απευθυνθείτε στον εκπαιδευτή σας."
        ),
    },
    # assess_relevance == AMBIGUOUS, and we have real topic names to list.
    # {topics} is the only placeholder in this table.
    "ambiguous_topics": {
        "fr": (
            "Votre question peut correspondre à plusieurs sujets du corpus "
            "({topics}). Pourriez-vous préciser votre demande ?"
        ),
        "en": (
            "Your question could match several topics in the corpus "
            "({topics}). Could you be more specific?"
        ),
        "el": (
            "Η ερώτησή σας μπορεί να αντιστοιχεί σε περισσότερα θέματα του σώματος κειμένων "
            "({topics}). Μπορείτε να τη διατυπώσετε πιο συγκεκριμένα;"
        ),
    },
    # assess_relevance == AMBIGUOUS with no nameable topic left after
    # placeholder filtering — see pipeline._build_ambiguous_clarification.
    "ambiguous_generic": {
        "fr": (
            "Votre question n'est pas assez précise pour que je trouve une réponse fiable "
            "dans le corpus. Pourriez-vous la reformuler avec plus de détails ?"
        ),
        "en": (
            "Your question is not specific enough for me to find a reliable answer "
            "in the corpus. Could you rephrase it with more detail?"
        ),
        "el": (
            "Η ερώτησή σας δεν είναι αρκετά συγκεκριμένη ώστε να βρω μια αξιόπιστη απάντηση "
            "στο σώμα κειμένων. Μπορείτε να την αναδιατυπώσετε με περισσότερες λεπτομέρειες;"
        ),
    },
}


def localized_message(key: str, language: Optional[str] = None, **fields: Any) -> str:
    """Return USER_MESSAGES[key] in `language`, degrading instead of raising.

    `language` is an ISO 639-1 code as produced by py3langid (see
    RAGService.detect_query_language); None/empty means French, matching the
    query_language default used everywhere else in the pipeline.

    Never raises: an unknown language falls back to
    USER_MESSAGE_FALLBACK_LANG and then to French, and a formatting failure
    returns the unformatted template. A refusal that blows up instead of
    printing is strictly worse than a refusal in the wrong language.
    """
    variants = USER_MESSAGES.get(key)
    if not variants:
        logger.error(f"localized_message: unknown message key '{key}'")
        return ""

    lang = (language or "fr").lower()
    text = variants.get(lang)
    if text is None:
        logger.warning(
            f"localized_message: no '{lang}' translation for '{key}' — "
            f"falling back to '{USER_MESSAGE_FALLBACK_LANG}'. Add '{lang}' to "
            "USER_MESSAGES if learners are writing in it."
        )
        text = variants.get(USER_MESSAGE_FALLBACK_LANG) or variants["fr"]

    if not fields:
        return text
    try:
        return text.format(**fields)
    except Exception as e:  # pragma: no cover — defensive
        logger.error(f"localized_message: formatting '{key}' failed: {e}")
        return text


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
        self._langid = self._initialize_langid()
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
                f"\"{self.INSUFFICIENT_CONTEXT_MESSAGE}\" "
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

        depth_preference = state.get("depth_preference", "normal")
        depth_suffix = ""
        if depth_preference == "brief":
            depth_suffix = "\n\nRéponds de manière brève et concise."
        elif depth_preference == "detailed":
            depth_suffix = "\n\nRéponds de manière détaillée et approfondie."

        desired_video_count = state.get("desired_video_count", 1)
        shown_video_count = len(state.get("video_metadata") or [])
        undersupply_suffix = ""
        if shown_video_count < desired_video_count:
            undersupply_suffix = (
                f"\n\n(Note interne : seulement {shown_video_count} vidéo(s) pertinente(s) "
                f"trouvée(s) sur les {desired_video_count} demandées — mentionne-le brièvement "
                "dans ta réponse, dans la langue de la question.)"
            )

        query = (
            str(state.get("messages")[-1].content)
            + domain_suffix + depth_suffix + undersupply_suffix
        )
        history_lines = [
            f"{msg.type}: {msg.content}"
            for msg in state.get("messages", [])[:-1]
        ]
        history_text = "\n".join(history_lines) if history_lines else "(début de conversation)"

        query_language = state.get("query_language")
        if query_language and query_language != "fr":
            system_prompt = self.system_prompt.replace(
                "- Répondez TOUJOURS en français correct et soigné, sans fautes d'orthographe ni de grammaire.\n",
                "- Répondez TOUJOURS dans la même langue que la question de l'apprenti "
                "(ci-dessous), avec une orthographe et une grammaire soignées.\n",
            )
            # The prompt also pins the exact refusal sentence to reply with when
            # the context is insufficient, in French. Left as-is, the "answer in
            # the learner's language" rule above and that literal French sentence
            # contradict each other, and the model resolves the contradiction
            # however it likes. Swap in the same refusal the deterministic
            # INSUFFICIENT path would have emitted, so both agree.
            system_prompt = system_prompt.replace(
                self.INSUFFICIENT_CONTEXT_MESSAGE,
                self.insufficient_context_message(query_language),
            )
        else:
            system_prompt = self.system_prompt

        if self.system_prompt and self.user_template:
            user_text = self.user_template.format(
                history=history_text,
                context=context_data,
                query=query,
            )
            return [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]

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

    # Single source of truth for the FRENCH "nothing relevant found" message —
    # used both in the system prompt (as a fallback if assess_relevance somehow
    # doesn't run) and deterministically by stream_response when
    # assess_relevance classifies the retrieved context as INSUFFICIENT.
    # Deterministic beats letting the LLM paraphrase it: a fixed string can't
    # drift or contradict itself the way free-form generation did.
    #
    # Deliberately still a plain ``str``, not a per-language mapping: __init__
    # interpolates it into self.system_prompt with an f-string, and the
    # system prompt is French regardless of the learner's language (only the
    # "answer in French" rule is swapped out — see _build_messages). Other
    # languages live in the module-level USER_MESSAGES table and are reached
    # through insufficient_context_message() below, which keeps honouring an
    # instance-level override of this attribute.
    INSUFFICIENT_CONTEXT_MESSAGE = USER_MESSAGES["insufficient_context"]["fr"]

    def insufficient_context_message(self, language: Optional[str] = None) -> str:
        """The INSUFFICIENT refusal, in the learner's language.

        French — and None/unknown, which is the pipeline-wide default for
        query_language — returns ``self.INSUFFICIENT_CONTEXT_MESSAGE`` rather
        than the table entry, so this attribute stays the single French source
        of truth and an instance-level override still wins.
        """
        if not language or language == "fr":
            return self.INSUFFICIENT_CONTEXT_MESSAGE
        return localized_message("insufficient_context", language)

    def detect_query_language(self, text: str) -> str:
        """ISO 639-1 language of `text` — local detection only, never an LLM call.

        detect_and_translate_query already computes this, but only as pipeline
        step 0, i.e. *after* the pre-LLM topic classifier has already had a
        chance to refuse the question. This exposes the same py3langid gate on
        its own so a refusal raised before (or without) that step can still be
        written in the learner's language.

        Uses exactly the same thresholds as detect_and_translate_query, so the
        two never disagree, and shares its bias: langid unavailable, low
        confidence, or too-short text all mean "fr". Never raises.
        """
        try:
            lang, _ = translation_service.decide_translation(
                text,
                self._langid,
                self.config.langid_confidence_threshold,
                self.config.min_langid_chars,
            )
            return lang if isinstance(lang, str) and lang else "fr"
        except Exception as e:
            logger.warning(f"detect_query_language failed ({e}) — assuming French")
            return "fr"

    # Per-document preview length for assess_relevance's classifier prompt.
    # Must cover a full chunk, not an arbitrary short slice — SemanticChunker
    # (course_rag_service.py) targets ~1600 chars per chunk (TARGET_TOKENS=400).
    # A shorter preview silently hides whichever part of a chunk didn't fit,
    # and the classifier then judges relevance on partial content without
    # any indication that anything was cut — confirmed live (2026-09-02) on a
    # 672-char chunk whose answer-bearing sentence sat past char 300 with the
    # previous limit, causing a well-matched (rerank score 0.92+) chunk to be
    # misjudged as insufficient.
    RELEVANCE_PREVIEW_CHARS = 1600

    # How many documents the classifier is shown. The same incident had a
    # second half that went unnoticed: the prompt used only the first 5
    # documents while retrieval supplies up to MAX_CONTEXT_DOCS. After
    # reranking, annotation clips can hold the top 5 on cross-craft vocabulary
    # ("biseau", "meule" appear in both grinding and glass finishing), leaving
    # the course chunk that actually answers the question at rank 6-8 —
    # invisible to the classifier, which then declared the context
    # insufficient while `generate` had that same chunk in hand.
    # Kept at MAX_CONTEXT_DOCS so the gate judges exactly what generate sees.
    RELEVANCE_MAX_DOCS = 8

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

    def _initialize_langid(self):
        """Load py3langid — see services.translation_service.load_langid."""
        return translation_service.load_langid()

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store, replacing any earlier copy.

        Ids are deterministic (see stable_document_id) and Chroma upserts on
        id, so re-running a sync updates existing documents instead of
        appending duplicates.
        """
        try:
            self.vector_store.add_documents(documents, ids=stable_document_ids(documents))
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

    # ── MMR candidate-pool sizing ──────────────────────────────────────────
    # max_marginal_relevance_search first pulls `fetch_k` nearest neighbours,
    # then greedily picks `k` of them trading relevance off against redundancy.
    # fetch_k is therefore the diversity budget: at fetch_k == k, MMR has
    # nothing to choose between and degenerates into plain similarity search.
    #
    # This call passed no fetch_k at all, so langchain's flat default of 20
    # applied — by accident, not by design. Two consequences:
    #   * against the 16-document annotation collection every single query
    #     logged "Number of requested results 20 is greater than number of
    #     elements in index 16, updating n_results = 16"
    #     (chromadb/segment/impl/vector/local_persistent_hnsw.py:424);
    #   * at k=15 (the configured similarity_search_k) a 20-document pool left
    #     MMR almost no room to diversify, quietly.
    #
    # 5x mirrors langchain's own default ratio (k=4 / fetch_k=20) and the usual
    # dense-retrieval rule of thumb of 4-5x. The absolute cap matters once the
    # corpus grows: fetch_k is how many full 3584-dim embeddings Chroma
    # materialises and hands back per query, so 5 x 15 = 75 of them just to
    # discard 60 is real cost for no measurable diversity gain. Both are then
    # clamped to the collection's live size, which is what actually silences
    # the warning — Chroma compares the requested n_results against the whole
    # segment, not against the cohort-filtered subset.
    MMR_FETCH_K_MULTIPLIER = 5
    MMR_FETCH_K_CAP = 50

    def _collection_count(self) -> Optional[int]:
        """Number of documents in the annotation collection, or None.

        Cheap (a SQLite COUNT), but never load-bearing: any failure, or a
        mocked vector store that returns a non-int, yields None and callers
        simply skip the clamp.
        """
        try:
            count = self.vector_store._collection.count()
        except Exception as e:
            logger.debug(f"Collection count unavailable: {e}")
            return None
        return count if isinstance(count, int) else None

    def _mmr_fetch_k(self, k: int) -> int:
        """MMR candidate-pool size for a request of `k` results.

        See MMR_FETCH_K_MULTIPLIER. Always >= 1 and never above the collection
        size, so a small collection cannot trigger Chroma's over-query warning
        and a large one cannot pull an unbounded candidate set.
        """
        fetch_k = min(max(k * self.MMR_FETCH_K_MULTIPLIER, k), self.MMR_FETCH_K_CAP)
        count = self._collection_count()
        if count is not None and count > 0:
            fetch_k = min(fetch_k, count)
        return max(fetch_k, 1)

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

            results = self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=self._mmr_fetch_k(k), **kwargs
            )
            for doc in results:
                logger.info(f"Document {doc.metadata.get('source', '')}")
                doc_content = str(doc.metadata.get("source"))
                if doc_content not in seen_docs:
                    seen_docs.add(doc_content)
                    unique_results.append(doc)

            logger.info(f"Similarity search returned {len(unique_results)} results")
            return unique_results

        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}", exc_info=True)
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
            user_cohort_ids = state.get("user_cohort_ids")
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
        user_cohort_ids = state.get("user_cohort_ids")
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
            return {"context": [], "video_metadata": []}

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
            video_metadata = self._extract_video_metadata(
                passing,
                limit=state.get("desired_video_count", 1),
                exclude_ids=set(state.get("shown_video_ids") or []),
                preferred_video_id=state.get("referenced_video_id"),
            )
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

        video_metadata = self._extract_video_metadata(
            passing,
            limit=state.get("desired_video_count", 1),
            exclude_ids=set(state.get("shown_video_ids") or []),
            preferred_video_id=state.get("referenced_video_id"),
        )
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

    def assess_relevance(self, state: ConversationState) -> Dict[str, Any]:
        """Pipeline step — after the final retrieval/rerank, before generate.

        One LLM call judges whether the retrieved context actually answers
        the learner's request, so stream_response can skip generation and
        emit a deterministic, consistent message instead of letting
        `generate` decide ad-hoc. That ad-hoc decision was producing
        self-contradicting turns: video cards shown alongside a text
        refusal, non-fixed refusal wording (the LLM paraphrases despite the
        system prompt mandating an exact string), and follow-up questions
        generated even when refusing.

        A high rerank score alone isn't a reliable relevance signal here —
        cross-craft vocabulary (e.g. "biseau"/"meule" grinding technique
        terms also used in glass-finishing) can score >0.9 on a genuinely
        wrong topic. This is a second, independent check.

        Returns one of SUFFICIENT | AMBIGUOUS | INSUFFICIENT. Fails open to
        SUFFICIENT on any LLM error or unparseable response — same
        fail-open philosophy as route_query, so a broken classifier never
        blocks a real answer.
        """
        context_docs = state.get("context") or []

        if not context_docs:
            return {"relevance_assessment": "INSUFFICIENT"}

        if not self.llm:
            return {"relevance_assessment": "SUFFICIENT"}

        query = str(state.get("messages")[-1].content)
        snippets = []
        for i, doc in enumerate(context_docs[: self.RELEVANCE_MAX_DOCS], 1):
            limit = self.RELEVANCE_PREVIEW_CHARS
            preview = doc.page_content[:limit] + "..." if len(doc.page_content) > limit else doc.page_content
            snippets.append(f"[Document {i}]\n{preview}")
        context_text = "\n\n".join(snippets)

        prompt = (
            "Tu es un classificateur de pertinence pour un assistant pédagogique en arts et métiers.\n"
            "On te donne la question de l'apprenti et les documents effectivement récupérés du corpus.\n"
            "Détermine si ces documents permettent réellement de répondre à la question posée.\n\n"
            f'Question : "{query}"\n\n'
            f"Documents récupérés :\n{context_text}\n\n"
            "Réponds avec EXACTEMENT un mot parmi :\n"
            "  SUFFISANT — les documents traitent bien du sujet demandé et permettent de répondre.\n"
            "  AMBIGU — les documents traitent d'un sujet proche du domaine mais pas exactement "
            "celui demandé (ex : une autre technique du même métier), une clarification aiderait.\n"
            "  INSUFFISANT — les documents ne traitent pas du tout du sujet demandé."
        )

        try:
            response = self.llm.invoke(prompt)
            raw = str(response.content).strip().upper()
            # Check French first — this classifier's prompt is otherwise all
            # French, so the LLM naturally answers in French despite being
            # asked for these exact words; English is kept as a fallback.
            # INSUFFISANT/INSUFFICIENT must be checked before
            # SUFFISANT/SUFFICIENT — each contains the other as a substring.
            if "INSUFFISANT" in raw or "INSUFFICIENT" in raw:
                assessment = "INSUFFICIENT"
            elif "AMBIGU" in raw or "AMBIGUOUS" in raw:
                assessment = "AMBIGUOUS"
            elif "SUFFISANT" in raw or "SUFFICIENT" in raw:
                assessment = "SUFFICIENT"
            else:
                logger.warning(f"assess_relevance: unparseable response: {raw!r} — defaulting to SUFFICIENT")
                assessment = "SUFFICIENT"
        except Exception as e:
            logger.error(f"assess_relevance failed: {e} — defaulting to SUFFICIENT")
            assessment = "SUFFICIENT"

        return {"relevance_assessment": assessment}

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

    # ── Orphaned-HNSW-label alarm ──────────────────────────────────────────
    # hnswlib allocates a monotonically increasing label for every element ever
    # added to an index and never reuses one; deleting a document frees its id
    # in Chroma's metadata segment but not its label in the vector segment. A
    # segment whose label high-water mark far exceeds its live count can fail
    # to reload with "Cannot return the results in a contigious 2D array"
    # (chroma-core/chroma#2620; PR #2621 closed unmerged; no chroma-hnswlib
    # release since 0.7.6) — and when it does, every vector query in that
    # process returns nothing, silently, for the life of the process.
    #
    # The annotation collection reached 381 allocated labels against 16
    # addressable documents once, because an older ingest path minted a fresh
    # random UUID per document per restart; a later dedupe deleted ~365 of
    # them and left the labels behind. stable_document_id (top of this module)
    # defused it — the sync is now an upsert in place that allocates no new
    # labels — so this is latent, not active. These thresholds exist so that
    # if it ever comes back it announces itself instead of costing another
    # silent outage.
    #
    # Ratio, not a bare difference: a handful of orphans is normal churn on
    # any collection. The absolute floor keeps a tiny collection (2 alive, 5
    # allocated) from crying wolf.
    HNSW_ORPHAN_MIN_LABELS = 32
    HNSW_ORPHAN_RATIO = 2.0

    def _hnsw_label_stats(self) -> Optional[Tuple[int, int]]:
        """(allocated_labels, live_documents), or None when unavailable.

        Chroma exposes no public API for a segment's label high-water mark, so
        this reads ``_total_elements_added`` off the persistent-HNSW segment —
        private, version-pinned chromadb internals (0.6.3).

        Two deliberate properties:

        * It peeks at the segment manager's *already instantiated* segment and
          never calls ``get_segment()``, which would build one — building a
          persistent HNSW segment reads the metadata pickle and reloads the
          index from disk. If the vector segment has not been used yet there
          is nothing to check, and this returns None rather than doing I/O to
          find that out. That is what keeps it free to call on a startup path.
        * Every step is inside one try/except that degrades to None. A
          chromadb upgrade that moves any of these attributes must cost us the
          diagnostic, never an exception on a query or startup path.

        One honest caveat: ``_total_elements_added`` only advances when a write
        batch is flushed into the HNSW index, while ``count()`` includes the
        pending batch. Right after an ingest the pair can therefore read low
        (a freshly seeded 16-document collection reports 0 allocated / 16
        live). That is fine for what this is — a leak detector, where the
        signal is allocated running far *ahead* of live — but it is not an
        accounting audit, and allocated < live is normal, not a second fault.
        """
        try:
            from chromadb.types import SegmentScope

            collection = self.vector_store._collection
            manager = collection._client._manager
            record = manager.segment_cache[SegmentScope.VECTOR].get(collection.id)
            if record is None:
                return None
            segment = manager._instances.get(record["id"])
            if segment is None:
                return None
            allocated = getattr(segment, "_total_elements_added", None)
            live = collection.count()
            if not isinstance(allocated, int) or not isinstance(live, int):
                return None
            return allocated, live
        except Exception as e:
            logger.debug(f"HNSW label stats unavailable: {e}")
            return None

    def warn_if_hnsw_labels_orphaned(self, context: str = "") -> Optional[Tuple[int, int]]:
        """Log loudly if allocated HNSW labels far exceed live documents.

        Returns the (allocated, live) pair it looked at, or None if the check
        could not run — callers treat None as "no information", not as "fine".
        Never raises; see _hnsw_label_stats.
        """
        stats = self._hnsw_label_stats()
        if stats is None:
            return None

        allocated, live = stats
        orphaned = allocated - live
        where = f" after {context}" if context else ""

        if orphaned < self.HNSW_ORPHAN_MIN_LABELS or allocated < live * self.HNSW_ORPHAN_RATIO:
            if orphaned > 0:
                logger.debug(
                    "HNSW labels%s: %d allocated / %d live (%d orphaned, within normal churn)",
                    where, allocated, live, orphaned,
                )
            return stats

        logger.error(
            "HNSW label leak in collection '%s'%s: %d labels allocated for only %d live "
            "documents (%d orphaned). A segment in this state can fail to reload with "
            "'Cannot return the results in a contigious 2D array', after which every "
            "vector query in the process silently returns nothing. Deleting documents "
            "cannot reclaim these labels (see _clear_annotation_documents); the only "
            "fix is to rebuild the collection offline with the backend stopped.",
            self.config.collection_name, where, allocated, live, orphaned,
        )
        return stats

    def _clear_annotation_documents(self) -> None:
        """Delete every addressable annotation document from the vector store.

        Limitation, stated plainly: this cannot purge orphaned HNSW labels, and
        so cannot repair a collection already suffering from the fault above.
        It asks Chroma for the ids matching ``type=video_annotation`` and
        deletes those — by construction it only ever touches ids Chroma can
        still address. Labels stranded in the HNSW index by earlier deletes
        have no id left to name them, are invisible to ``get()``, and survive
        this call untouched. Calling it in the hope of "resetting" a collection
        that fails with 'Cannot return the results in a contigious 2D array'
        will not help, and adds a fresh round of deletes on top.

        Reclaiming labels requires rebuilding the collection from scratch —
        offline, with the backend stopped, since the local Chroma
        PersistentClient is not process-safe. There is no in-process remedy and
        no upstream fix (chroma-core/chroma#2620).

        What this call can do is *notice*: it reports the label high-water mark
        against the live count afterwards, so a leak is loud in the log rather
        than silent until the next restart.
        """
        try:
            results = self.vector_store.get(where={"type": "video_annotation"})
            if results and "ids" in results and results["ids"]:
                self.vector_store.delete(ids=results["ids"])
                logger.info(
                    f"Cleared {len(results['ids'])} annotation documents from vector store"
                )
            self.warn_if_hnsw_labels_orphaned("clearing annotation documents")
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
        self,
        documents: List[Document],
        limit: int = 1,
        exclude_ids: Optional[set] = None,
        preferred_video_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract video metadata from retrieved documents.

        Looks for video annotation documents and extracts video playback
        information, returning up to `limit` distinct videos (deduplicated
        by video_id), in document order except that `preferred_video_id`
        (if present among the candidates) is moved to the front — used to
        ground a follow-up like "tell me more about the second video" in
        the right one.

        Args:
            documents: List of retrieved documents (just Document objects, not tuples)
            limit: Maximum number of distinct videos to return
            exclude_ids: video_ids to skip entirely (already shown this conversation)
            preferred_video_id: video_id to prioritize to the front, if found

        Returns:
            List of video metadata dicts, empty if none found
        """
        import hashlib

        exclude_ids = exclude_ids or set()
        candidates: List[Dict[str, Any]] = []
        seen_video_ids: set = set()

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

                if video_id in seen_video_ids or video_id in exclude_ids:
                    continue
                seen_video_ids.add(video_id)

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
                candidates.append(video_metadata)

        if preferred_video_id:
            candidates.sort(key=lambda v: v["video_id"] != preferred_video_id)

        results = candidates[:limit]

        for video_metadata in results:
            video_metadata["thumbnail"] = self._generate_video_thumbnail(
                video_metadata["filepath"], video_metadata["start_time"]
            )
            logger.info(f"Extracted video metadata for {video_metadata['filename']}")

        if not results:
            logger.info("No video annotations found in retrieved documents")

        return results

    def _generate_video_thumbnail(
        self,
        filepath: str,
        start_time: float,
        max_width: int = 200,
        jpeg_quality: int = 60,
    ) -> Optional[str]:
        """
        Extract a downscaled JPEG frame from the video at start_time and
        return it as a base64 data URL for embedding directly in the
        video_metadata event, or None if extraction isn't possible.
        """
        try:
            import cv2
        except ImportError:
            logger.warning(
                "opencv-python-headless not installed; skipping video thumbnail"
            )
            return None

        try:
            capture = cv2.VideoCapture(filepath)
            if not capture.isOpened():
                logger.warning(f"Could not open video for thumbnail: {filepath}")
                return None

            fps = capture.get(cv2.CAP_PROP_FPS) or 0
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration_ms = (frame_count / fps * 1000) if fps else 0
            seek_ms = max(0.0, start_time * 1000)
            if duration_ms:
                seek_ms = min(seek_ms, max(duration_ms - 100, 0))

            capture.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
            success, frame = capture.read()

            if not success and seek_ms != 0:
                # Some codecs can only seek to keyframes; fall back to frame 0
                # rather than showing nothing.
                capture.set(cv2.CAP_PROP_POS_MSEC, 0)
                success, frame = capture.read()

            capture.release()

            if not success or frame is None:
                logger.warning(f"Could not read frame for thumbnail: {filepath}")
                return None

            height, width = frame.shape[:2]
            if width > max_width:
                scale = max_width / width
                frame = cv2.resize(
                    frame,
                    (max_width, int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            ok, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not ok:
                return None

            return "data:image/jpeg;base64," + base64.b64encode(buffer).decode(
                "ascii"
            )
        except Exception as e:
            logger.warning(f"Thumbnail generation failed for {filepath}: {e}")
            return None

    # ============================================================================
    # PRF PIPELINE — corpus-grounded query refinement
    # Replaces HyDE as the active retrieval strategy.
    # Graph: retrieve_initial → refine_query_prf → retrieve_final_dual → generate
    # ============================================================================

    def _merge_dedup(
        self, a: List[Document], b: List[Document]
    ) -> List[Document]:
        """Merge two document lists, deduplicating by metadata.source.

        Delegates to merge_dedup_interleaved so that truncating the result
        cannot starve one retrieval source — see that function for why.
        """
        return merge_dedup_interleaved(a, b)

    # Maximum number of docs fed into the LLM context across all sources.
    # Keeps the total prompt well within the Apertus-70B 16 384-token window
    # (system prompt ≈ 450 tok + 8 chunks × 400 tok = 3 650 tok + 1 200 tok output
    # = 4 850 tok, leaving a comfortable margin).
    MAX_CONTEXT_DOCS = 8

    @traceable(name="detect_and_translate_query", run_type="chain")
    def detect_and_translate_query(self, state: ConversationState) -> Dict[str, Any]:
        """Pipeline step 0 — language detection + French translation.

        French queries (the common case) pass through untouched with zero LLM
        calls. Non-French queries get one LLM translation call so every
        downstream retrieval node can keep embedding French text — the corpus
        and refine_query_prf's prompt are both French-only, so this reuses
        that already-tuned pipeline instead of asking it to also handle
        translation.

        Every failure path (langid unavailable, low confidence, short query,
        translation error) degrades to {"query_language": "fr",
        "search_query": <original>} — i.e. today's existing behavior.
        """
        original_query = str(state["messages"][-1].content)

        lang, should_translate = translation_service.decide_translation(
            original_query, self._langid,
            self.config.langid_confidence_threshold, self.config.min_langid_chars,
        )
        if not should_translate:
            return {"query_language": "fr", "search_query": original_query}

        prompt = translation_service.build_query_translation_prompt(original_query, lang)
        translated = translation_service.translate_to_french(prompt, self.llm)
        if translated:
            logger.info(f"detect_and_translate_query: [{lang}] '{original_query}' -> '{translated}'")
        else:
            logger.warning("detect_and_translate_query: empty translation — using original query")

        # query_language is trusted independently of translation success — a
        # failed translation shouldn't also force a French-language answer to
        # a question we know was asked in another language.
        return {"query_language": lang, "search_query": translated or original_query}

    @traceable(name="retrieve_initial", run_type="retriever")
    def parse_query_intent(self, state: ConversationState) -> Dict[str, Any]:
        """Pipeline step 1 (after detect_and_translate_query) — one LLM call
        extracts how many videos were requested, how much depth/detail the
        learner wants, whether this message is a "show me another" pagination
        follow-up, and whether it references a specific previously-shown
        video by ordinal position.

        Defaults and the hard count cap are always applied here in code —
        the LLM output is never trusted directly for the cap.
        """
        query = state.get("search_query") or str(state.get("messages")[-1].content)
        previous_videos = state.get("previous_video_metadata") or []

        desired_video_count = 1
        depth_preference = "normal"
        is_pagination_request = False
        referenced_video_id = None

        if self.llm:
            previous_list_text = "\n".join(
                f"{i + 1}. {v.get('filename', 'unknown')}"
                for i, v in enumerate(previous_videos)
            ) or "(none)"

            prompt = (
                "You are an intent classifier for a vocational-training chat assistant "
                "that can show instructional videos.\n"
                "Given the learner's message, extract:\n"
                "  COUNT: how many videos they're asking for, a digit 1-5 (default 1 if not specified).\n"
                "  DEPTH: brief | normal | detailed — how much detail they want in the answer.\n"
                "  PAGINATION: YES if this message is a follow-up asking for another/more video(s) "
                "on the same topic (e.g. 'show me another one', 'un autre', 'encore une'), NO otherwise.\n"
                "  ORDINAL: which previously-shown video they refer to by position (1, 2, 3...), or NONE.\n\n"
                f"Videos already shown this turn, by position:\n{previous_list_text}\n\n"
                f'Message: "{query}"\n\n'
                "Reply with EXACTLY this format, one line, no explanation:\n"
                "COUNT=<n>;DEPTH=<brief|normal|detailed>;PAGINATION=<YES|NO>;ORDINAL=<n|NONE>"
            )

            try:
                response = self.llm.invoke(prompt)
                raw = str(response.content).strip()
                match = re.match(
                    r"COUNT=(\d+);DEPTH=(brief|normal|detailed);PAGINATION=(YES|NO);ORDINAL=(\d+|NONE)",
                    raw,
                    re.IGNORECASE,
                )
                if match:
                    desired_video_count = int(match.group(1))
                    depth_preference = match.group(2).lower()
                    is_pagination_request = match.group(3).upper() == "YES"
                    ordinal_raw = match.group(4)
                    if ordinal_raw.upper() != "NONE":
                        ordinal = int(ordinal_raw)
                        if 1 <= ordinal <= len(previous_videos):
                            referenced = previous_videos[ordinal - 1]
                            referenced_video_id = referenced.get("video_id") or referenced.get("id")
                else:
                    logger.warning(f"parse_query_intent: could not parse LLM response: {raw!r} — using defaults")
            except Exception as e:
                logger.error(f"parse_query_intent failed: {e} — using defaults")
        else:
            logger.warning("No LLM available for intent parsing — defaulting to count=1")

        desired_video_count = max(1, min(desired_video_count, 5))

        last_topical_query = state.get("last_topical_query") or ""
        if not is_pagination_request:
            last_topical_query = query

        return {
            "desired_video_count": desired_video_count,
            "depth_preference": depth_preference,
            "is_pagination_request": is_pagination_request,
            "referenced_video_id": referenced_video_id,
            "last_topical_query": last_topical_query,
        }

    def retrieve_initial(self, state: ConversationState) -> Dict[str, Any]:
        """PRF step 1 — first-pass retrieval with the raw user query.

        Queries both the video annotation collection and the per-course collection
        (if course_id is present in state).  Results are stored in state["context"]
        for the subsequent PRF reformulation step.  Relevance filtering is handled
        downstream by the cross-encoder rerank node — this step casts a wide net.
        """
        vector_data = self.get_vector_store_data()
        has_annotation_docs = bool(vector_data.get("ids"))

        query = state.get("search_query") or str(state.get("messages")[-1].content)
        if state.get("is_pagination_request") and state.get("last_topical_query"):
            query = state["last_topical_query"]
        course_id = state.get("course_id")

        results: List[Document] = []

        # 1. Video annotations collection.
        user_cohort_ids = state.get("user_cohort_ids")
        # An explicitly clicked domain button wins; otherwise fall back to the
        # craft implied by the course the question was asked from (see
        # stream_response). Keeps annotation retrieval on-craft even when the
        # user never touches the domain selector.
        domain_craft = (
            DOMAIN_MAP.get(state.get("selected_domain"), {}).get("craft")
            or state.get("domain_craft")
        )
        cohort_filter = (
            build_cohort_filter(user_cohort_ids, craft=domain_craft)
            if user_cohort_ids is not None else None
        )
        if has_annotation_docs:
            annotation_results = self.similarity_search(query, k=5, cohort_filter=cohort_filter)
            if not annotation_results:
                # The collection is not empty, so zero hits means retrieval failed rather
                # than genuinely matching nothing — most likely the hnswlib
                # "contigious 2D array" fault (see similarity_search's traceback above),
                # which silently costs every video source for the rest of this process.
                logger.error(
                    "Annotation retrieval returned 0 of %d documents — video sources "
                    "will be missing site-wide until the backend is restarted",
                    len(vector_data.get("ids") or []),
                )
            results.extend(annotation_results)
        else:
            logger.info("Annotation collection empty — skipping annotation retrieval")

        # 2. Course collections — priority course gets k=6, all others k=1.
        if self.course_rag_service:
            enrolled_course_ids = state.get("enrolled_course_ids")
            course_results = self.course_rag_service.similarity_search_all_courses(
                query,
                k_per_course=4,
                priority_course_id=course_id,
                allowed_course_ids=enrolled_course_ids,
            )
            results = self._merge_dedup(results, course_results)
        elif course_id:
            logger.warning("course_id provided but course_rag_service not injected")

        if not results:
            logger.info("retrieve_initial: no documents found")
            return {"context": [], "video_metadata": []}

        results = results[: self.MAX_CONTEXT_DOCS]
        video_metadata = self._extract_video_metadata(
            results,
            limit=state.get("desired_video_count", 1),
            exclude_ids=set(state.get("shown_video_ids") or []),
            preferred_video_id=state.get("referenced_video_id"),
        )
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
        original_query = state.get("search_query") or str(state.get("messages")[-1].content)
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
        refined_query = (
            state.get("refined_query")
            or state.get("search_query")
            or str(state.get("messages")[-1].content)
        )
        if state.get("is_pagination_request") and state.get("last_topical_query"):
            refined_query = state["last_topical_query"]
        course_id = state.get("course_id")

        vector_data = self.get_vector_store_data()
        has_annotation_docs = bool(vector_data.get("ids"))

        annotation_results: List[Document] = []
        user_cohort_ids = state.get("user_cohort_ids")
        # An explicitly clicked domain button wins; otherwise fall back to the
        # craft implied by the course the question was asked from (see
        # stream_response). Keeps annotation retrieval on-craft even when the
        # user never touches the domain selector.
        domain_craft = (
            DOMAIN_MAP.get(state.get("selected_domain"), {}).get("craft")
            or state.get("domain_craft")
        )
        cohort_filter = (
            build_cohort_filter(user_cohort_ids, craft=domain_craft)
            if user_cohort_ids is not None else None
        )

        # 1. Video annotations.
        if has_annotation_docs:
            annotation_results = self.similarity_search(refined_query, k=5, cohort_filter=cohort_filter)

        # 2. Course collections.
        course_results: List[Document] = []
        if self.course_rag_service:
            enrolled_course_ids = state.get("enrolled_course_ids")
            course_results = self.course_rag_service.similarity_search_all_courses(
                refined_query,
                k_per_course=4,
                priority_course_id=course_id,
                allowed_course_ids=enrolled_course_ids,
            )

        results = self._merge_dedup(annotation_results, course_results)

        if not results:
            logger.info("retrieve_final_dual: no documents found with refined query")
            return {"context": [], "video_metadata": []}

        results = results[: self.MAX_CONTEXT_DOCS]
        video_metadata = self._extract_video_metadata(
            results,
            limit=state.get("desired_video_count", 1),
            exclude_ids=set(state.get("shown_video_ids") or []),
            preferred_video_id=state.get("referenced_video_id"),
        )
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
            user_cohort_ids = state.get("user_cohort_ids")
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
            user_cohort_ids = state.get("user_cohort_ids")
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
