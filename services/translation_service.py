"""Shared language-detection + translate-to-French helpers.

Used by:
  - RAGService.detect_and_translate_query (query-side, ephemeral; passes
    self.llm, the shared temperature=0.4 generation client)
  - AnnotationService.annotation_to_documents (ingestion, persisted; passes a
    dedicated temperature=0 client, see build_translation_llm)
  - CourseRAGService.ingest_module (ingestion, persisted; same dedicated client)

No dependency on services.rag_service / services.annotation_service /
services.course_rag_service — importable by all three without a cycle.
"""

import logging
import time
from collections import Counter
from typing import Any, Optional, Tuple

from langchain_openai import ChatOpenAI

from config.settings import ConfigurationManager

logger = logging.getLogger(__name__)


def load_langid():
    """Load py3langid with a normalized-probability identifier, or None on failure.

    The bare module-level `py3langid.classify()` returns unnormalized
    log-probabilities, not a usable [0, 1] confidence — the
    LanguageIdentifier instance with norm_probs=True is required for
    decide_translation's confidence threshold to mean anything.
    """
    try:
        import py3langid as langid
        identifier = langid.langid.LanguageIdentifier.from_pickled_model(
            langid.langid.MODEL_FILE, norm_probs=True
        )
        logger.info("py3langid initialized (normalized probabilities)")
        return identifier
    except Exception as e:
        logger.error(f"py3langid initialization failed: {e} — cross-lingual detection disabled")
        return None


def decide_translation(
    text: str,
    langid_identifier: Any,
    confidence_threshold: float,
    min_chars: int,
) -> Tuple[str, bool]:
    """Decide whether `text` should be translated to French.

    Returns (detected_lang, should_translate). Unavailable identifier,
    detected French, low confidence, or too-short text all default to
    ("fr", False) — biased toward never spuriously translating real French
    content, matching the query-side gating this was extracted from.
    """
    if langid_identifier is None:
        return "fr", False

    lang, confidence = langid_identifier.classify(text)

    if lang == "fr" or confidence < confidence_threshold or len(text) < min_chars:
        return "fr", False

    return lang, True


def is_degenerate_text(text: str, threshold: float = 0.3) -> bool:
    """Detect OCR/text-extraction garbage that shouldn't be sent to an LLM.

    Scanned PDF forms routinely extract as walls of repeated characters —
    dot-leaders on blank fill-in lines (". . . . . . . .") being the case
    that actually hung a backfill run: pathological input like this can
    send a translation call into a slow, repetitive generation loop instead
    of a normal quick response. Heuristic: if one character dominates more
    than `threshold` of the non-whitespace content, it's not real prose.
    """
    stripped = "".join(ch for ch in text if not ch.isspace())
    if not stripped:
        return False
    _, count = Counter(stripped).most_common(1)[0]
    return (count / len(stripped)) > threshold


def build_translation_llm(config_manager: ConfigurationManager) -> ChatOpenAI:
    """A second ChatOpenAI client dedicated to translation calls.

    Ingestion-time translations are persisted permanently, unlike ephemeral
    per-query translations, so determinism matters more here than for the
    shared generation/PRF client (temperature=0.4) — temperature=0, and no
    streaming since this is only ever called via .invoke().
    """
    config = config_manager.get_config().rag
    api_key = config_manager.get_env_var("INFOMANIAK_API_KEY")
    product_id = config_manager.get_env_var("INFOMANIAK_PRODUCT_ID")
    base_url = f"https://api.infomaniak.com/2/ai/{product_id}/openai/v1"
    return ChatOpenAI(
        model=config.llm_model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        streaming=False,
        temperature=0,
        # A stalled request with no timeout blocks forever and never raises
        # — translate_to_french's own retry-with-backoff never even gets a
        # chance to run. request_timeout=60 makes a stuck call fail fast
        # instead; max_retries=0 disables the SDK's own hidden retry layer
        # so translate_to_french's max_retries is the only one in effect —
        # two independent retry layers would make total wait time
        # unpredictable and compound with each other.
        request_timeout=60,
        max_retries=0,
        max_tokens=1200,
        model_kwargs={"tool_choice": "none"},
    )


def extract_text(response: Any) -> str:
    """Normalize a ChatOpenAI response's .content into a plain string."""
    if isinstance(response.content, str):
        return response.content.strip()
    elif isinstance(response.content, list):
        return " ".join(str(item) for item in response.content).strip()
    return str(response.content).strip()


def translate_to_french(prompt: str, llm: ChatOpenAI, max_retries: int = 0) -> Optional[str]:
    """Invoke `llm` with `prompt`, returning the translated text or None.

    Never raises — every failure (API error, empty response) degrades to
    None so callers can fall back to the original, untranslated text.

    `max_retries` defaults to 0 — a single ephemeral query-side translation
    shouldn't add retry latency to a live request. Bulk/sequential callers
    (course chunk translation, the backfill script) pass a higher value,
    since firing many calls back-to-back is exactly what triggers Infomaniak's
    rate limit — a rate-limited call retries with exponential backoff
    (5s, 10s, 20s, ...); any other error fails immediately, same as before.
    """
    delay = 5.0
    attempt = 0
    while True:
        try:
            response = llm.invoke(prompt)
            text = extract_text(response)
            return text or None
        except Exception as e:
            is_rate_limited = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limited and attempt < max_retries:
                attempt += 1
                logger.warning(
                    f"translate_to_french: rate limited, retrying in {delay:.0f}s "
                    f"(attempt {attempt}/{max_retries})"
                )
                time.sleep(delay)
                delay *= 2
                continue
            logger.error(f"translate_to_french: translation failed: {e}")
            return None


def build_query_translation_prompt(original_query: str, source_lang: str) -> str:
    """Byte-identical to the original detect_and_translate_query prompt."""
    return (
        "Traduis la question suivante en français, en conservant tout son sens "
        "technique et son intention.\n\n"
        f'Question originale ({source_lang}) :\n"{original_query}"\n\n'
        "Réponds avec UNIQUEMENT la traduction française, sans explication."
    )


def build_transcript_translation_prompt(transcription: str, source_lang: str) -> str:
    """Translation prompt for a spoken, first-person craft-elicitation transcript."""
    return (
        "Traduis la transcription suivante en français, en conservant tout son sens "
        "technique, son registre oral et son intention. Il s'agit de la transcription "
        "d'un artisan expliquant son geste métier à voix haute — conserve le ton "
        "parlé, à la première personne.\n\n"
        f'Transcription originale ({source_lang}) :\n"{transcription}"\n\n'
        "Réponds avec UNIQUEMENT la traduction française, sans explication."
    )


def build_chunk_translation_prompt(chunk_text: str, source_lang: str) -> str:
    """Translation prompt for a course-content chunk.

    `chunk_text` may already include a heading breadcrumb baked in by
    SemanticChunker (see course_rag_service.py) — translate the whole thing
    as plain text, there's no markup to preserve.
    """
    return (
        "Traduis le contenu pédagogique suivant en français, en conservant tout son "
        "sens technique et sa structure (y compris un éventuel titre de section en "
        "début de texte).\n\n"
        f'Contenu original ({source_lang}) :\n"{chunk_text}"\n\n'
        "Réponds avec UNIQUEMENT la traduction française, sans explication."
    )
