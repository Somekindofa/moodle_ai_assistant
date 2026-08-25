"""Unit tests for CourseRAGService's ingestion-time translation."""

from unittest.mock import MagicMock, patch

from langchain_core.documents.base import Document

from config.settings import ConfigurationManager
from services.course_rag_service import CourseRAGService


def _make_service(config_manager=None):
    mock_embeddings = MagicMock()
    return CourseRAGService(
        embeddings=mock_embeddings,
        persist_directory="/tmp/test_chroma",
        config_manager=config_manager,
    )


def _chunks(*texts, heading_path="Safety"):
    return [
        Document(page_content=text, metadata={"heading_path": heading_path, "chunk_index": i})
        for i, text in enumerate(texts)
    ]


# ── Backward compatibility: existing 2-arg construction still works ────

def test_construction_without_config_manager_disables_translation():
    svc = _make_service(config_manager=None)

    assert svc._translation_llm is None

    chunks = _chunks("Wear goggles at all times")
    result = svc._translate_chunks_if_needed(chunks, rag_config=None)
    assert result == chunks


# ── _translate_chunks_if_needed: language detected once, not per chunk ──

def _rag_config(enable_ingestion_translation=True):
    cm = ConfigurationManager()
    cm.get_config().rag.enable_ingestion_translation = enable_ingestion_translation
    return cm.get_config().rag


def test_french_module_is_not_translated():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("fr", 0.95)
    svc._translation_llm = MagicMock()

    chunks = _chunks("Portez des lunettes de protection en tout temps")
    result = svc._translate_chunks_if_needed(chunks, _rag_config())

    assert result[0].page_content == chunks[0].page_content
    assert "source_language" not in result[0].metadata
    svc._translation_llm.invoke.assert_not_called()


def test_english_module_translates_every_chunk():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc._translation_llm = MagicMock()
    responses = [MagicMock(content="Portez des lunettes"), MagicMock(content="Restez prudent")]
    svc._translation_llm.invoke.side_effect = responses

    chunks = _chunks("Wear goggles at all times", "Stay careful around the furnace")
    result = svc._translate_chunks_if_needed(chunks, _rag_config())

    assert result[0].page_content == "Portez des lunettes"
    assert result[1].page_content == "Restez prudent"
    assert result[0].metadata["source_language"] == "en"
    assert result[0].metadata["original_text"] == "Wear goggles at all times"
    assert svc._translation_llm.invoke.call_count == 2


def test_language_detected_once_from_first_chunk_not_per_chunk():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc._translation_llm = MagicMock()
    svc._translation_llm.invoke.return_value = MagicMock(content="traduit")

    chunks = _chunks("First chunk", "Second chunk", "Third chunk")
    svc._translate_chunks_if_needed(chunks, _rag_config())

    svc._langid.classify.assert_called_once()


def test_degenerate_chunk_is_left_untranslated_without_calling_the_llm():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc._translation_llm = MagicMock()
    svc._translation_llm.invoke.return_value = MagicMock(content="Portez des lunettes")

    dot_leader_garbage = "Nom  " + ". . " * 200 + "MINES ParisTech"
    chunks = _chunks("Wear goggles at all times", dot_leader_garbage)
    result = svc._translate_chunks_if_needed(chunks, _rag_config())

    # first (normal) chunk translates; second (degenerate) is passed through
    # untouched, and never reaches the LLM.
    assert result[0].page_content == "Portez des lunettes"
    assert result[1].page_content == dot_leader_garbage
    assert result[1].metadata == chunks[1].metadata
    svc._translation_llm.invoke.assert_called_once()


def test_heading_path_metadata_is_never_translated():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc._translation_llm = MagicMock()
    svc._translation_llm.invoke.return_value = MagicMock(content="Portez des lunettes")

    chunks = _chunks("Wear goggles at all times", heading_path="Safety > Eyewear")
    result = svc._translate_chunks_if_needed(chunks, _rag_config())

    # heading_path is a citation/navigation field surfaced to the frontend —
    # translating page_content must not touch it.
    assert result[0].metadata["heading_path"] == "Safety > Eyewear"


def test_translation_failure_falls_back_to_original_chunk_text():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc._translation_llm = MagicMock()
    svc._translation_llm.invoke.side_effect = Exception("API timeout")

    chunks = _chunks("Wear goggles at all times")
    result = svc._translate_chunks_if_needed(chunks, _rag_config())

    assert result[0].page_content == "Wear goggles at all times"
    assert result[0].metadata["source_language"] == "en"
    assert "original_text" not in result[0].metadata


def test_ingestion_translation_disabled_skips_translation():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc._translation_llm = MagicMock()

    chunks = _chunks("Wear goggles at all times")
    result = svc._translate_chunks_if_needed(chunks, _rag_config(enable_ingestion_translation=False))

    assert result == chunks
    svc._translation_llm.invoke.assert_not_called()


# ── ingest_module wires translation in between chunking and embedding ──

def test_ingest_module_calls_translate_chunks_before_embedding():
    svc = _make_service()
    canned_chunks = _chunks("Wear goggles at all times")
    svc.chunker.chunk_html = MagicMock(return_value=canned_chunks)
    translated_chunks = _chunks("Portez des lunettes")
    svc._translate_chunks_if_needed = MagicMock(return_value=translated_chunks)

    mock_collection = MagicMock()
    svc._get_collection = MagicMock(return_value=mock_collection)

    count = svc.ingest_module(
        course_id="1", module_id="10", module_type="page", module_name="Safety",
        section_name="Intro", content_html="<p>Wear goggles at all times</p>",
    )

    svc._translate_chunks_if_needed.assert_called_once_with(canned_chunks, svc.config_manager.get_config().rag if svc.config_manager else None)
    mock_collection.add_documents.assert_called_once_with(translated_chunks)
    assert count == 1
