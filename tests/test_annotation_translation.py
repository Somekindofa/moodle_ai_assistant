"""Unit tests for AnnotationService's ingestion-time translation — all
langid/LLM calls mocked except the one real-identifier sanity check."""

from unittest.mock import MagicMock, patch

from config.settings import ConfigurationManager
from services.annotation_service import AnnotationService


def _make_service():
    return AnnotationService(ConfigurationManager())


def _annotation(language=None, transcription="Hello world, this is a test transcript"):
    return {
        "annotation_id": 1,
        "video_id": 10,
        "video_filename": "demo.mp4",
        "video_filepath": "/videos/demo.mp4",
        "start_time": 0.0,
        "end_time": 5.0,
        "audio_filepath": "",
        "source_type": "local",
        "project_name": "Test",
        "annotation_created_at": "2026-01-01",
        "transcription": transcription,
        "language": language,
    }


# ── _BASE_QUERY selects the language column ─────────────────────────────

def test_base_query_selects_language_column():
    assert "a.language AS language" in AnnotationService._BASE_QUERY


# ── __init__ wires up langid + a translation client, degrading safely ──

def test_init_sets_up_langid():
    svc = _make_service()

    assert svc._langid is not None


def test_init_sets_translation_llm_when_construction_succeeds():
    with patch("services.annotation_service.translation_service.build_translation_llm",
               return_value=MagicMock()):
        svc = _make_service()

    assert svc._translation_llm is not None


def test_init_degrades_to_none_when_translation_llm_construction_fails():
    with patch("services.annotation_service.translation_service.build_translation_llm",
               side_effect=RuntimeError("missing API key")):
        svc = _make_service()

    assert svc._translation_llm is None


# ── annotation_to_documents: Whisper-tagged language is trusted directly ──

def test_french_tagged_annotation_is_not_translated():
    svc = _make_service()
    svc._translation_llm = MagicMock()

    docs = svc.annotation_to_documents(_annotation(language="fr", transcription="Bonjour le monde"))

    assert docs[0].page_content == "Bonjour le monde"
    assert "source_language" not in docs[0].metadata
    assert "original_transcription" not in docs[0].metadata
    svc._translation_llm.invoke.assert_not_called()


def test_english_tagged_annotation_is_translated_and_original_preserved():
    svc = _make_service()
    svc._translation_llm = MagicMock()
    response = MagicMock()
    response.content = "Bonjour, ceci est un test de transcription"
    svc._translation_llm.invoke.return_value = response

    docs = svc.annotation_to_documents(_annotation(language="en"))

    assert docs[0].page_content == "Bonjour, ceci est un test de transcription"
    assert docs[0].metadata["source_language"] == "en"
    assert docs[0].metadata["original_transcription"] == "Hello world, this is a test transcript"
    svc._translation_llm.invoke.assert_called_once()


# ── Whisper language is None: fall back to detecting it ourselves ──────

def test_null_language_falls_back_to_langid_detection_non_french():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc._translation_llm = MagicMock()
    response = MagicMock()
    response.content = "Traduction française"
    svc._translation_llm.invoke.return_value = response

    docs = svc.annotation_to_documents(_annotation(language=None))

    assert docs[0].page_content == "Traduction française"
    assert docs[0].metadata["source_language"] == "en"


def test_null_language_falls_back_to_langid_detection_french():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("fr", 0.95)
    svc._translation_llm = MagicMock()

    docs = svc.annotation_to_documents(_annotation(language=None, transcription="Bonjour tout le monde ici"))

    assert docs[0].page_content == "Bonjour tout le monde ici"
    assert "source_language" not in docs[0].metadata
    svc._translation_llm.invoke.assert_not_called()


# ── Failure modes degrade to untranslated indexing, never a crash ──────

def test_translation_failure_indexes_untranslated_but_tags_source_language():
    svc = _make_service()
    svc._translation_llm = MagicMock()
    svc._translation_llm.invoke.side_effect = Exception("API timeout")

    docs = svc.annotation_to_documents(_annotation(language="en"))

    assert docs[0].page_content == "Hello world, this is a test transcript"
    assert docs[0].metadata["source_language"] == "en"
    assert "original_transcription" not in docs[0].metadata


def test_no_translation_llm_available_indexes_untranslated():
    svc = _make_service()
    svc._translation_llm = None

    docs = svc.annotation_to_documents(_annotation(language="en"))

    assert docs[0].page_content == "Hello world, this is a test transcript"


def test_ingestion_translation_disabled_skips_translation_entirely():
    svc = _make_service()
    svc._translation_llm = MagicMock()
    svc.config_manager.get_config().rag.enable_ingestion_translation = False

    docs = svc.annotation_to_documents(_annotation(language="en"))

    assert docs[0].page_content == "Hello world, this is a test transcript"
    assert "source_language" not in docs[0].metadata
    svc._translation_llm.invoke.assert_not_called()
