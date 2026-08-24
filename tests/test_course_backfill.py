"""Unit tests for CourseRAGService.backfill_translations — in-place translation
of already-ingested course chunks that predate the ingestion-translation feature."""

from unittest.mock import MagicMock, patch

from config.settings import ConfigurationManager
from services.course_rag_service import CourseRAGService


def _make_service():
    return CourseRAGService(embeddings=MagicMock(), persist_directory="/tmp/test_chroma", config_manager=None)


def _rag_config():
    return ConfigurationManager().get_config().rag


def _collection(ids, documents, metadatas):
    col = MagicMock()
    col.get.return_value = {"ids": ids, "documents": documents, "metadatas": metadatas}
    return col


def test_no_translation_llm_returns_zero_stats_and_touches_nothing():
    svc = _make_service()
    svc._translation_llm = None
    collection = _collection(["1"], ["Wear goggles"], [{}])
    svc._get_collection = MagicMock(return_value=collection)

    stats = svc.backfill_translations("1", _rag_config())

    assert stats == {"total": 0, "already_tagged": 0, "translated": 0, "unchanged_french": 0, "failed": 0}
    collection.update_documents.assert_not_called()


def test_already_tagged_chunks_are_skipped():
    svc = _make_service()
    svc._translation_llm = MagicMock()
    svc._langid = MagicMock()
    collection = _collection(
        ["1"], ["already translated text"], [{"source_language": "en"}],
    )
    svc._get_collection = MagicMock(return_value=collection)

    stats = svc.backfill_translations("1", _rag_config())

    assert stats["total"] == 1
    assert stats["already_tagged"] == 1
    assert stats["translated"] == 0
    svc._translation_llm.invoke.assert_not_called()
    collection.update_documents.assert_not_called()


def test_french_chunk_is_left_untouched_and_stays_untagged():
    svc = _make_service()
    svc._translation_llm = MagicMock()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("fr", 0.95)
    collection = _collection(["1"], ["Portez des lunettes de protection en tout temps"], [{}])
    svc._get_collection = MagicMock(return_value=collection)

    stats = svc.backfill_translations("1", _rag_config())

    assert stats["unchanged_french"] == 1
    assert stats["translated"] == 0
    svc._translation_llm.invoke.assert_not_called()
    collection.update_documents.assert_not_called()


def test_untagged_non_french_chunk_is_translated_and_updated_in_place():
    svc = _make_service()
    svc._translation_llm = MagicMock()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc._translation_llm.invoke.return_value = MagicMock(content="Portez des lunettes")
    collection = _collection(
        ["chunk_1"], ["Wear goggles at all times"], [{"heading_path": "Safety", "chunk_index": 0}],
    )
    svc._get_collection = MagicMock(return_value=collection)

    stats = svc.backfill_translations("1", _rag_config())

    assert stats["translated"] == 1
    collection.update_documents.assert_called_once()
    call_ids, call_docs = collection.update_documents.call_args.args
    assert call_ids == ["chunk_1"]
    assert call_docs[0].page_content == "Portez des lunettes"
    assert call_docs[0].metadata["source_language"] == "en"
    assert call_docs[0].metadata["original_text"] == "Wear goggles at all times"
    assert call_docs[0].metadata["heading_path"] == "Safety"  # untouched, preserved


def test_translation_failure_is_counted_and_chunk_is_not_updated():
    svc = _make_service()
    svc._translation_llm = MagicMock()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc._translation_llm.invoke.side_effect = Exception("API timeout")
    collection = _collection(["chunk_1"], ["Wear goggles at all times"], [{}])
    svc._get_collection = MagicMock(return_value=collection)

    stats = svc.backfill_translations("1", _rag_config())

    assert stats["failed"] == 1
    assert stats["translated"] == 0
    collection.update_documents.assert_not_called()


def test_throttle_sleeps_only_after_real_translation_attempts():
    svc = _make_service()
    svc._translation_llm = MagicMock()
    svc._langid = MagicMock()
    # chunk 1: French (no LLM call), chunk 2: English (real translation attempt)
    svc._langid.classify.side_effect = [("fr", 0.95), ("en", 0.95)]
    svc._translation_llm.invoke.return_value = MagicMock(content="traduit")
    collection = _collection(
        ["1", "2"],
        ["Portez des lunettes en tout temps", "Wear goggles at all times"],
        [{}, {}],
    )
    svc._get_collection = MagicMock(return_value=collection)

    with patch("services.course_rag_service.time.sleep") as mock_sleep:
        stats = svc.backfill_translations("1", _rag_config(), throttle_seconds=0.3)

    assert stats["unchanged_french"] == 1
    assert stats["translated"] == 1
    mock_sleep.assert_called_once_with(0.3)


def test_throttle_defaults_to_no_delay():
    svc = _make_service()
    svc._translation_llm = MagicMock()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc._translation_llm.invoke.return_value = MagicMock(content="traduit")
    collection = _collection(["1"], ["Wear goggles at all times"], [{}])
    svc._get_collection = MagicMock(return_value=collection)

    with patch("services.course_rag_service.time.sleep") as mock_sleep:
        svc.backfill_translations("1", _rag_config())

    mock_sleep.assert_not_called()


def test_updates_are_batched_at_99():
    svc = _make_service()
    svc._translation_llm = MagicMock()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc._translation_llm.invoke.return_value = MagicMock(content="traduit")

    n = 150
    ids = [f"chunk_{i}" for i in range(n)]
    documents = ["Wear goggles at all times" for _ in range(n)]
    metadatas = [{} for _ in range(n)]
    collection = _collection(ids, documents, metadatas)
    svc._get_collection = MagicMock(return_value=collection)

    stats = svc.backfill_translations("1", _rag_config())

    assert stats["translated"] == n
    assert collection.update_documents.call_count == 2
    first_ids, first_docs = collection.update_documents.call_args_list[0].args
    second_ids, second_docs = collection.update_documents.call_args_list[1].args
    assert len(first_ids) == 99
    assert len(second_ids) == 51
