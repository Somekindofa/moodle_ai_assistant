"""Unit tests for RAGService.detect_and_translate_query — all langid/LLM calls mocked."""

from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage
from config.settings import ConfigurationManager
from services.rag_service import RAGService


def _make_service():
    with patch.object(RAGService, "_initialize_embeddings", return_value=MagicMock()), \
         patch.object(RAGService, "_initialize_vector_store", return_value=MagicMock()), \
         patch.object(RAGService, "_initialize_llm", return_value=MagicMock()), \
         patch.object(RAGService, "_initialize_cross_encoder", return_value=None):
        svc = RAGService(ConfigurationManager())
    return svc


def _state(query):
    return {"messages": [HumanMessage(content=query)]}


def test_french_query_passes_through_with_no_llm_call():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("fr", 0.99)
    svc.llm = MagicMock()

    result = svc.detect_and_translate_query(_state("Comment souffler le verre correctement ?"))

    assert result == {
        "query_language": "fr",
        "search_query": "Comment souffler le verre correctement ?",
    }
    svc.llm.invoke.assert_not_called()


def test_confident_non_french_query_gets_translated():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc.llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Comment souffler le verre ?"
    svc.llm.invoke.return_value = mock_response

    result = svc.detect_and_translate_query(_state("How do you blow glass?"))

    assert result["query_language"] == "en"
    assert result["search_query"] == "Comment souffler le verre ?"
    svc.llm.invoke.assert_called_once()


def test_low_confidence_defaults_to_french():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.2)  # below threshold (0.5)
    svc.llm = MagicMock()

    result = svc.detect_and_translate_query(_state("ok glass work thing"))

    assert result == {"query_language": "fr", "search_query": "ok glass work thing"}
    svc.llm.invoke.assert_not_called()


def test_short_query_defaults_to_french_even_if_confident():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.99)
    svc.llm = MagicMock()

    result = svc.detect_and_translate_query(_state("ok thx"))  # < 12 chars

    assert result == {"query_language": "fr", "search_query": "ok thx"}
    svc.llm.invoke.assert_not_called()


def test_langid_unavailable_is_a_safe_noop():
    svc = _make_service()
    svc._langid = None  # init failed at startup
    svc.llm = MagicMock()

    result = svc.detect_and_translate_query(_state("How do you blow glass?"))

    assert result == {"query_language": "fr", "search_query": "How do you blow glass?"}
    svc.llm.invoke.assert_not_called()


def test_translation_failure_falls_back_to_original_but_keeps_detected_language():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc.llm = MagicMock()
    svc.llm.invoke.side_effect = Exception("API timeout")

    result = svc.detect_and_translate_query(_state("How do you blow glass?"))

    assert result["query_language"] == "en"          # detection is trusted independently
    assert result["search_query"] == "How do you blow glass?"  # falls back, doesn't crash


def test_empty_translation_response_falls_back_to_original():
    svc = _make_service()
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("en", 0.95)
    svc.llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "   "  # blank
    svc.llm.invoke.return_value = mock_response

    result = svc.detect_and_translate_query(_state("How do you blow glass?"))

    assert result["query_language"] == "en"
    assert result["search_query"] == "How do you blow glass?"


# ── retrieve_initial / refine_query_prf / retrieve_final_dual read search_query ──

def test_retrieve_initial_embeds_search_query_when_present():
    svc = _make_service()
    svc.vector_store = MagicMock()
    svc.vector_store.get.return_value = {"ids": ["1"]}
    svc.similarity_search = MagicMock(return_value=[])
    svc.course_rag_service = None

    state = _state("How do you blow glass?")
    state["search_query"] = "Comment souffler le verre ?"
    state["course_id"] = None

    svc.retrieve_initial(state)

    svc.similarity_search.assert_called_once()
    called_query = svc.similarity_search.call_args.args[0]
    assert called_query == "Comment souffler le verre ?"


def test_retrieve_initial_falls_back_to_raw_message_without_search_query():
    svc = _make_service()
    svc.vector_store = MagicMock()
    svc.vector_store.get.return_value = {"ids": ["1"]}
    svc.similarity_search = MagicMock(return_value=[])
    svc.course_rag_service = None

    state = _state("Comment souffler le verre ?")
    state["course_id"] = None
    # no search_query set at all

    svc.retrieve_initial(state)

    called_query = svc.similarity_search.call_args.args[0]
    assert called_query == "Comment souffler le verre ?"


def test_refine_query_prf_grounds_on_search_query_not_raw_message():
    from langchain_core.documents.base import Document

    svc = _make_service()
    svc.llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Requête reformulée en français"
    svc.llm.invoke.return_value = mock_response

    state = _state("How do you blow glass?")
    state["search_query"] = "Comment souffler le verre ?"
    state["context"] = [Document(page_content="soufflage de verre technique", metadata={})]

    svc.refine_query_prf(state)

    prompt_sent = svc.llm.invoke.call_args.args[0]
    assert "Comment souffler le verre ?" in prompt_sent
    assert "How do you blow glass?" not in prompt_sent


def test_retrieve_final_dual_falls_back_to_search_query_before_raw_message():
    svc = _make_service()
    svc.vector_store = MagicMock()
    svc.vector_store.get.return_value = {"ids": ["1"]}
    svc.similarity_search = MagicMock(return_value=[])
    svc.course_rag_service = None

    state = _state("How do you blow glass?")
    state["search_query"] = "Comment souffler le verre ?"
    state["refined_query"] = None  # PRF was skipped
    state["course_id"] = None

    svc.retrieve_final_dual(state)

    called_query = svc.similarity_search.call_args.args[0]
    assert called_query == "Comment souffler le verre ?"


# ── _build_messages answer-language rule ────────────────────────────────────

def test_build_messages_keeps_french_rule_when_query_language_is_fr():
    svc = _make_service()
    state = _state("Comment souffler le verre ?")
    state["query_language"] = "fr"
    state["selected_domain"] = None

    messages = svc._build_messages(state, context_data="contexte")

    system_content = messages[0].content
    assert "Répondez TOUJOURS en français correct et soigné" in system_content


def test_build_messages_keeps_french_rule_when_query_language_absent():
    svc = _make_service()
    state = _state("Comment souffler le verre ?")
    state["selected_domain"] = None
    # query_language not set at all — must default to the French rule

    messages = svc._build_messages(state, context_data="contexte")

    assert "Répondez TOUJOURS en français correct et soigné" in messages[0].content


def test_build_messages_swaps_rule_for_non_french_query():
    svc = _make_service()
    state = _state("How do you blow glass?")
    state["query_language"] = "en"
    state["selected_domain"] = None

    messages = svc._build_messages(state, context_data="context")

    system_content = messages[0].content
    assert "Répondez TOUJOURS en français correct et soigné" not in system_content
    assert "même langue que la question de l'apprenti" in system_content


def test_build_messages_query_tag_always_uses_original_not_search_query():
    svc = _make_service()
    state = _state("How do you blow glass?")
    state["query_language"] = "en"
    state["search_query"] = "Comment souffler le verre ?"  # must NOT leak into <query>
    state["selected_domain"] = None

    messages = svc._build_messages(state, context_data="context")

    human_content = messages[1].content
    assert "How do you blow glass?" in human_content
    assert "Comment souffler le verre ?" not in human_content
