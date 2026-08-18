from services.rag_service import RAGService
from config.settings import ConfigurationManager
from langchain_core.messages import HumanMessage
from langchain_core.documents.base import Document
from core.types import ConversationState
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest

def test_hyde_generates_document():
    """Test that hyde_generate produces a non-empty Document."""
    mock_llm = Mock()
    mock_llm.invoke.return_value = "This is a fake hypothetical document about glassblowing techniques."

    mock_config = ConfigurationManager()
    rag_service = RAGService(config_manager=mock_config)
    rag_service.llm = mock_llm

    state = ConversationState(
        messages=[HumanMessage(content="How do I fix drooling glass?")],
        context=[],
        video_metadata=None,
        hyde_doc=None
    )

    result = rag_service.hyde_generate(state)

    assert "hyde_doc" in result
    assert isinstance(result["hyde_doc"], Document)
    assert len(result["hyde_doc"].page_content) > 0

    mock_llm.invoke.assert_called_once()


# Tests for _classify_in_domain method
def _make_pipeline_with_mock_llm(response_content: str):
    """Return a MoodleAIAssistantPipeline whose LLM is fully mocked."""
    from pipeline import MoodleAIAssistantPipeline

    with patch.object(MoodleAIAssistantPipeline, "__init__", lambda self, *a, **kw: None):
        pipeline = MoodleAIAssistantPipeline.__new__(MoodleAIAssistantPipeline)

    mock_response = MagicMock()
    mock_response.content = response_content

    mock_bound_llm = MagicMock()
    mock_bound_llm.ainvoke = AsyncMock(return_value=mock_response)

    mock_llm = MagicMock()
    mock_llm.bind.return_value = mock_bound_llm

    mock_rag_service = MagicMock()
    mock_rag_service.llm = mock_llm

    pipeline.rag_service = mock_rag_service
    return pipeline


@pytest.mark.asyncio
async def test_classify_in_domain_returns_true_for_oui():
    pipeline = _make_pipeline_with_mock_llm("OUI")
    assert await pipeline._classify_in_domain("Comment souffler le verre ?") is True


@pytest.mark.asyncio
async def test_classify_in_domain_returns_true_for_yes():
    pipeline = _make_pipeline_with_mock_llm("YES")
    assert await pipeline._classify_in_domain("How do I shape molten glass?") is True


@pytest.mark.asyncio
async def test_classify_in_domain_returns_false_for_non():
    pipeline = _make_pipeline_with_mock_llm("NON")
    assert await pipeline._classify_in_domain("Qui est le président des États-Unis ?") is False


@pytest.mark.asyncio
async def test_classify_in_domain_returns_false_for_no():
    pipeline = _make_pipeline_with_mock_llm("NO")
    assert await pipeline._classify_in_domain("What is the capital of France?") is False


@pytest.mark.asyncio
async def test_classify_in_domain_fails_open_on_exception():
    """Any LLM error must return True (fail-open) so real questions are never blocked."""
    from pipeline import MoodleAIAssistantPipeline

    with patch.object(MoodleAIAssistantPipeline, "__init__", lambda self, *a, **kw: None):
        pipeline = MoodleAIAssistantPipeline.__new__(MoodleAIAssistantPipeline)

    mock_bound_llm = MagicMock()
    mock_bound_llm.ainvoke = AsyncMock(side_effect=Exception("network error"))

    mock_llm = MagicMock()
    mock_llm.bind.return_value = mock_bound_llm

    mock_rag_service = MagicMock()
    mock_rag_service.llm = mock_llm
    pipeline.rag_service = mock_rag_service

    result = await pipeline._classify_in_domain("Qui est Napoleon ?")
    assert result is True


@pytest.mark.asyncio
async def test_stream_response_short_circuits_off_topic():
    """Off-topic question must yield status + refusal token + [DONE] and nothing else."""
    pipeline = _make_pipeline_with_mock_llm("NON")

    events = []
    async for line in pipeline.stream_response(
        message="Qui est le président des États-Unis ?",
        conversation_thread_id="test-thread",
        is_first_message=False,
    ):
        import json as _json
        events.append(_json.loads(line))

    assert events[0] == {"event": "status", "data": "Vérification de la question…"}
    assert events[1]["event"] == "token"
    assert "arts et métiers" in events[1]["data"]
    assert events[2] == {"content": "[DONE]"}
    assert len(events) == 3


@pytest.mark.asyncio
async def test_stream_response_does_not_short_circuit_in_domain():
    """In-domain question must NOT be short-circuited (retrieval pipeline runs)."""
    pipeline = _make_pipeline_with_mock_llm("OUI")

    # Patch the retrieval steps so they don't hit real services
    async def _fake_retrieve(state):
        return {"context": [], "video_metadata": None}

    pipeline.rag_service.config = MagicMock(enable_cross_lingual_detection=True)
    pipeline.rag_service.detect_and_translate_query = MagicMock(return_value={
        "query_language": "fr", "search_query": "Comment souffler le verre ?",
    })
    pipeline.rag_service.retrieve_initial = MagicMock(return_value={
        "context": [], "video_metadata": None, "refined_query": None,
        "hypothetical_document": None, "enhanced_query": None, "query_variants": [],
    })
    pipeline.rag_service.refine_query_prf = MagicMock(return_value={})
    pipeline.rag_service.retrieve_final_dual = MagicMock(return_value={"context": []})
    pipeline.rag_service.rerank = MagicMock(return_value={"context": []})

    async def _fake_stream_generate(state):
        yield "Bonjour"

    pipeline.rag_service.stream_generate = _fake_stream_generate

    events = []
    async for line in pipeline.stream_response(
        message="Comment souffler le verre ?",
        conversation_thread_id="test-thread",
        is_first_message=False,
    ):
        import json as _json
        events.append(_json.loads(line))

    event_types = [e.get("event") or e.get("content") for e in events]
    assert "token" in event_types
    # Must NOT have stopped after 3 events
    assert len(events) > 3


@pytest.mark.asyncio
async def test_stream_response_calls_detect_and_translate_first():
    """detect_and_translate_query must run before retrieve_initial, and its
    output (query_language, search_query) must be visible to the rest of the
    pipeline via state."""
    pipeline = _make_pipeline_with_mock_llm("OUI")
    pipeline.rag_service.config = MagicMock(enable_cross_lingual_detection=True)

    call_order = []

    def _fake_detect(state):
        call_order.append("detect_and_translate_query")
        return {"query_language": "en", "search_query": "Comment souffler le verre ?"}

    def _fake_retrieve_initial(state):
        call_order.append("retrieve_initial")
        assert state["search_query"] == "Comment souffler le verre ?"
        assert state["query_language"] == "en"
        return {
            "context": [], "video_metadata": None, "refined_query": None,
            "hypothetical_document": None, "enhanced_query": None, "query_variants": [],
        }

    pipeline.rag_service.detect_and_translate_query = MagicMock(side_effect=_fake_detect)
    pipeline.rag_service.retrieve_initial = MagicMock(side_effect=_fake_retrieve_initial)
    pipeline.rag_service.refine_query_prf = MagicMock(return_value={})
    pipeline.rag_service.retrieve_final_dual = MagicMock(return_value={"context": []})
    pipeline.rag_service.rerank = MagicMock(return_value={"context": []})

    async def _fake_stream_generate(state):
        yield "Hello"
    pipeline.rag_service.stream_generate = _fake_stream_generate

    events = []
    async for line in pipeline.stream_response(
        message="How do you blow glass?",
        conversation_thread_id="test-thread",
        is_first_message=False,
    ):
        import json as _json
        events.append(_json.loads(line))

    assert call_order == ["detect_and_translate_query", "retrieve_initial"]
    assert {"event": "status", "data": "Traduction de la question…"} in events


@pytest.mark.asyncio
async def test_stream_response_skips_translation_status_for_french():
    pipeline = _make_pipeline_with_mock_llm("OUI")
    pipeline.rag_service.config = MagicMock(enable_cross_lingual_detection=True)

    pipeline.rag_service.detect_and_translate_query = MagicMock(return_value={
        "query_language": "fr", "search_query": "Comment souffler le verre ?",
    })
    pipeline.rag_service.retrieve_initial = MagicMock(return_value={
        "context": [], "video_metadata": None, "refined_query": None,
        "hypothetical_document": None, "enhanced_query": None, "query_variants": [],
    })
    pipeline.rag_service.refine_query_prf = MagicMock(return_value={})
    pipeline.rag_service.retrieve_final_dual = MagicMock(return_value={"context": []})
    pipeline.rag_service.rerank = MagicMock(return_value={"context": []})

    async def _fake_stream_generate(state):
        yield "Bonjour"
    pipeline.rag_service.stream_generate = _fake_stream_generate

    events = []
    async for line in pipeline.stream_response(
        message="Comment souffler le verre ?",
        conversation_thread_id="test-thread",
        is_first_message=False,
    ):
        import json as _json
        events.append(_json.loads(line))

    statuses = [e["data"] for e in events if e.get("event") == "status"]
    assert "Traduction de la question…" not in statuses


@pytest.mark.asyncio
async def test_stream_response_kill_switch_skips_node_entirely():
    pipeline = _make_pipeline_with_mock_llm("OUI")
    pipeline.rag_service.config = MagicMock(enable_cross_lingual_detection=False)

    pipeline.rag_service.detect_and_translate_query = MagicMock()
    pipeline.rag_service.retrieve_initial = MagicMock(return_value={
        "context": [], "video_metadata": None, "refined_query": None,
        "hypothetical_document": None, "enhanced_query": None, "query_variants": [],
    })
    pipeline.rag_service.refine_query_prf = MagicMock(return_value={})
    pipeline.rag_service.retrieve_final_dual = MagicMock(return_value={"context": []})
    pipeline.rag_service.rerank = MagicMock(return_value={"context": []})

    async def _fake_stream_generate(state):
        yield "Bonjour"
    pipeline.rag_service.stream_generate = _fake_stream_generate

    async for _ in pipeline.stream_response(
        message="Comment souffler le verre ?",
        conversation_thread_id="test-thread",
        is_first_message=False,
    ):
        pass

    pipeline.rag_service.detect_and_translate_query.assert_not_called()
