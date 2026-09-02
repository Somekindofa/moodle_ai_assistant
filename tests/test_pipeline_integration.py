from services.rag_service import RAGService
from config.settings import ConfigurationManager
from langchain_core.messages import HumanMessage
from langchain_core.documents.base import Document
from core.types import ConversationState
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest

def _make_rag_service(llm=None):
    """Return a RAGService with __init__ bypassed — for testing node methods
    in isolation without hitting real embeddings/vector-store/LLM init."""
    with patch.object(RAGService, "__init__", lambda self, *a, **kw: None):
        rag_service = RAGService.__new__(RAGService)
    rag_service.llm = llm
    return rag_service


def test_extract_video_metadata_returns_list_with_multiple_videos():
    """_extract_video_metadata should return up to `limit` distinct videos, not just the first."""
    rag_service = _make_rag_service()

    docs = [
        Document(page_content="a", metadata={
            "type": "video_annotation", "video_filepath": "/tmp/a.mp4",
            "video_filename": "a.mp4", "annotation_id": "1",
            "start_time": 0, "end_time": 5, "duration": 5,
        }),
        Document(page_content="b", metadata={
            "type": "video_annotation", "video_filepath": "/tmp/b.mp4",
            "video_filename": "b.mp4", "annotation_id": "2",
            "start_time": 10, "end_time": 15, "duration": 5,
        }),
    ]

    result = rag_service._extract_video_metadata(docs, limit=2)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["filename"] == "a.mp4"
    assert result[1]["filename"] == "b.mp4"


def test_extract_video_metadata_respects_limit():
    rag_service = _make_rag_service()
    docs = [
        Document(page_content="a", metadata={
            "type": "video_annotation", "video_filepath": "/tmp/a.mp4",
            "video_filename": "a.mp4", "annotation_id": "1",
        }),
        Document(page_content="b", metadata={
            "type": "video_annotation", "video_filepath": "/tmp/b.mp4",
            "video_filename": "b.mp4", "annotation_id": "2",
        }),
    ]

    result = rag_service._extract_video_metadata(docs, limit=1)

    assert len(result) == 1
    assert result[0]["filename"] == "a.mp4"


def test_extract_video_metadata_dedups_by_video_id():
    """The same video (same filepath+annotation_id) appearing in multiple chunks counts once."""
    rag_service = _make_rag_service()
    docs = [
        Document(page_content="chunk1", metadata={
            "type": "video_annotation", "video_filepath": "/tmp/a.mp4",
            "video_filename": "a.mp4", "annotation_id": "1",
        }),
        Document(page_content="chunk2", metadata={
            "type": "video_annotation", "video_filepath": "/tmp/a.mp4",
            "video_filename": "a.mp4", "annotation_id": "1",
        }),
    ]

    result = rag_service._extract_video_metadata(docs, limit=5)

    assert len(result) == 1


def test_extract_video_metadata_returns_empty_list_when_none_found():
    rag_service = _make_rag_service()
    docs = [Document(page_content="x", metadata={"type": "course_content"})]

    result = rag_service._extract_video_metadata(docs, limit=3)

    assert result == []


def test_extract_video_metadata_excludes_already_shown_videos():
    rag_service = _make_rag_service()
    docs = [
        Document(page_content="a", metadata={
            "type": "video_annotation", "video_filepath": "/tmp/a.mp4",
            "video_filename": "a.mp4", "annotation_id": "1",
        }),
        Document(page_content="b", metadata={
            "type": "video_annotation", "video_filepath": "/tmp/b.mp4",
            "video_filename": "b.mp4", "annotation_id": "2",
        }),
    ]
    already_shown_id = rag_service._extract_video_metadata(docs[:1], limit=1)[0]["video_id"]

    result = rag_service._extract_video_metadata(docs, limit=5, exclude_ids={already_shown_id})

    assert len(result) == 1
    assert result[0]["filename"] == "b.mp4"


def test_extract_video_metadata_prioritizes_preferred_video():
    rag_service = _make_rag_service()
    docs = [
        Document(page_content="a", metadata={
            "type": "video_annotation", "video_filepath": "/tmp/a.mp4",
            "video_filename": "a.mp4", "annotation_id": "1",
        }),
        Document(page_content="b", metadata={
            "type": "video_annotation", "video_filepath": "/tmp/b.mp4",
            "video_filename": "b.mp4", "annotation_id": "2",
        }),
    ]
    preferred_id = rag_service._extract_video_metadata(docs[1:], limit=1)[0]["video_id"]

    result = rag_service._extract_video_metadata(docs, limit=1, preferred_video_id=preferred_id)

    assert len(result) == 1
    assert result[0]["filename"] == "b.mp4"


def test_retrieve_initial_passes_desired_count_and_exclusions_to_extraction():
    rag_service = _make_rag_service()
    rag_service.course_rag_service = None
    doc = Document(page_content="a", metadata={"type": "video_annotation"})
    rag_service.get_vector_store_data = Mock(return_value={"ids": ["1"]})
    rag_service.similarity_search = Mock(return_value=[doc])
    rag_service._extract_video_metadata = Mock(return_value=[])

    state = ConversationState(
        messages=[HumanMessage(content="deux vidéos sur le verre")],
        search_query="deux vidéos sur le verre",
        desired_video_count=2,
        shown_video_ids=["abc123"],
        referenced_video_id=None,
        is_pagination_request=False,
        last_topical_query="",
        user_cohort_ids=None,
    )

    rag_service.retrieve_initial(state)

    rag_service._extract_video_metadata.assert_called_once()
    _, kwargs = rag_service._extract_video_metadata.call_args
    assert kwargs["limit"] == 2
    assert kwargs["exclude_ids"] == {"abc123"}
    assert kwargs["preferred_video_id"] is None


def test_retrieve_initial_uses_last_topical_query_for_pagination():
    rag_service = _make_rag_service()
    rag_service.course_rag_service = None
    rag_service.get_vector_store_data = Mock(return_value={"ids": ["1"]})
    rag_service.similarity_search = Mock(return_value=[])
    rag_service._extract_video_metadata = Mock(return_value=[])

    state = ConversationState(
        messages=[HumanMessage(content="un autre")],
        search_query="un autre",
        desired_video_count=1,
        shown_video_ids=[],
        referenced_video_id=None,
        is_pagination_request=True,
        last_topical_query="souffler le verre",
        user_cohort_ids=None,
    )

    rag_service.retrieve_initial(state)

    rag_service.similarity_search.assert_called_once()
    args, kwargs = rag_service.similarity_search.call_args
    assert args[0] == "souffler le verre"


def test_retrieve_final_dual_passes_desired_count_and_preferred_to_extraction():
    rag_service = _make_rag_service()
    rag_service.course_rag_service = None
    doc = Document(page_content="a", metadata={"type": "video_annotation"})
    rag_service.get_vector_store_data = Mock(return_value={"ids": ["1"]})
    rag_service.similarity_search = Mock(return_value=[doc])
    rag_service._extract_video_metadata = Mock(return_value=[])

    state = ConversationState(
        messages=[HumanMessage(content="le deuxième")],
        refined_query="souffler le verre",
        desired_video_count=3,
        shown_video_ids=[],
        referenced_video_id="xyz789",
        is_pagination_request=False,
        last_topical_query="",
        user_cohort_ids=None,
    )

    rag_service.retrieve_final_dual(state)

    rag_service._extract_video_metadata.assert_called_once()
    _, kwargs = rag_service._extract_video_metadata.call_args
    assert kwargs["limit"] == 3
    assert kwargs["preferred_video_id"] == "xyz789"


def _make_intent_llm(response_content: str):
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content=response_content)
    return mock_llm


def test_parse_query_intent_extracts_count_and_depth_from_llm_response():
    rag_service = _make_rag_service(
        llm=_make_intent_llm("COUNT=2;DEPTH=detailed;PAGINATION=NO;ORDINAL=NONE")
    )
    state = ConversationState(
        messages=[HumanMessage(content="donne-moi 2 vidéos détaillées sur le verre")],
        search_query="donne-moi 2 vidéos détaillées sur le verre",
    )

    result = rag_service.parse_query_intent(state)

    assert result["desired_video_count"] == 2
    assert result["depth_preference"] == "detailed"
    assert result["is_pagination_request"] is False
    assert result["referenced_video_id"] is None


def test_parse_query_intent_caps_count_at_5():
    rag_service = _make_rag_service(
        llm=_make_intent_llm("COUNT=99;DEPTH=normal;PAGINATION=NO;ORDINAL=NONE")
    )
    state = ConversationState(
        messages=[HumanMessage(content="montre-moi toutes les vidéos")],
        search_query="montre-moi toutes les vidéos",
    )

    result = rag_service.parse_query_intent(state)

    assert result["desired_video_count"] == 5


def test_parse_query_intent_defaults_to_1_when_llm_unavailable():
    rag_service = _make_rag_service(llm=None)
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        search_query="comment souffler le verre ?",
    )

    result = rag_service.parse_query_intent(state)

    assert result["desired_video_count"] == 1
    assert result["depth_preference"] == "normal"
    assert result["is_pagination_request"] is False
    assert result["referenced_video_id"] is None


def test_parse_query_intent_falls_back_to_defaults_on_malformed_llm_response():
    rag_service = _make_rag_service(llm=_make_intent_llm("I'm not sure, sorry!"))
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        search_query="comment souffler le verre ?",
    )

    result = rag_service.parse_query_intent(state)

    assert result["desired_video_count"] == 1
    assert result["depth_preference"] == "normal"
    assert result["is_pagination_request"] is False


def test_parse_query_intent_resolves_ordinal_to_referenced_video_id():
    rag_service = _make_rag_service(
        llm=_make_intent_llm("COUNT=1;DEPTH=normal;PAGINATION=NO;ORDINAL=2")
    )
    state = ConversationState(
        messages=[HumanMessage(content="parle-moi plus de la deuxième vidéo")],
        search_query="parle-moi plus de la deuxième vidéo",
        previous_video_metadata=[
            {"id": "first-id", "filename": "a.mp4"},
            {"id": "second-id", "filename": "b.mp4"},
        ],
    )

    result = rag_service.parse_query_intent(state)

    assert result["referenced_video_id"] == "second-id"


def test_parse_query_intent_ignores_out_of_range_ordinal():
    rag_service = _make_rag_service(
        llm=_make_intent_llm("COUNT=1;DEPTH=normal;PAGINATION=NO;ORDINAL=5")
    )
    state = ConversationState(
        messages=[HumanMessage(content="parle-moi de la cinquième vidéo")],
        search_query="parle-moi de la cinquième vidéo",
        previous_video_metadata=[{"id": "first-id", "filename": "a.mp4"}],
    )

    result = rag_service.parse_query_intent(state)

    assert result["referenced_video_id"] is None


def test_parse_query_intent_sets_last_topical_query_when_not_pagination():
    rag_service = _make_rag_service(
        llm=_make_intent_llm("COUNT=1;DEPTH=normal;PAGINATION=NO;ORDINAL=NONE")
    )
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        search_query="comment souffler le verre ?",
        last_topical_query="ancien sujet",
    )

    result = rag_service.parse_query_intent(state)

    assert result["last_topical_query"] == "comment souffler le verre ?"


def test_parse_query_intent_preserves_last_topical_query_when_pagination():
    rag_service = _make_rag_service(
        llm=_make_intent_llm("COUNT=1;DEPTH=normal;PAGINATION=YES;ORDINAL=NONE")
    )
    state = ConversationState(
        messages=[HumanMessage(content="montre-moi en un autre")],
        search_query="montre-moi en un autre",
        last_topical_query="souffler le verre",
    )

    result = rag_service.parse_query_intent(state)

    assert result["is_pagination_request"] is True
    assert result["last_topical_query"] == "souffler le verre"


def _make_rag_service_for_build_messages():
    rag_service = _make_rag_service()
    rag_service.system_prompt = "SYSTEM"
    rag_service.user_template = "Q: {query}"
    return rag_service


def test_build_messages_appends_brief_depth_instruction():
    rag_service = _make_rag_service_for_build_messages()
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        depth_preference="brief",
        desired_video_count=1,
        video_metadata=[],
    )

    messages = rag_service._build_messages(state, context_data="")

    assert messages[1].content != "Q: comment souffler le verre ?"
    assert "brève" in messages[1].content.lower() or "concis" in messages[1].content.lower()


def test_build_messages_appends_detailed_depth_instruction():
    rag_service = _make_rag_service_for_build_messages()
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        depth_preference="detailed",
        desired_video_count=1,
        video_metadata=[],
    )

    messages = rag_service._build_messages(state, context_data="")

    assert "détaill" in messages[1].content.lower() or "approfond" in messages[1].content.lower()


def test_build_messages_no_depth_instruction_for_normal():
    rag_service = _make_rag_service_for_build_messages()
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        depth_preference="normal",
        desired_video_count=1,
        video_metadata=[{"video_id": "v1"}],
    )

    messages = rag_service._build_messages(state, context_data="")

    assert messages[1].content == "Q: comment souffler le verre ?"


def test_build_messages_appends_undersupply_note_when_fewer_videos_than_requested():
    rag_service = _make_rag_service_for_build_messages()
    state = ConversationState(
        messages=[HumanMessage(content="montre-moi 3 vidéos")],
        depth_preference="normal",
        desired_video_count=3,
        video_metadata=[{"video_id": "v1", "filename": "a.mp4"}],
    )

    messages = rag_service._build_messages(state, context_data="")

    assert "1" in messages[1].content
    assert "3" in messages[1].content


def test_build_messages_no_undersupply_note_when_count_satisfied():
    rag_service = _make_rag_service_for_build_messages()
    state = ConversationState(
        messages=[HumanMessage(content="montre-moi 2 vidéos")],
        depth_preference="normal",
        desired_video_count=2,
        video_metadata=[{"video_id": "v1"}, {"video_id": "v2"}],
    )

    messages = rag_service._build_messages(state, context_data="")

    assert messages[1].content == "Q: montre-moi 2 vidéos"


def test_assess_relevance_returns_sufficient_when_llm_says_so():
    rag_service = _make_rag_service(llm=_make_intent_llm("SUFFICIENT"))
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        context=[Document(page_content="Le soufflage du verre consiste à...")],
    )

    result = rag_service.assess_relevance(state)

    assert result["relevance_assessment"] == "SUFFICIENT"


def test_assess_relevance_returns_insufficient_when_llm_says_so():
    rag_service = _make_rag_service(llm=_make_intent_llm("INSUFFICIENT"))
    state = ConversationState(
        messages=[HumanMessage(content="montre-moi une vidéo de soufflage du verre")],
        context=[Document(page_content="La taille d'un biseau sur une meule consiste à...")],
    )

    result = rag_service.assess_relevance(state)

    assert result["relevance_assessment"] == "INSUFFICIENT"


def test_assess_relevance_returns_ambiguous_when_llm_says_so():
    rag_service = _make_rag_service(llm=_make_intent_llm("AMBIGUOUS"))
    state = ConversationState(
        messages=[HumanMessage(content="montre-moi une vidéo sur le verre")],
        context=[Document(page_content="La taille d'un biseau sur une meule consiste à...")],
    )

    result = rag_service.assess_relevance(state)

    assert result["relevance_assessment"] == "AMBIGUOUS"


def test_assess_relevance_defaults_to_insufficient_when_no_context():
    rag_service = _make_rag_service(llm=_make_intent_llm("SUFFICIENT"))
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        context=[],
    )

    result = rag_service.assess_relevance(state)

    assert result["relevance_assessment"] == "INSUFFICIENT"


def test_assess_relevance_fails_open_to_sufficient_when_llm_unavailable():
    rag_service = _make_rag_service(llm=None)
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        context=[Document(page_content="Le soufflage du verre consiste à...")],
    )

    result = rag_service.assess_relevance(state)

    assert result["relevance_assessment"] == "SUFFICIENT"


def test_assess_relevance_fails_open_to_sufficient_on_llm_exception():
    broken_llm = Mock()
    broken_llm.invoke.side_effect = Exception("network error")
    rag_service = _make_rag_service(llm=broken_llm)
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        context=[Document(page_content="Le soufflage du verre consiste à...")],
    )

    result = rag_service.assess_relevance(state)

    assert result["relevance_assessment"] == "SUFFICIENT"


def test_assess_relevance_accepts_french_insuffisant():
    """The LLM naturally answers in French despite being asked for an English
    keyword (reproduced live: logged as 'unparseable response: INSUFFISANT'
    — defaulted to SUFFICIENT, which is exactly wrong here)."""
    rag_service = _make_rag_service(llm=_make_intent_llm("INSUFFISANT"))
    state = ConversationState(
        messages=[HumanMessage(content="montre-moi une vidéo de soufflage du verre")],
        context=[Document(page_content="La taille d'un biseau sur une meule consiste à...")],
    )

    result = rag_service.assess_relevance(state)

    assert result["relevance_assessment"] == "INSUFFICIENT"


def test_assess_relevance_accepts_french_ambigu():
    rag_service = _make_rag_service(llm=_make_intent_llm("AMBIGU"))
    state = ConversationState(
        messages=[HumanMessage(content="montre-moi une vidéo sur le verre")],
        context=[Document(page_content="La taille d'un biseau sur une meule consiste à...")],
    )

    result = rag_service.assess_relevance(state)

    assert result["relevance_assessment"] == "AMBIGUOUS"


def test_assess_relevance_accepts_french_suffisant():
    rag_service = _make_rag_service(llm=_make_intent_llm("SUFFISANT"))
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        context=[Document(page_content="Le soufflage du verre consiste à...")],
    )

    result = rag_service.assess_relevance(state)

    assert result["relevance_assessment"] == "SUFFICIENT"


def test_assess_relevance_fails_open_to_sufficient_on_unparseable_response():
    rag_service = _make_rag_service(llm=_make_intent_llm("je ne sais pas trop"))
    state = ConversationState(
        messages=[HumanMessage(content="comment souffler le verre ?")],
        context=[Document(page_content="Le soufflage du verre consiste à...")],
    )

    result = rag_service.assess_relevance(state)

    assert result["relevance_assessment"] == "SUFFICIENT"


def test_assess_relevance_prompt_includes_answer_past_300_chars():
    """Reproduces a live bug (2026-09-02): a real ingested chunk was 672
    chars long — well within SemanticChunker's ~1600-char target
    (TARGET_TOKENS=400, course_rag_service.py) — but the fact that actually
    answered the learner's question ("apprentices must never work alone
    near the glory hole") sat past character 300. assess_relevance's
    snippet preview truncates each document to doc.page_content[:300],
    so the classifier LLM never saw it and misjudged well-matched,
    correctly-translated content as insufficient, even though rerank
    scored it 0.92+. The preview window must cover a full target-size
    chunk, not an arbitrary short slice."""
    llm = _make_intent_llm("SUFFICIENT")
    rag_service = _make_rag_service(llm=llm)
    filler = "Le soufflage du verre nécessite des précautions de sécurité. " * 6
    assert len(filler) > 300  # sanity: filler alone already exceeds the truncation point
    # Marker deliberately absent from the query — the query is phrased
    # generically so the only way it can appear in the prompt sent to the
    # LLM is via the (possibly truncated) document preview, not by leaking
    # in from the question text itself.
    marker = "MARKER_ANSWER_TEXT_APPRENTIS_NE_DOIVENT_JAMAIS_TRAVAILLER_SEULS"
    page_content = filler + marker
    state = ConversationState(
        messages=[HumanMessage(content="Pourquoi les apprentis ne doivent-ils jamais travailler seuls ?")],
        context=[Document(page_content=page_content)],
    )

    rag_service.assess_relevance(state)

    prompt_sent_to_llm = llm.invoke.call_args[0][0]
    assert marker in prompt_sent_to_llm, (
        "assess_relevance truncated the document before the classifier ever "
        "saw the answer-bearing text — it will misjudge this as INSUFFISANT "
        "regardless of what the LLM does with what little it received."
    )


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
async def test_stream_response_calls_parse_query_intent_and_emits_intent_event():
    pipeline = _make_pipeline_with_mock_llm("OUI")
    pipeline.rag_service.config = MagicMock(enable_cross_lingual_detection=False)

    pipeline.rag_service.parse_query_intent = MagicMock(return_value={
        "desired_video_count": 1, "depth_preference": "normal",
        "is_pagination_request": True, "referenced_video_id": None,
        "last_topical_query": "souffler le verre",
    })
    pipeline.rag_service.retrieve_initial = MagicMock(return_value={
        "context": [], "video_metadata": [], "refined_query": None,
        "hypothetical_document": None, "enhanced_query": None, "query_variants": [],
    })
    pipeline.rag_service.refine_query_prf = MagicMock(return_value={})
    pipeline.rag_service.retrieve_final_dual = MagicMock(return_value={"context": []})
    pipeline.rag_service.rerank = MagicMock(return_value={"context": [], "video_metadata": []})

    async def _fake_stream_generate(state):
        yield "Bonjour"
    pipeline.rag_service.stream_generate = _fake_stream_generate

    events = []
    async for line in pipeline.stream_response(
        message="un autre",
        conversation_thread_id="test-thread",
        is_first_message=False,
    ):
        import json as _json
        events.append(_json.loads(line))

    pipeline.rag_service.parse_query_intent.assert_called_once()
    intent_events = [e for e in events if e.get("event") == "intent"]
    assert len(intent_events) == 1
    assert intent_events[0]["data"]["is_pagination_request"] is True


@pytest.mark.asyncio
async def test_stream_response_seeds_state_from_previous_sources_and_message():
    pipeline = _make_pipeline_with_mock_llm("OUI")
    pipeline.rag_service.config = MagicMock(enable_cross_lingual_detection=False)

    captured_state = {}

    def _fake_parse_intent(state):
        captured_state.update(state)
        return {
            "desired_video_count": 1, "depth_preference": "normal",
            "is_pagination_request": False, "referenced_video_id": None,
            "last_topical_query": "x",
        }

    pipeline.rag_service.parse_query_intent = MagicMock(side_effect=_fake_parse_intent)
    pipeline.rag_service.retrieve_initial = MagicMock(return_value={
        "context": [], "video_metadata": [], "refined_query": None,
        "hypothetical_document": None, "enhanced_query": None, "query_variants": [],
    })
    pipeline.rag_service.refine_query_prf = MagicMock(return_value={})
    pipeline.rag_service.retrieve_final_dual = MagicMock(return_value={"context": []})
    pipeline.rag_service.rerank = MagicMock(return_value={"context": [], "video_metadata": []})

    async def _fake_stream_generate(state):
        yield "Bonjour"
    pipeline.rag_service.stream_generate = _fake_stream_generate

    previous_sources = [{"id": "abc123", "filename": "a.mp4"}]
    async for _ in pipeline.stream_response(
        message="un autre",
        conversation_thread_id="test-thread",
        is_first_message=False,
        previous_sources=previous_sources,
        previous_message="souffler le verre",
    ):
        pass

    assert captured_state["previous_video_metadata"] == previous_sources
    assert captured_state["shown_video_ids"] == ["abc123"]
    assert captured_state["last_topical_query"] == "souffler le verre"


@pytest.mark.asyncio
async def test_stream_response_emits_one_event_per_video():
    pipeline = _make_pipeline_with_mock_llm("OUI")
    pipeline.rag_service.config = MagicMock(enable_cross_lingual_detection=False)

    pipeline.rag_service.parse_query_intent = MagicMock(return_value={
        "desired_video_count": 2, "depth_preference": "normal",
        "is_pagination_request": False, "referenced_video_id": None,
        "last_topical_query": "x",
    })
    pipeline.rag_service.retrieve_initial = MagicMock(return_value={
        "context": [], "video_metadata": [], "refined_query": None,
        "hypothetical_document": None, "enhanced_query": None, "query_variants": [],
    })
    pipeline.rag_service.refine_query_prf = MagicMock(return_value={})
    pipeline.rag_service.retrieve_final_dual = MagicMock(return_value={"context": []})
    pipeline.rag_service.rerank = MagicMock(return_value={
        "context": [],
        "video_metadata": [
            {"video_id": "v1", "filename": "a.mp4"},
            {"video_id": "v2", "filename": "b.mp4"},
        ],
    })

    async def _fake_stream_generate(state):
        yield "Bonjour"
    pipeline.rag_service.stream_generate = _fake_stream_generate

    events = []
    async for line in pipeline.stream_response(
        message="2 vidéos",
        conversation_thread_id="test-thread",
        is_first_message=False,
    ):
        import json as _json
        events.append(_json.loads(line))

    video_events = [e for e in events if e.get("event") == "video_metadata"]
    assert len(video_events) == 2
    assert video_events[0]["data"]["filename"] == "a.mp4"
    assert video_events[1]["data"]["filename"] == "b.mp4"


def _mock_common_pipeline_nodes(pipeline, video_metadata=None):
    pipeline.rag_service.parse_query_intent = MagicMock(return_value={
        "desired_video_count": 1, "depth_preference": "normal",
        "is_pagination_request": False, "referenced_video_id": None,
        "last_topical_query": "x",
    })
    pipeline.rag_service.retrieve_initial = MagicMock(return_value={
        "context": [Document(page_content="pertinent")], "video_metadata": [],
        "refined_query": None, "hypothetical_document": None,
        "enhanced_query": None, "query_variants": [],
    })
    pipeline.rag_service.refine_query_prf = MagicMock(return_value={})
    pipeline.rag_service.retrieve_final_dual = MagicMock(return_value={
        "context": [Document(page_content="pertinent")],
    })
    pipeline.rag_service.rerank = MagicMock(return_value={
        "context": [Document(page_content="pertinent")],
        "video_metadata": video_metadata or [],
    })


@pytest.mark.asyncio
async def test_stream_response_sufficient_relevance_proceeds_normally():
    pipeline = _make_pipeline_with_mock_llm("OUI")
    pipeline.rag_service.config = MagicMock(enable_cross_lingual_detection=False)
    _mock_common_pipeline_nodes(pipeline, video_metadata=[{"video_id": "v1", "filename": "a.mp4"}])
    pipeline.rag_service.assess_relevance = MagicMock(return_value={"relevance_assessment": "SUFFICIENT"})

    async def _fake_stream_generate(state):
        yield "Bonjour"
    pipeline.rag_service.stream_generate = _fake_stream_generate

    events = []
    async for line in pipeline.stream_response(
        message="comment souffler le verre ?", conversation_thread_id="t", is_first_message=False,
    ):
        import json as _json
        events.append(_json.loads(line))

    assert any(e.get("event") == "video_metadata" for e in events)
    assert any(e.get("data") == "Bonjour" for e in events)
    assert any(e.get("event") == "documents" for e in events)


@pytest.mark.asyncio
async def test_stream_response_insufficient_relevance_skips_videos_and_generation():
    pipeline = _make_pipeline_with_mock_llm("OUI")
    pipeline.rag_service.config = MagicMock(enable_cross_lingual_detection=False)
    _mock_common_pipeline_nodes(pipeline, video_metadata=[{"video_id": "v1", "filename": "biseau.mp4"}])
    pipeline.rag_service.assess_relevance = MagicMock(return_value={"relevance_assessment": "INSUFFICIENT"})
    pipeline.rag_service.INSUFFICIENT_CONTEXT_MESSAGE = "Je n'ai pas trouvé d'information pertinente."

    stream_generate_called = False
    async def _fake_stream_generate(state):
        nonlocal stream_generate_called
        stream_generate_called = True
        yield "should not run"
    pipeline.rag_service.stream_generate = _fake_stream_generate

    events = []
    async for line in pipeline.stream_response(
        message="montre-moi 2 vidéos sur le soufflage du verre",
        conversation_thread_id="t", is_first_message=False,
    ):
        import json as _json
        events.append(_json.loads(line))

    assert stream_generate_called is False
    assert not any(e.get("event") == "video_metadata" for e in events)
    assert not any(e.get("event") == "documents" for e in events)
    token_events = [e for e in events if e.get("event") == "token"]
    assert len(token_events) == 1
    assert token_events[0]["data"] == "Je n'ai pas trouvé d'information pertinente."
    assert events[-1] == {"content": "[DONE]"}


@pytest.mark.asyncio
async def test_stream_response_ambiguous_relevance_skips_videos_and_generation():
    pipeline = _make_pipeline_with_mock_llm("OUI")
    pipeline.rag_service.config = MagicMock(enable_cross_lingual_detection=False)
    _mock_common_pipeline_nodes(pipeline, video_metadata=[{"video_id": "v1", "filename": "biseau.mp4"}])
    pipeline.rag_service.assess_relevance = MagicMock(return_value={"relevance_assessment": "AMBIGUOUS"})

    stream_generate_called = False
    async def _fake_stream_generate(state):
        nonlocal stream_generate_called
        stream_generate_called = True
        yield "should not run"
    pipeline.rag_service.stream_generate = _fake_stream_generate

    events = []
    async for line in pipeline.stream_response(
        message="montre-moi une vidéo sur le verre",
        conversation_thread_id="t", is_first_message=False,
    ):
        import json as _json
        events.append(_json.loads(line))

    assert stream_generate_called is False
    assert not any(e.get("event") == "video_metadata" for e in events)
    token_events = [e for e in events if e.get("event") == "token"]
    assert len(token_events) == 1
    assert events[-1] == {"content": "[DONE]"}


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
