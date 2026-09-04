"""Tests that similarity_search passes cohort filter to ChromaDB."""

import sys
import types
import pytest
from unittest.mock import MagicMock, patch

# Stub modules that are unavailable in the test environment before any imports.
for _mod in ("langchain", "langchain.hub", "langchain_chroma", "sentence_transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Provide the `hub` attribute on the langchain stub so `from langchain import hub` works.
sys.modules["langchain"].hub = sys.modules["langchain.hub"]  # type: ignore[attr-defined]

# Provide Chroma class stub so `from langchain_chroma import Chroma` works.
sys.modules["langchain_chroma"].Chroma = MagicMock  # type: ignore[attr-defined]

# Provide CrossEncoder stub for sentence_transformers.
sys.modules["sentence_transformers"].CrossEncoder = MagicMock  # type: ignore[attr-defined]

from langchain_core.documents.base import Document
from services.rag_service import RAGService, build_cohort_filter


def _make_service():
    mock_config = MagicMock()
    mock_config.get_config.return_value.rag.similarity_search_k = 5
    with patch.object(RAGService, "_initialize_embeddings", return_value=MagicMock()), \
         patch.object(RAGService, "_initialize_vector_store", return_value=MagicMock()), \
         patch.object(RAGService, "_initialize_llm", return_value=MagicMock()), \
         patch.object(RAGService, "_initialize_cross_encoder", return_value=MagicMock()):
        svc = RAGService(mock_config)
    svc.vector_store = MagicMock()
    svc.vector_store.max_marginal_relevance_search.return_value = []
    return svc


def test_similarity_search_no_filter_when_no_cohorts():
    svc = _make_service()
    svc.similarity_search("hello", cohort_filter=None)
    call_kwargs = svc.vector_store.max_marginal_relevance_search.call_args
    assert call_kwargs.kwargs.get("filter") is None or "filter" not in (call_kwargs.kwargs or {})


def test_similarity_search_passes_cohort_filter():
    svc = _make_service()
    cohort_filter = {"$or": [{"cohort_id": {"$in": [7, 42]}}, {"open_access": True}]}
    svc.similarity_search("hello", cohort_filter=cohort_filter)
    call_kwargs = svc.vector_store.max_marginal_relevance_search.call_args
    assert call_kwargs.kwargs.get("filter") == cohort_filter


def test_build_cohort_filter_with_cohorts():
    f = build_cohort_filter([1, 2])
    assert f == {"$or": [{"cohort_id": {"$in": [1, 2]}}, {"open_access": True}]}


def test_build_cohort_filter_no_cohorts_returns_open_only():
    f = build_cohort_filter([])
    assert f == {"open_access": True}


def test_build_cohort_filter_with_craft_ands_craft_condition():
    f = build_cohort_filter([1, 2], craft="glassblowing")
    assert f == {
        "$and": [
            {"$or": [{"cohort_id": {"$in": [1, 2]}}, {"open_access": True}]},
            {"craft": "glassblowing"},
        ]
    }


def test_build_cohort_filter_no_craft_unchanged():
    f = build_cohort_filter([1, 2], craft=None)
    assert f == {"$or": [{"cohort_id": {"$in": [1, 2]}}, {"open_access": True}]}


def test_retrieve_no_filter_when_cohorts_not_in_state():
    """When user_cohort_ids is absent from state, no filter reaches ChromaDB."""
    from langchain_core.messages import HumanMessage
    svc = _make_service()
    state = {
        "messages": [HumanMessage(content="test query")],
        "query_variants": [],
        "user_cohort_ids": None,   # explicitly not set
    }
    svc.get_vector_store_data = MagicMock(return_value={"ids": ["x"]})
    svc.retrieve(state)
    call_kwargs = svc.vector_store.max_marginal_relevance_search.call_args
    assert call_kwargs is None or "filter" not in (call_kwargs.kwargs or {})
