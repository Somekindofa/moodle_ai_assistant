"""Tests that course RAG service respects enrolled_course_ids."""

import pytest
from unittest.mock import MagicMock, patch, call


@pytest.fixture
def mock_service():
    """Create a CourseRAGService instance with all dependencies mocked."""
    # This fixture is called after conftest autouse fixtures have run
    from services.course_rag_service import CourseRAGService

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 384
    return CourseRAGService(embeddings=mock_embeddings, persist_directory="/tmp/test_chroma")


def test_similarity_search_all_courses_respects_allowed_list(mock_service):
    """Test that allowed_course_ids filters the courses queried."""
    svc = mock_service
    # Pretend three courses exist in ChromaDB, user is enrolled in two
    with patch.object(svc, "_enumerate_populated_courses", return_value=["1", "2", "3"]), \
         patch.object(svc, "_search_with_embedding", return_value=[]) as mock_search:
        svc.similarity_search_all_courses("query", allowed_course_ids=["1", "3"])

    searched_ids = [call_obj.args[1] for call_obj in mock_search.call_args_list]
    assert "1" in searched_ids
    assert "3" in searched_ids
    assert "2" not in searched_ids


def test_similarity_search_all_courses_no_filter_queries_all(mock_service):
    """Test that None allowed_course_ids queries all courses."""
    svc = mock_service
    with patch.object(svc, "_enumerate_populated_courses", return_value=["1", "2"]), \
         patch.object(svc, "_search_with_embedding", return_value=[]) as mock_search:
        svc.similarity_search_all_courses("query", allowed_course_ids=None)

    searched_ids = [call_obj.args[1] for call_obj in mock_search.call_args_list]
    assert set(searched_ids) == {"1", "2"}
