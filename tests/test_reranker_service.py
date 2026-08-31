from config.settings import RAGConfig


def test_rag_config_remote_reranker_defaults():
    cfg = RAGConfig()
    assert cfg.use_remote_reranker is False
    assert cfg.reranker_model == "rerank-multilingual-v3.0"
    assert cfg.remote_reranker_score_threshold == 0.1


import httpx
import respx
import pytest
from langchain_core.documents.base import Document
from services.reranker_service import InfomaniakReranker


def _make_docs(texts):
    return [Document(page_content=t) for t in texts]


@respx.mock
def test_reranker_returns_docs_sorted_by_score():
    docs = _make_docs(["low relevance text", "high relevance text", "medium text"])

    respx.post("https://api.infomaniak.com/2/ai/prod-42/cohere/v2/rerank").mock(
        return_value=httpx.Response(200, json={
            "id": "abc",
            "results": [
                {"index": 1, "relevance_score": 0.92},
                {"index": 2, "relevance_score": 0.45},
                {"index": 0, "relevance_score": 0.03},
            ],
            "meta": {}
        })
    )

    reranker = InfomaniakReranker(
        api_key="test-key", product_id="prod-42",
        model="rerank-multilingual-v3.0", threshold=0.1
    )
    result = reranker.rerank("test query", docs)

    assert len(result) == 2  # index 0 (score 0.03) filtered out
    assert result[0].page_content == "high relevance text"
    assert result[1].page_content == "medium text"


@respx.mock
def test_reranker_logs_all_candidate_scores_not_just_top(caplog):
    """Debugging relevance issues needs every candidate's score, not just the
    winner — the local cross-encoder path already logs the full sorted list;
    the remote path was silently dropping everything but the top score."""
    docs = _make_docs(["doc a", "doc b", "doc c"])

    respx.post("https://api.infomaniak.com/2/ai/prod-42/cohere/v2/rerank").mock(
        return_value=httpx.Response(200, json={
            "id": "abc",
            "results": [
                {"index": 0, "relevance_score": 0.918},
                {"index": 1, "relevance_score": 0.203},
                {"index": 2, "relevance_score": 0.05},
            ],
            "meta": {}
        })
    )

    reranker = InfomaniakReranker(
        api_key="test-key", product_id="prod-42",
        model="rerank-multilingual-v3.0", threshold=0.1
    )
    with caplog.at_level("INFO"):
        reranker.rerank("test query", docs)

    log_text = caplog.text
    assert "0.918" in log_text
    assert "0.203" in log_text
    assert "0.05" in log_text


@respx.mock
def test_reranker_returns_empty_when_all_below_threshold():
    docs = _make_docs(["irrelevant a", "irrelevant b"])

    respx.post("https://api.infomaniak.com/2/ai/prod-42/cohere/v2/rerank").mock(
        return_value=httpx.Response(200, json={
            "id": "xyz",
            "results": [
                {"index": 0, "relevance_score": 0.05},
                {"index": 1, "relevance_score": 0.02},
            ],
            "meta": {}
        })
    )

    reranker = InfomaniakReranker(
        api_key="test-key", product_id="prod-42",
        model="rerank-multilingual-v3.0", threshold=0.1
    )
    result = reranker.rerank("test query", docs)
    assert result == []


@respx.mock
def test_reranker_raises_on_http_error():
    docs = _make_docs(["some doc"])

    respx.post("https://api.infomaniak.com/2/ai/prod-42/cohere/v2/rerank").mock(
        return_value=httpx.Response(401, json={"error": "Unauthorized"})
    )

    reranker = InfomaniakReranker(
        api_key="bad-key", product_id="prod-42",
        model="rerank-multilingual-v3.0", threshold=0.1
    )
    with pytest.raises(RuntimeError, match="Infomaniak reranker API error 401"):
        reranker.rerank("test query", docs)


import respx
import httpx
from unittest.mock import patch, MagicMock
from config.settings import ConfigurationManager, AppConfig, RAGConfig


@respx.mock
def test_rag_service_rerank_uses_remote_when_flag_set(monkeypatch):
    """RAGService.rerank() should call the remote API when use_remote_reranker=True.

    Stubs heavy optional deps via sys.modules and reloads services.rag_service
    against them — both are reverted after the test (monkeypatch restores
    sys.modules; the finally block reloads the module again against the real
    deps) so this doesn't leak a stubbed module into tests that run after it
    in the same session.
    """
    import sys

    # Stub out heavy optional dependencies that are not installed in the test env
    for mod_name in [
        "langchain_chroma", "langchain_chroma.vectorstores",
        "langchain_openai", "langchain_openai.chat_models", "langchain_openai.embeddings",
        "langsmith",
        "sentence_transformers",
        "langchain", "langchain.hub",
        "langchain_core.prompts",
    ]:
        if mod_name not in sys.modules:
            monkeypatch.setitem(sys.modules, mod_name, MagicMock())

    # Provide traceable as a pass-through decorator
    import langsmith
    monkeypatch.setattr(langsmith, "traceable", lambda *args, **kwargs: (lambda fn: fn))

    import services.rag_service  # now importable
    import importlib

    try:
        # Reload in case it was partially imported before stubs were in place
        importlib.reload(services.rag_service)
        from services.rag_service import RAGService
        return _run_remote_rerank_assertions(RAGService)
    finally:
        # sys.modules/langsmith.traceable are back to real once monkeypatch
        # tears down after this test — reload once more so the module object
        # every other test imports (`from services.rag_service import ...`)
        # reflects the real implementation again, not this test's stubs.
        monkeypatch.undo()
        importlib.reload(services.rag_service)


def _run_remote_rerank_assertions(RAGService):

    cfg = RAGConfig(use_remote_reranker=True)
    app_cfg = AppConfig(rag=cfg)

    # Prevent real model loading
    with patch.object(RAGService, "_initialize_cross_encoder", return_value=None), \
         patch.object(RAGService, "_initialize_vector_store", return_value=MagicMock()), \
         patch.object(RAGService, "_initialize_embeddings", return_value=MagicMock()), \
         patch.object(RAGService, "_initialize_llm", return_value=MagicMock()):

        with patch.object(ConfigurationManager, "_load_environment", return_value=None), \
             patch.object(ConfigurationManager, "_validate_environment", return_value=None):
            config_manager = ConfigurationManager(config=app_cfg)
        config_manager.env_vars["INFOMANIAK_API_KEY"] = "test-key"
        config_manager.env_vars["INFOMANIAK_PRODUCT_ID"] = "prod-42"

        svc = RAGService(config_manager)

        # Mock the remote endpoint
        respx.post("https://api.infomaniak.com/2/ai/prod-42/cohere/v2/rerank").mock(
            return_value=httpx.Response(200, json={
                "id": "t1",
                "results": [{"index": 0, "relevance_score": 0.8}],
                "meta": {}
            })
        )

        state = {
            "messages": [MagicMock(content="verre soufflé technique")],
            "context": [Document(page_content="Le soufflage du verre est une technique artisanale.")],
        }
        result = svc.rerank(state)

    assert len(result["context"]) == 1
    assert result["rerank_debug"]["backend"] == "remote"
