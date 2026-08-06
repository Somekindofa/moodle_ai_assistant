# Infomaniak Remote Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the local `bge-reranker-v2-m3` cross-encoder (20-30 s on 2-core CPU) with a call to Infomaniak's Cohere-compatible `/cohere/v2/rerank` endpoint, controlled by a config flag so the local path stays available as a fallback.

**Architecture:** A new `InfomaniakReranker` class in `services/reranker_service.py` wraps the HTTP call. `RAGService` instantiates either the local `CrossEncoder` or the remote reranker based on a `RAGConfig` flag; the public `rerank()` method signature is unchanged so `pipeline.py` needs no edits. `httpx` (already installed, v0.28.1) is used for the synchronous HTTP call — the existing `asyncio.to_thread()` wrapper in `pipeline.py` already keeps it off the event loop.

**Tech Stack:** Python 3.11+, `httpx` 0.28.1 (sync client), `pytest` + `respx` for HTTP mocking, Infomaniak Cohere-compatible API at `https://api.infomaniak.com/2/ai/{product_id}/cohere/v2/rerank`.

## Global Constraints

- Config flag defaults to `False` (local reranker) — no behaviour change unless explicitly set.
- `INFOMANIAK_API_KEY` and `INFOMANIAK_PRODUCT_ID` are already loaded by `ConfigurationManager`; do not add new env var names.
- `rerank()` method on `RAGService` must remain synchronous — do not change its signature or the `asyncio.to_thread()` call in `pipeline.py`.
- Threshold semantics differ between backends: BGE scores are raw logits (threshold `0.0`), Cohere scores are calibrated probabilities in [0, 1]. The threshold must be separately configurable per-backend, with a sensible default for the remote path (`0.1`).
- Model name: `rerank-multilingual-v3.0` (Cohere multilingual model, matching the French corpus). **Verify this is available on your Infomaniak product before running** — list available models via the Infomaniak developer portal.
- No emojis, no new inline styles, no new environment variable names beyond what already exists.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| **Modify** | `config/settings.py` | Add `use_remote_reranker`, `reranker_model`, `remote_reranker_score_threshold` to `RAGConfig` |
| **Create** | `services/reranker_service.py` | `InfomaniakReranker` — HTTP call, response parsing, error handling |
| **Modify** | `services/rag_service.py` | Conditional init (local vs remote), dispatch in `rerank()`, remove direct `CrossEncoder` import when remote |
| **Create** | `tests/test_reranker_service.py` | Unit tests for `InfomaniakReranker` with mocked HTTP responses |

---

### Task 1: Add remote-reranker config fields

**Files:**
- Modify: `config/settings.py:34-50`

**Interfaces:**
- Produces: `RAGConfig.use_remote_reranker: bool`, `RAGConfig.reranker_model: str`, `RAGConfig.remote_reranker_score_threshold: float` — consumed by Task 3.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reranker_service.py  (create this file)
from config.settings import RAGConfig

def test_rag_config_remote_reranker_defaults():
    cfg = RAGConfig()
    assert cfg.use_remote_reranker is False
    assert cfg.reranker_model == "rerank-multilingual-v3.0"
    assert cfg.remote_reranker_score_threshold == 0.1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_reranker_service.py::test_rag_config_remote_reranker_defaults -v
```

Expected: `FAILED` — `RAGConfig` has no `use_remote_reranker` attribute.

- [ ] **Step 3: Add fields to `RAGConfig`**

In `config/settings.py`, add three fields inside the `RAGConfig` dataclass after `similarity_search_k`:

```python
    # Remote reranker via Infomaniak Cohere-compatible endpoint
    use_remote_reranker: bool = False
    reranker_model: str = "rerank-multilingual-v3.0"
    # Cohere scores are calibrated probabilities in [0,1]; BGE scores are raw
    # logits where 0.0 is the threshold.  Keep thresholds separate.
    remote_reranker_score_threshold: float = 0.1
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_reranker_service.py::test_rag_config_remote_reranker_defaults -v
```

Expected: `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add config/settings.py tests/test_reranker_service.py
git commit -m "feat: add remote reranker config fields to RAGConfig"
```

---

### Task 2: Build `InfomaniakReranker` service

**Files:**
- Create: `services/reranker_service.py`
- Modify: `tests/test_reranker_service.py`

**Interfaces:**
- Consumes: `api_key: str`, `product_id: str`, `model: str`, `threshold: float`
- Produces: `InfomaniakReranker.rerank(query: str, documents: List[str]) -> List[Document]` — takes the same `docs` list that `RAGService.rerank()` currently uses, returns them sorted and filtered. Consumed by Task 3.

- [ ] **Step 1: Install `respx` for HTTP mocking (if not present)**

```bash
pip show respx 2>/dev/null || pip install respx
```

- [ ] **Step 2: Write failing tests**

Append to `tests/test_reranker_service.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_reranker_service.py -k "reranker" -v
```

Expected: `ERROR` — `services.reranker_service` module not found.

- [ ] **Step 4: Create `services/reranker_service.py`**

```python
"""Infomaniak Cohere-compatible remote reranker."""

import logging
from typing import List

import httpx
from langchain_core.documents.base import Document

logger = logging.getLogger(__name__)


class InfomaniakReranker:
    """Calls Infomaniak's /cohere/v2/rerank endpoint to score and filter documents.

    Scores returned are calibrated probabilities in [0, 1].  Documents below
    `threshold` are dropped; survivors are returned sorted by score descending.
    """

    _ENDPOINT = "https://api.infomaniak.com/2/ai/{product_id}/cohere/v2/rerank"

    def __init__(self, api_key: str, product_id: str, model: str, threshold: float):
        self._api_key = api_key
        self._url = self._ENDPOINT.format(product_id=product_id)
        self._model = model
        self._threshold = threshold

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Return documents filtered by threshold and sorted by relevance score (desc)."""
        if not documents:
            return []

        payload = {
            "model": self._model,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(self._url, json=payload, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(
                f"Infomaniak reranker API error {response.status_code}: {response.text}"
            )

        results = response.json().get("results", [])
        scored = [(r["relevance_score"], documents[r["index"]]) for r in results]
        passing = [(score, doc) for score, doc in scored if score >= self._threshold]
        passing.sort(key=lambda x: x[0], reverse=True)

        logger.info(
            f"remote rerank: {len(documents)} candidates → {len(passing)} passed "
            f"threshold={self._threshold} (top score="
            f"{passing[0][0]:.3f})" if passing else
            f"remote rerank: {len(documents)} candidates → 0 passed threshold={self._threshold}"
        )
        return [doc for _, doc in passing]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_reranker_service.py -k "reranker" -v
```

Expected: all three reranker tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add services/reranker_service.py tests/test_reranker_service.py
git commit -m "feat: add InfomaniakReranker service with httpx and respx tests"
```

---

### Task 3: Wire `InfomaniakReranker` into `RAGService`

**Files:**
- Modify: `services/rag_service.py:16,40,213-245,613-661`

**Interfaces:**
- Consumes: `RAGConfig.use_remote_reranker`, `RAGConfig.reranker_model`, `RAGConfig.remote_reranker_score_threshold` from Task 1; `InfomaniakReranker` from Task 2.
- Produces: `RAGService.rerank()` still returns `{"context": [...], "video_metadata": ..., "rerank_debug": {...}}` — no change to callers.

- [ ] **Step 1: Write a failing integration test**

Append to `tests/test_reranker_service.py`:

```python
import respx
import httpx
from unittest.mock import patch, MagicMock
from config.settings import ConfigurationManager, AppConfig, RAGConfig


@respx.mock
def test_rag_service_rerank_uses_remote_when_flag_set():
    """RAGService.rerank() should call the remote API when use_remote_reranker=True."""
    cfg = RAGConfig(use_remote_reranker=True)
    app_cfg = AppConfig(rag=cfg)

    # Prevent real model loading
    with patch("services.rag_service.RAGService._initialize_cross_encoder") as mock_init, \
         patch("services.rag_service.RAGService._initialize_vector_store", return_value=MagicMock()), \
         patch("services.rag_service.RAGService._initialize_embeddings", return_value=MagicMock()), \
         patch("services.rag_service.RAGService._initialize_llm", return_value=MagicMock()):

        mock_init.return_value = None  # skip local model load

        from config.settings import ConfigurationManager
        config_manager = ConfigurationManager(config=app_cfg)
        config_manager.env_vars["INFOMANIAK_API_KEY"] = "test-key"
        config_manager.env_vars["INFOMANIAK_PRODUCT_ID"] = "prod-42"

        from services.rag_service import RAGService
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_reranker_service.py::test_rag_service_rerank_uses_remote_when_flag_set -v
```

Expected: `FAILED` — `RAGService` does not yet have remote dispatch.

- [ ] **Step 3: Update `rag_service.py` — conditional init**

At the top of `rag_service.py`, add the import alongside the existing `CrossEncoder` import:

```python
# Only imported when use_remote_reranker=False (guarded in _initialize_cross_encoder)
from sentence_transformers import CrossEncoder
from services.reranker_service import InfomaniakReranker
```

Replace `_initialize_cross_encoder` (lines ~224-245) with:

```python
def _initialize_cross_encoder(self):
    """Load local cross-encoder or skip if remote reranker is configured."""
    if self.config_manager.get_config().rag.use_remote_reranker:
        logger.info("Remote reranker configured — skipping local cross-encoder load")
        return None

    try:
        model = CrossEncoder(
            self.CROSS_ENCODER_MODEL,
            device="cpu",
            trust_remote_code=True,
        )
        test_scores = model.predict([("test query", "test document")])
        if float(test_scores[0]) == 0.0:
            raise RuntimeError(
                f"Cross-encoder {self.CROSS_ENCODER_MODEL} returned a zero "
                "score on a sanity-check pair — classification head likely "
                "uninitialised. Check model name and sentence-transformers version."
            )
        logger.info(f"Cross-encoder reranker loaded: {self.CROSS_ENCODER_MODEL}")
        return model
    except Exception as e:
        logger.error(f"Failed to load cross-encoder ({e})")
        raise
```

Also replace the `rerank()` method (lines ~613-661) with:

```python
@traceable(name="rerank", run_type="chain")
def rerank(self, state: ConversationState) -> Dict[str, Any]:
    """Rerank retrieved docs by relevance — local cross-encoder or remote API."""
    query = str(state.get("messages")[-1].content)
    docs = state.get("context", [])

    if not docs:
        return {"context": [], "video_metadata": None}

    rag_cfg = self.config_manager.get_config().rag

    if rag_cfg.use_remote_reranker:
        api_key = self.config_manager.get_env_var("INFOMANIAK_API_KEY")
        product_id = self.config_manager.get_env_var("INFOMANIAK_PRODUCT_ID")
        remote = InfomaniakReranker(
            api_key=api_key,
            product_id=product_id,
            model=rag_cfg.reranker_model,
            threshold=rag_cfg.remote_reranker_score_threshold,
        )
        passing = remote.rerank(query, docs)
        logger.info(
            f"rerank (remote): {len(docs)} candidates → {len(passing)} passed "
            f"threshold={rag_cfg.remote_reranker_score_threshold}"
        )
        video_metadata = self._extract_video_metadata(passing)
        return {
            "context": passing,
            "video_metadata": video_metadata,
            "rerank_debug": {
                "disabled": False,
                "backend": "remote",
                "model": rag_cfg.reranker_model,
                "candidates_in": len(docs),
                "passing_out": len(passing),
                "threshold": rag_cfg.remote_reranker_score_threshold,
            },
        }

    # Local cross-encoder path (unchanged)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = self.cross_encoder.predict(pairs)

    scored_docs = sorted(
        zip(scores, docs), key=lambda x: x[0], reverse=True
    )
    passing = [
        doc for score, doc in scored_docs
        if score >= self.RERANK_SCORE_THRESHOLD
    ]

    top_score = float(scores.max())
    all_scores_sorted = sorted([round(float(s), 4) for s in scores.tolist()], reverse=True)

    logger.info(
        f"rerank (local): {len(docs)} candidates → {len(passing)} passed threshold "
        f"(top score={top_score:.2f}, threshold={self.RERANK_SCORE_THRESHOLD})"
    )

    video_metadata = self._extract_video_metadata(passing)
    return {
        "context": passing,
        "video_metadata": video_metadata,
        "rerank_debug": {
            "disabled": False,
            "backend": "local",
            "candidates_in": len(docs),
            "passing_out": len(passing),
            "threshold": self.RERANK_SCORE_THRESHOLD,
            "top_score": round(top_score, 4),
            "scores": all_scores_sorted,
        },
    }
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/ -v
```

Expected: all tests pass, including the new `test_rag_service_rerank_uses_remote_when_flag_set`.

- [ ] **Step 5: Commit**

```bash
git add services/rag_service.py tests/test_reranker_service.py
git commit -m "feat: wire InfomaniakReranker into RAGService behind use_remote_reranker flag"
```

---

### Task 4: Enable and smoke-test

This task is manual — it activates the flag, confirms the API is reachable, and measures latency improvement.

- [ ] **Step 1: Enable remote reranker in `.env`**

Add or verify in `/opt/craftpilot_backend/.env`:

```
USE_REMOTE_RERANKER=true
```

Then update `RAGConfig` to read the flag from env:

In `config/settings.py`, change the `use_remote_reranker` field default to read from env:

```python
    use_remote_reranker: bool = field(
        default_factory=lambda: os.getenv("USE_REMOTE_RERANKER", "false").lower() == "true"
    )
```

Add `import os` at the top of `config/settings.py` if not present (it already is).

- [ ] **Step 2: Confirm the model name is available**

Check the Infomaniak developer portal for the list of available Cohere rerank models on your product. Confirm `rerank-multilingual-v3.0` is listed, or substitute the correct name in `.env`:

```
RERANKER_MODEL=rerank-multilingual-v3.0
```

Update `RAGConfig.reranker_model` to read from env similarly:

```python
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "rerank-multilingual-v3.0")
    )
```

- [ ] **Step 3: Restart the backend**

Ask the user to run:

```bash
! sudo systemctl restart craftpilot-backend
```

- [ ] **Step 4: Tail the logs and send a test message**

```bash
! sudo journalctl -u craftpilot-backend -f
```

Send a test query through the Moodle chat interface. In the logs, confirm you see:

```
rerank (remote): N candidates → M passed threshold=0.1
```

and that the rerank step completes in under 3 seconds instead of ~25.

- [ ] **Step 5: Commit env-driven config**

```bash
git add config/settings.py
git commit -m "feat: make remote reranker flag and model name env-configurable"
```

---

## Self-Review

**Spec coverage:**
- Remote rerank via Infomaniak Cohere endpoint: Task 2 + Task 3 ✓
- Flag to switch between local and remote: Task 1 + Task 3 ✓
- No behaviour change when flag is off: `use_remote_reranker=False` default ✓
- Threshold handling (different scales, local vs remote): separate `remote_reranker_score_threshold` field ✓
- No new env var names for auth: reuses `INFOMANIAK_API_KEY` + `INFOMANIAK_PRODUCT_ID` ✓
- `rerank()` signature unchanged: `pipeline.py` untouched ✓

**Placeholder scan:** None found — all test bodies, all implementation bodies are complete.

**Type consistency:**
- `InfomaniakReranker.rerank()` takes `List[Document]`, returns `List[Document]` — matches what `RAGService.rerank()` has available in `state["context"]` ✓
- `rerank_debug["backend"]` added in both paths so callers reading that key won't KeyError ✓

**Known open question:** The exact model name available on the Infomaniak product is unverified (requires portal access). Task 4 Step 2 flags this explicitly.
