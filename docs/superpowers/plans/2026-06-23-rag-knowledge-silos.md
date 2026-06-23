# RAG Knowledge Silos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce Moodle cohort-level access boundaries so RAG retrieval is pre-filtered to content the requesting user is authorised to see — no restricted content is ever fetched for an unauthorised user.

**Architecture:** A new `SiloService` queries the Moodle MySQL DB for a user's cohort memberships and course enrolments, caches results for 60 s, and returns the allowed scope. The CraftPilot pipeline injects this scope into the initial `ConversationState`; retrieve nodes pass a `filter` clause to ChromaDB so only matching documents are returned. Video annotations are tagged at ingest time with the project's `cohort_id`; the Video Elicitation Tool UI lets experts set this cohort per project.

**Tech Stack:** Python 3.11 · FastAPI · LangChain + ChromaDB · pymysql · SQLite (video elicitation) · PHP 8 + Moodle 4.x AMD JS

## Global Constraints

- Moodle DB connection: `pymysql` with host `localhost`, user `moodleuser`, password from `os.getenv("MOODLE_DB_PASSWORD")`, database `moodle` — match pattern already in `api/routes.py:399-407`
- ChromaDB metadata values must be scalars (`str`, `int`, `float`, `bool`) — no lists
- Sentinel for "open access" annotation in ChromaDB: `cohort_id = -1`, `open_access = True`
- Silo filter MUST include `{"open_access": True}` branch so un-restricted content stays visible to all
- Video elicitation backend URL: `http://127.0.0.1:8005` · CraftPilot backend URL: `http://127.0.0.1:8000`
- CraftPilot ingest endpoint: `POST http://127.0.0.1:8000/api/ingest-annotation` (requires `X-Internal-Token` header)
- All tests run from `/opt/craftpilot_backend/` with `pytest`
- Never commit `.env` files or secrets
- Frequent small commits — one per task minimum

---

## File Map

### CraftPilot backend (`/opt/craftpilot_backend/`)
| Action | Path | Responsibility |
|---|---|---|
| Create | `services/silo_service.py` | Moodle DB cohort/enrolment lookup with 60 s cache |
| Create | `tests/test_silo_service.py` | Unit tests for SiloService (mocked DB) |
| Modify | `core/types.py` | Add `user_cohort_ids` + `enrolled_course_ids` to ConversationState |
| Modify | `api/models.py` | Add `user_id` to ChatRequest; `allowed_cohort_id` to AnnotationIngestRequest |
| Modify | `api/routes.py` | 403 guard; pass `user_id` to stream_response; resync-project-annotations endpoint |
| Modify | `pipeline.py` | Accept `user_id`; call SiloService; inject scope into initial state |
| Modify | `services/annotation_service.py` | Add `cohort_id` + `open_access` to ChromaDB document metadata |
| Modify | `services/rag_service.py` | Accept `cohort_filter` in `similarity_search`; apply in all retrieve nodes |
| Modify | `services/course_rag_service.py` | Filter `_enumerate_populated_courses` to `allowed_course_ids` |

### Video Elicitation Tool (`/opt/video_elicitation_annotation_tool/`)
| Action | Path | Responsibility |
|---|---|---|
| Modify | `backend/migration.py` | Migration: add `allowed_cohort_id` to `projects` table |
| Modify | `backend/models.py` | Add `allowed_cohort_id` to Project, ProjectCreate, ProjectUpdate, ProjectResponse |
| Modify | `backend/main.py` | `/api/cohorts/managed`; trigger ChromaDB resync on project cohort update |

### Video Elicitation frontend (`/opt/video_elicitation_annotation_tool/`)
| Action | Path | Responsibility |
|---|---|---|
| Modify | `js/app.js` | Cohort selector in project create/edit modal; dismissible silo banner |

### Moodle plugin: videoelicit (`/var/www/html/public/local/videoelicit/`)
| Action | Path | Responsibility |
|---|---|---|
| Modify | `settings.php` | Add Knowledge Silo section + `silo_contact_email` field |
| Modify | `lang/en/local_videoelicit.php` | Lang strings for new settings |
| Modify | `classes/jwt_helper.php` | Embed `silo_contact_email` in JWT payload |
| Modify | `index.php` | Read setting and pass to `create_token` |

### Moodle plugin: craftpilot (`/var/www/html/public/local/craftpilot/`)
| Action | Path | Responsibility |
|---|---|---|
| Modify | `chat_proxy.php` | Cross-check `user_id` in body against `$USER->id`; 403 on mismatch |
| Modify | `amd/src/chat_interface.js` | Add `user_id: M.cfg.userid` to POST payload |

---

## Task 1: SiloService — cohort and enrolment lookup

**Files:**
- Create: `services/silo_service.py`
- Create: `tests/test_silo_service.py`

**Interfaces:**
- Produces:
  - `SiloService.get_allowed_cohorts(user_id: int) -> list[int]`
  - `SiloService.get_enrolled_course_ids(user_id: int) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `/opt/craftpilot_backend/tests/test_silo_service.py`:

```python
"""Unit tests for SiloService — all DB calls are mocked."""

import time
import pytest
from unittest.mock import MagicMock, patch


def _make_service():
    from services.silo_service import SiloService
    return SiloService(db_password="test")


def _mock_cursor(rows):
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    return cursor


def _mock_conn(cursor):
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


# ── get_allowed_cohorts ──────────────────────────────────────────────────────

def test_get_allowed_cohorts_returns_ids():
    svc = _make_service()
    cursor = _mock_cursor([(7,), (42,)])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn):
        result = svc.get_allowed_cohorts(99)
    assert result == [7, 42]


def test_get_allowed_cohorts_empty_for_unknown_user():
    svc = _make_service()
    cursor = _mock_cursor([])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn):
        result = svc.get_allowed_cohorts(0)
    assert result == []


def test_get_allowed_cohorts_cached_on_second_call():
    svc = _make_service()
    cursor = _mock_cursor([(1,)])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn) as mock_connect:
        svc.get_allowed_cohorts(5)
        svc.get_allowed_cohorts(5)   # should use cache
    assert mock_connect.call_count == 1


def test_get_allowed_cohorts_cache_expires():
    svc = _make_service()
    svc._cache_ttl = 0.05   # 50 ms for test speed
    cursor = _mock_cursor([(1,)])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn) as mock_connect:
        svc.get_allowed_cohorts(5)
        time.sleep(0.1)
        svc.get_allowed_cohorts(5)   # cache expired
    assert mock_connect.call_count == 2


# ── get_enrolled_course_ids ──────────────────────────────────────────────────

def test_get_enrolled_course_ids_returns_string_ids():
    svc = _make_service()
    cursor = _mock_cursor([(10,), (23,)])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn):
        result = svc.get_enrolled_course_ids(99)
    assert result == ["10", "23"]


def test_get_enrolled_course_ids_empty_when_not_enrolled():
    svc = _make_service()
    cursor = _mock_cursor([])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn):
        result = svc.get_enrolled_course_ids(1)
    assert result == []


def test_get_enrolled_course_ids_cached():
    svc = _make_service()
    cursor = _mock_cursor([(10,)])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn) as mock_connect:
        svc.get_enrolled_course_ids(5)
        svc.get_enrolled_course_ids(5)
    assert mock_connect.call_count == 1


# ── DB failure ───────────────────────────────────────────────────────────────

def test_get_allowed_cohorts_raises_on_db_error():
    svc = _make_service()
    with patch("services.silo_service.pymysql.connect", side_effect=Exception("DB down")):
        with pytest.raises(Exception, match="DB down"):
            svc.get_allowed_cohorts(1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_silo_service.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'services.silo_service'`

- [ ] **Step 3: Implement SiloService**

Create `/opt/craftpilot_backend/services/silo_service.py`:

```python
"""Per-user access scope resolver — queries Moodle DB for cohort membership
and course enrolments. Results are cached in-memory for 60 seconds."""

import logging
import os
import time
from typing import Optional

import pymysql
import pymysql.cursors

logger = logging.getLogger(__name__)

_COHORT_QUERY = """
    SELECT DISTINCT cm.cohortid
    FROM mdl_cohort_members cm
    WHERE cm.userid = %s
"""

_ENROL_QUERY = """
    SELECT DISTINCT e.courseid
    FROM mdl_user_enrolments ue
    JOIN mdl_enrol e ON e.id = ue.enrolid
    WHERE ue.userid = %s
      AND ue.status = 0
      AND e.status = 0
"""


class SiloService:
    """Resolves per-user access scope from the Moodle MySQL DB.

    Both methods cache their results for ``_cache_ttl`` seconds (default 60)
    keyed by user_id.  Raises on DB failure — callers must treat that as 503.
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_user: str = "moodleuser",
        db_password: Optional[str] = None,
        db_name: str = "moodle",
        cache_ttl: float = 60.0,
    ):
        self._db_host = db_host
        self._db_user = db_user
        self._db_password = db_password or os.getenv("MOODLE_DB_PASSWORD", "")
        self._db_name = db_name
        self._cache_ttl = cache_ttl
        self._cohort_cache: dict[int, tuple[list[int], float]] = {}
        self._course_cache: dict[int, tuple[list[str], float]] = {}

    def _connect(self):
        return pymysql.connect(
            host=self._db_host,
            user=self._db_user,
            password=self._db_password,
            database=self._db_name,
            cursorclass=pymysql.cursors.Cursor,
            connect_timeout=5,
        )

    def get_allowed_cohorts(self, user_id: int) -> list[int]:
        """Return Moodle cohort IDs the user belongs to."""
        cached, ts = self._cohort_cache.get(user_id, (None, 0.0))
        if cached is not None and (time.time() - ts) < self._cache_ttl:
            return cached

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_COHORT_QUERY, (user_id,))
                rows = cur.fetchall()

        result = [row[0] for row in rows]
        self._cohort_cache[user_id] = (result, time.time())
        logger.debug(f"SiloService: user {user_id} cohorts={result}")
        return result

    def get_enrolled_course_ids(self, user_id: int) -> list[str]:
        """Return active Moodle course IDs the user is enrolled in."""
        cached, ts = self._course_cache.get(user_id, (None, 0.0))
        if cached is not None and (time.time() - ts) < self._cache_ttl:
            return cached

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_ENROL_QUERY, (user_id,))
                rows = cur.fetchall()

        result = [str(row[0]) for row in rows]
        self._course_cache[user_id] = (result, time.time())
        logger.debug(f"SiloService: user {user_id} courses={result}")
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_silo_service.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /opt/craftpilot_backend
git add services/silo_service.py tests/test_silo_service.py
git commit -m "feat: add SiloService for Moodle cohort and enrolment lookup"
```

---

## Task 2: Annotation cohort metadata at ingest

**Files:**
- Modify: `api/models.py` (AnnotationIngestRequest)
- Modify: `services/annotation_service.py` (annotation_to_documents)
- Modify: `api/routes.py` (ingest_annotation handler)

**Interfaces:**
- Consumes: nothing new
- Produces:
  - `AnnotationIngestRequest.allowed_cohort_id: Optional[int]`
  - ChromaDB docs now carry `cohort_id: int` and `open_access: bool`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_silo_service.py` (or create `tests/test_annotation_cohort.py`):

```python
# tests/test_annotation_cohort.py
"""Tests that annotation_to_documents emits correct cohort metadata."""

from services.annotation_service import AnnotationService
from unittest.mock import patch, MagicMock


def _make_annotation(cohort_id=None):
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
        "annotation_updated_at": "2026-01-01",
        "transcription": "Hello world",
        "extended_transcript": None,
        "allowed_cohort_id": cohort_id,
    }


def _make_service():
    from config.settings import ConfigurationManager
    with patch("services.annotation_service.Path.exists", return_value=True):
        return AnnotationService(ConfigurationManager())


def test_open_annotation_has_open_access_true():
    svc = _make_service()
    docs = svc.annotation_to_documents(_make_annotation(cohort_id=None), use_extended=False)
    assert docs
    assert docs[0].metadata["open_access"] is True
    assert docs[0].metadata["cohort_id"] == -1


def test_restricted_annotation_has_cohort_id():
    svc = _make_service()
    docs = svc.annotation_to_documents(_make_annotation(cohort_id=7), use_extended=False)
    assert docs
    assert docs[0].metadata["open_access"] is False
    assert docs[0].metadata["cohort_id"] == 7
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_annotation_cohort.py -v
```

Expected: FAIL — `KeyError: 'open_access'`

- [ ] **Step 3: Add `allowed_cohort_id` to `AnnotationIngestRequest`**

In `api/models.py`, find `class AnnotationIngestRequest` and add one field:

```python
class AnnotationIngestRequest(BaseModel):
    annotation_id: int
    video_id: int
    transcription: str
    start_time: float
    end_time: float
    video_filename: str
    video_filepath: str
    source_type: str = "local"
    project_name: str = "unknown"
    audio_filepath: str = ""
    allowed_cohort_id: Optional[int] = None          # NEW — None = open access
```

- [ ] **Step 4: Add cohort fields to `annotation_to_documents`**

In `services/annotation_service.py`, find `base_metadata` dict (around line 183) and add after the existing fields:

```python
        base_metadata = {
            "annotation_id": annotation["annotation_id"],
            "video_id": annotation["video_id"],
            "video_filename": annotation["video_filename"] or "unknown.mp4",
            "video_filepath": annotation["video_filepath"] or "",
            "start_time": float(annotation["start_time"]) if annotation["start_time"] is not None else 0.0,
            "end_time": float(annotation["end_time"]) if annotation["end_time"] is not None else 0.0,
            "duration": float(annotation["end_time"] - annotation["start_time"]) if annotation["end_time"] is not None and annotation["start_time"] is not None else 0.0,
            "audio_filepath": annotation["audio_filepath"] or "",
            "source_type": annotation["source_type"] or "unknown",
            "project_name": annotation.get("project_name") or "unknown",
            "annotation_created_at": annotation["annotation_created_at"] or "",
            "type": "video_annotation",
            # Silo fields — cohort_id=-1 and open_access=True mean visible to all
            "cohort_id": annotation.get("allowed_cohort_id") if annotation.get("allowed_cohort_id") is not None else -1,
            "open_access": annotation.get("allowed_cohort_id") is None,
        }
```

- [ ] **Step 5: Pass `allowed_cohort_id` through `ingest_annotation` route**

In `api/routes.py`, find the `ingest_annotation` handler. Update the `annotation_dict` to include:

```python
        annotation_dict = {
            "annotation_id":    request.annotation_id,
            "video_id":         request.video_id,
            "transcription":    request.transcription,
            "start_time":       request.start_time,
            "end_time":         request.end_time,
            "video_filename":   request.video_filename,
            "video_filepath":   request.video_filepath,
            "source_type":      request.source_type,
            "project_name":     request.project_name,
            "audio_filepath":   request.audio_filepath,
            "allowed_cohort_id": request.allowed_cohort_id,   # NEW
            "extended_transcript": None,
        }
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_annotation_cohort.py -v
```

Expected: both tests PASS

- [ ] **Step 7: Commit**

```bash
cd /opt/craftpilot_backend
git add api/models.py services/annotation_service.py api/routes.py tests/test_annotation_cohort.py
git commit -m "feat: add cohort_id and open_access metadata to annotation ChromaDB documents"
```

---

## Task 3: ChromaDB silo filter in RAGService

**Files:**
- Modify: `core/types.py`
- Modify: `services/rag_service.py`

**Interfaces:**
- Consumes: `ConversationState.user_cohort_ids: list[int]`
- Produces: `RAGService.similarity_search(query, k, cohort_filter)` with optional `filter` arg passed to ChromaDB

- [ ] **Step 1: Write the failing test**

Create `tests/test_cohort_filter.py`:

```python
"""Tests that similarity_search passes cohort filter to ChromaDB."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents.base import Document
from services.rag_service import RAGService
from config.settings import ConfigurationManager


def _make_service():
    with patch.object(RAGService, "_initialize_embeddings", return_value=MagicMock()), \
         patch.object(RAGService, "_initialize_vector_store", return_value=MagicMock()), \
         patch.object(RAGService, "_initialize_llm", return_value=MagicMock()):
        svc = RAGService(ConfigurationManager())
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
    from services.rag_service import build_cohort_filter
    f = build_cohort_filter([1, 2])
    assert f == {"$or": [{"cohort_id": {"$in": [1, 2]}}, {"open_access": True}]}


def test_build_cohort_filter_no_cohorts_returns_open_only():
    from services.rag_service import build_cohort_filter
    f = build_cohort_filter([])
    assert f == {"open_access": True}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_cohort_filter.py -v
```

Expected: FAIL — `ImportError: cannot import name 'build_cohort_filter'`

- [ ] **Step 3: Add `build_cohort_filter` and update `similarity_search`**

In `services/rag_service.py`, add this module-level function after the imports:

```python
def build_cohort_filter(user_cohort_ids: list) -> dict:
    """Build a ChromaDB `where` filter enforcing cohort-level access.

    Documents pass if they are open-access (cohort_id == -1, open_access == True)
    OR if their cohort_id is in the user's allowed cohort list.
    """
    if not user_cohort_ids:
        return {"open_access": True}
    return {
        "$or": [
            {"cohort_id": {"$in": list(user_cohort_ids)}},
            {"open_access": True},
        ]
    }
```

Then update the `similarity_search` method signature and body:

```python
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        cohort_filter: Optional[dict] = None,
    ) -> List[Document]:
        """Search the vector store, optionally scoped to a cohort filter."""
        try:
            k = k or self.config.similarity_search_k

            kwargs = {}
            if cohort_filter is not None:
                kwargs["filter"] = cohort_filter

            results = self.vector_store.max_marginal_relevance_search(query, k=k, **kwargs)
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []
```

- [ ] **Step 4: Propagate `cohort_filter` through retrieve nodes**

In `services/rag_service.py`, update every retrieve node that calls `self.similarity_search` to extract `user_cohort_ids` from state and pass a filter. The pattern to apply to `retrieve_with_hyde`, `retrieve_combined`, `retrieve_initial`, `retrieve_final_dual`, `retrieve`, and `retrieve_final`:

```python
# At the top of each retrieve node, extract the filter:
user_cohort_ids = state.get("user_cohort_ids") or []
cohort_filter = build_cohort_filter(user_cohort_ids) if user_cohort_ids is not None else None

# Then pass it to similarity_search:
retrieved_docs = self.similarity_search(search_query, k=5, cohort_filter=cohort_filter)
```

Apply this pattern to all six retrieve methods. Only change the `similarity_search` call lines — do not restructure the methods.

- [ ] **Step 5: Add `user_cohort_ids` to `ConversationState`**

In `core/types.py`:

```python
class ConversationState(MessagesState):
    context: List[Document]
    video_metadata: Optional[Dict[str, Any]]
    hypothetical_document: Optional[str]
    enhanced_query: Optional[str]
    query_variants: List[str]
    route: Optional[str]
    selected_domain: Optional[str]
    course_id: Optional[str]
    refined_query: Optional[str]
    user_cohort_ids: Optional[List[int]]          # NEW — from SiloService
    enrolled_course_ids: Optional[List[str]]      # NEW — from SiloService
```

- [ ] **Step 6: Run tests**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_cohort_filter.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 7: Commit**

```bash
cd /opt/craftpilot_backend
git add services/rag_service.py core/types.py tests/test_cohort_filter.py
git commit -m "feat: add cohort silo filter to RAGService similarity search"
```

---

## Task 4: Course content scoping in CourseRAGService

**Files:**
- Modify: `services/course_rag_service.py`

**Interfaces:**
- Consumes: `enrolled_course_ids: list[str]` from `ConversationState`
- Produces: `similarity_search_all_courses(query, allowed_course_ids)` — skips collections not in the allowed set

- [ ] **Step 1: Write the failing test**

Add `tests/test_course_silo.py`:

```python
"""Tests that course RAG service respects enrolled_course_ids."""

from unittest.mock import MagicMock, patch
from langchain_core.documents.base import Document


def _make_service():
    from services.course_rag_service import CourseRAGService
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1] * 384
    return CourseRAGService(embeddings=mock_embeddings, persist_directory="/tmp/test_chroma")


def test_similarity_search_all_courses_respects_allowed_list():
    svc = _make_service()
    # Pretend three courses exist in ChromaDB, user is enrolled in two
    with patch.object(svc, "_enumerate_populated_courses", return_value=["1", "2", "3"]), \
         patch.object(svc, "_search_with_embedding", return_value=[]) as mock_search:
        svc.similarity_search_all_courses("query", allowed_course_ids=["1", "3"])

    searched_ids = [call.args[1] for call in mock_search.call_args_list]
    assert "1" in searched_ids
    assert "3" in searched_ids
    assert "2" not in searched_ids


def test_similarity_search_all_courses_no_filter_queries_all():
    svc = _make_service()
    with patch.object(svc, "_enumerate_populated_courses", return_value=["1", "2"]), \
         patch.object(svc, "_search_with_embedding", return_value=[]) as mock_search:
        svc.similarity_search_all_courses("query", allowed_course_ids=None)

    searched_ids = [call.args[1] for call in mock_search.call_args_list]
    assert set(searched_ids) == {"1", "2"}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_course_silo.py -v
```

Expected: FAIL — `TypeError: unexpected keyword argument 'allowed_course_ids'`

- [ ] **Step 3: Update `similarity_search_all_courses` signature**

In `services/course_rag_service.py`, update the method:

```python
    def similarity_search_all_courses(
        self,
        query: str,
        k_per_course: int = 1,
        priority_course_id: Optional[str] = None,
        allowed_course_ids: Optional[list] = None,   # NEW — None = no restriction
    ) -> List[Document]:
        """Query course collections the user is enrolled in.

        If ``allowed_course_ids`` is provided, only those collections are queried.
        """
        all_docs: List[Document] = []
        course_ids = self._enumerate_populated_courses()
        if not course_ids:
            logger.info("similarity_search_all_courses: no populated courses found")
            return all_docs

        # Apply enrolment filter
        if allowed_course_ids is not None:
            allowed_set = set(str(cid) for cid in allowed_course_ids)
            course_ids = [cid for cid in course_ids if cid in allowed_set]
            if not course_ids:
                logger.info("similarity_search_all_courses: user enrolled in no indexed courses")
                return all_docs

        try:
            embedding = self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"similarity_search_all_courses: embedding failed: {e}")
            return all_docs

        priority_docs: List[Document] = []
        other_docs: List[Document] = []

        for cid in course_ids:
            if cid == priority_course_id:
                docs = self._search_with_embedding(embedding, cid, k=6)
                priority_docs.extend(docs)
            else:
                docs = self._search_with_embedding(embedding, cid, k=k_per_course)
                other_docs.extend(docs)

        all_docs = priority_docs + other_docs
        logger.info(
            f"similarity_search_all_courses: {len(all_docs)} docs across "
            f"{len(course_ids)} courses (priority={priority_course_id})"
        )
        return all_docs
```

- [ ] **Step 4: Update callers in `rag_service.py`**

Find all calls to `self.course_rag_service.similarity_search_all_courses(...)` in `rag_service.py` (in `retrieve_initial` and `retrieve_final_dual`). Add `allowed_course_ids` from state:

```python
enrolled_course_ids = state.get("enrolled_course_ids")   # already set in state

course_results = self.course_rag_service.similarity_search_all_courses(
    query,
    priority_course_id=course_id,
    allowed_course_ids=enrolled_course_ids,   # NEW
)
```

- [ ] **Step 5: Run tests**

```bash
cd /opt/craftpilot_backend && python -m pytest tests/test_course_silo.py -v
```

Expected: both tests PASS

- [ ] **Step 6: Commit**

```bash
cd /opt/craftpilot_backend
git add services/course_rag_service.py tests/test_course_silo.py
git commit -m "feat: filter CourseRAGService queries to user's enrolled courses"
```

---

## Task 5: Pipeline wiring — user_id guard + SiloService injection

**Files:**
- Modify: `api/models.py` (ChatRequest)
- Modify: `api/routes.py` (chat handler guard)
- Modify: `pipeline.py` (stream_response + SiloService)

**Interfaces:**
- Consumes: `SiloService` (Task 1), `ConversationState` new fields (Task 3), `build_cohort_filter` (Task 3)
- Produces: `pipeline.stream_response(message, ..., user_id)` sets silo state before graph runs

- [ ] **Step 1: Add `user_id` to `ChatRequest`**

In `api/models.py`:

```python
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_thread_id: str = Field(..., max_length=255)
    selected_domain: Optional[str] = Field(None, max_length=100)
    course_id: Optional[str] = Field(None, max_length=20)
    is_first_message: bool = False
    disable_rerank: bool = False
    user_id: Optional[int] = None          # NEW — validated by chat_proxy.php
```

- [ ] **Step 2: Add 403 guard and pass `user_id` in the chat route**

In `api/routes.py`, replace the `chat_stream` handler:

```python
@router.post("/chat")
async def chat_stream(request: ChatRequest):
    """Streaming chat — requires a validated user_id from chat_proxy.php."""
    if not request.user_id or request.user_id <= 0:
        raise HTTPException(status_code=403, detail="user_id required")

    return StreamingResponse(
        generate_simplified_stream(
            request.message,
            request.conversation_thread_id,
            request.selected_domain,
            request.course_id,
            request.is_first_message,
            request.disable_rerank,
            user_id=request.user_id,        # NEW
        ),
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 3: Update `generate_simplified_stream` and `stream_response` to accept `user_id`**

In `api/routes.py`, update the function signature:

```python
async def generate_simplified_stream(
    user_messages: str,
    conversation_thread_id: str,
    selected_domain: Optional[str] = None,
    course_id: Optional[str] = None,
    is_first_message: bool = False,
    disable_rerank: bool = False,
    user_id: Optional[int] = None,         # NEW
) -> AsyncGenerator[str, None]:
    async for line in pipeline.stream_response(
        user_messages,
        conversation_thread_id=conversation_thread_id,
        selected_domain=selected_domain,
        course_id=course_id,
        is_first_message=is_first_message,
        disable_rerank=disable_rerank,
        user_id=user_id,                   # NEW
    ):
        yield line
```

- [ ] **Step 4: Wire SiloService into the pipeline**

In `pipeline.py`, at the top of `__init__`, after other service instantiation:

```python
from services.silo_service import SiloService

# In __init__:
self.silo_service = SiloService()
```

Update `stream_response` signature:

```python
    async def stream_response(
        self,
        message: str,
        conversation_thread_id: str,
        selected_domain: Optional[str] = None,
        course_id: Optional[str] = None,
        is_first_message: bool = False,
        disable_rerank: bool = False,
        user_id: Optional[int] = None,     # NEW
    ):
```

In `stream_response`, before the initial state dict is built, resolve the silo:

```python
        import json
        from services.rag_service import build_cohort_filter

        # Resolve silo scope — raises on DB failure (caught below → 503)
        user_cohort_ids: list[int] = []
        enrolled_course_ids: list[str] = []
        if user_id and user_id > 0:
            try:
                user_cohort_ids = self.silo_service.get_allowed_cohorts(user_id)
                enrolled_course_ids = self.silo_service.get_enrolled_course_ids(user_id)
            except Exception as e:
                logger.error(f"SiloService failed for user {user_id}: {e}")
                yield json.dumps({"event": "error", "data": "Service temporarily unavailable"}) + "\n"
                yield json.dumps({"content": "[DONE]"}) + "\n"
                return
```

Then include them in the initial state dict (find where `state: Dict[str, Any] = {` is built and add):

```python
            state: Dict[str, Any] = {
                # ... existing fields ...
                "user_cohort_ids": user_cohort_ids,           # NEW
                "enrolled_course_ids": enrolled_course_ids,   # NEW
            }
```

- [ ] **Step 5: Smoke-test the guard**

```bash
cd /opt/craftpilot_backend && python -c "
from api.models import ChatRequest
r = ChatRequest(message='hello', conversation_thread_id='t1')
print('user_id default:', r.user_id)  # None
r2 = ChatRequest(message='hello', conversation_thread_id='t1', user_id=42)
print('user_id set:', r2.user_id)  # 42
"
```

Expected output:
```
user_id default: None
user_id set: 42
```

- [ ] **Step 6: Commit**

```bash
cd /opt/craftpilot_backend
git add api/models.py api/routes.py pipeline.py
git commit -m "feat: wire SiloService into pipeline — 403 guard + cohort state injection"
```

---

## Task 6: ChromaDB resync endpoint for project cohort changes

**Files:**
- Modify: `api/models.py` (new ResyncProjectRequest model)
- Modify: `api/routes.py` (new endpoint)

**Interfaces:**
- Produces: `POST /api/resync-project-annotations` — deletes and re-ingests all docs for a `project_name`

- [ ] **Step 1: Add the request model**

In `api/models.py`:

```python
class ResyncProjectRequest(BaseModel):
    """Payload for re-tagging a project's ChromaDB documents with a new cohort."""
    project_name: str = Field(..., max_length=255)
    allowed_cohort_id: Optional[int] = None   # None = open access
```

- [ ] **Step 2: Add the resync endpoint**

In `api/routes.py`, add after the `ingest-annotation` route:

```python
@router.post("/resync-project-annotations")
async def resync_project_annotations(request: ResyncProjectRequest):
    """Delete and re-ingest all ChromaDB documents for a project with updated cohort metadata.

    Called automatically by the video elicitation backend when an expert
    changes the allowed_cohort_id on an existing project.
    """
    try:
        project_name = request.project_name

        # 1. Find all existing ChromaDB docs for this project
        existing = pipeline.rag_service.vector_store.get(
            where={"project_name": project_name}
        )
        if existing and existing.get("ids"):
            pipeline.rag_service.vector_store.delete(ids=existing["ids"])
            logger.info(
                f"resync: deleted {len(existing['ids'])} docs for project '{project_name}'"
            )

        # 2. Fetch annotations from SQLite and re-ingest with new cohort metadata
        annotations = pipeline.annotation_service.get_completed_annotations(
            include_extended=True
        )
        project_annotations = [
            a for a in annotations if (a.get("project_name") or "unknown") == project_name
        ]

        if not project_annotations:
            return {"status": "ok", "documents_resynced": 0, "project_name": project_name}

        # Inject the new cohort_id into each annotation before converting
        for ann in project_annotations:
            ann["allowed_cohort_id"] = request.allowed_cohort_id

        docs = []
        for ann in project_annotations:
            docs.extend(
                pipeline.annotation_service.annotation_to_documents(ann, use_extended=True)
            )

        if docs:
            pipeline.rag_service.add_documents(docs)

        return {
            "status": "ok",
            "documents_resynced": len(docs),
            "project_name": project_name,
            "allowed_cohort_id": request.allowed_cohort_id,
        }

    except Exception as e:
        logger.error(f"resync-project-annotations failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 3: Smoke-test the model**

```bash
cd /opt/craftpilot_backend && python -c "
from api.models import ResyncProjectRequest
r = ResyncProjectRequest(project_name='Soudure', allowed_cohort_id=7)
print(r)
r2 = ResyncProjectRequest(project_name='Soudure')
print(r2.allowed_cohort_id)  # None
"
```

Expected: prints model repr, then `None`

- [ ] **Step 4: Commit**

```bash
cd /opt/craftpilot_backend
git add api/models.py api/routes.py
git commit -m "feat: add resync-project-annotations endpoint for cohort re-tagging"
```

---

## Task 7: Video Elicitation Tool — SQLite migration + models

**Files:**
- Modify: `/opt/video_elicitation_annotation_tool/backend/migration.py`
- Modify: `/opt/video_elicitation_annotation_tool/backend/models.py`

**Interfaces:**
- Produces:
  - `projects.allowed_cohort_id INTEGER DEFAULT NULL` column
  - `Project.allowed_cohort_id: Optional[int]`
  - `ProjectCreate.allowed_cohort_id: Optional[int]`
  - `ProjectUpdate.allowed_cohort_id: Optional[int]`
  - `ProjectResponse.allowed_cohort_id: Optional[int]`

- [ ] **Step 1: Add the migration**

In `/opt/video_elicitation_annotation_tool/backend/migration.py`, find the `MIGRATIONS` list at the bottom. Add before it:

```python
def migration_add_project_cohort_id(cursor):
    """Add allowed_cohort_id to projects table. NULL = open access."""
    columns = get_table_columns(cursor, "projects")
    if "allowed_cohort_id" not in columns:
        cursor.execute(
            "ALTER TABLE projects ADD COLUMN allowed_cohort_id INTEGER DEFAULT NULL"
        )
        logger.info("Added allowed_cohort_id to projects table (NULL = open access)")
    else:
        logger.info("allowed_cohort_id already exists in projects table")
```

Then append to `MIGRATIONS`:

```python
MIGRATIONS = [
    # ... existing entries ...
    ("add_project_cohort_id", migration_add_project_cohort_id),
]
```

- [ ] **Step 2: Run the migration**

```bash
cd /opt/video_elicitation_annotation_tool/backend && python migration.py
```

Expected output includes: `Added allowed_cohort_id to projects table`

- [ ] **Step 3: Verify column exists**

```bash
cd /opt/video_elicitation_annotation_tool/backend && python -c "
import sqlite3
from config import CHROMA_DIR
conn = sqlite3.connect(str(CHROMA_DIR / 'annotations.db'))
cols = [r[1] for r in conn.execute('PRAGMA table_info(projects)').fetchall()]
print('allowed_cohort_id' in cols)
"
```

Expected: `True`

- [ ] **Step 4: Update Pydantic models**

In `/opt/video_elicitation_annotation_tool/backend/models.py`, find and update:

```python
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    allowed_cohort_id: Optional[int] = None    # NEW

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    allowed_cohort_id: Optional[int] = None    # NEW

class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    allowed_cohort_id: Optional[int] = None    # NEW

    class Config:
        from_attributes = True
```

Also update the SQLAlchemy `Project` model class to add the column:

```python
class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    allowed_cohort_id = Column(Integer, nullable=True, default=None)   # NEW
```

- [ ] **Step 5: Commit**

```bash
cd /opt/video_elicitation_annotation_tool
git add backend/migration.py backend/models.py
git commit -m "feat: add allowed_cohort_id to projects table and models"
```

---

## Task 8: Video Elicitation backend — `/api/cohorts/managed` + project resync trigger

**Files:**
- Modify: `/opt/video_elicitation_annotation_tool/backend/main.py`

**Interfaces:**
- Produces:
  - `GET /api/cohorts/managed` → `[{"cohort_id": int, "cohort_name": str}]`
  - `PUT /api/projects/{id}` — triggers CraftPilot resync when `allowed_cohort_id` changes

- [ ] **Step 1: Add `/api/cohorts/managed` endpoint**

In `backend/main.py`, add after the imports (add `pymysql` import if not present):

```python
import pymysql
import pymysql.cursors
```

Add the endpoint (place it near the other project endpoints around line 816):

```python
_MANAGED_COHORTS_QUERY = """
    SELECT DISTINCT c.id, c.name
    FROM mdl_cohort c
    JOIN mdl_enrol e ON e.customint1 = c.id AND e.enrol = 'cohort'
    JOIN mdl_context ctx ON ctx.instanceid = e.courseid AND ctx.contextlevel = 50
    JOIN mdl_role_assignments ra ON ra.contextid = ctx.id AND ra.userid = %s
    JOIN mdl_role r ON r.id = ra.roleid
        AND r.shortname IN ('teacher', 'editingteacher', 'manager')
"""


@app.get("/api/cohorts/managed")
async def get_managed_cohorts(request: Request):
    """Return cohorts the JWT user is responsible for (has teacher role in an enrolled course).

    Returns empty list if user has no teacher roles — frontend shows the contact-admin message.
    """
    # Decode JWT from Authorization header
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing JWT")

    token = auth[7:]
    try:
        import base64, json as _json
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = _json.loads(base64.urlsafe_b64decode(payload_b64))
        user_id = payload["userid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid JWT")

    try:
        conn = pymysql.connect(
            host="localhost",
            user="moodleuser",
            password=os.getenv("MOODLE_DB_PASSWORD", ""),
            database="moodle",
            cursorclass=pymysql.cursors.DictCursor,
        )
        with conn:
            with conn.cursor() as cur:
                cur.execute(_MANAGED_COHORTS_QUERY, (user_id,))
                rows = cur.fetchall()
        return [{"cohort_id": r["id"], "cohort_name": r["name"]} for r in rows]
    except Exception as e:
        logger.error(f"get_managed_cohorts failed: {e}")
        raise HTTPException(status_code=503, detail="Could not query Moodle DB")
```

- [ ] **Step 2: Trigger CraftPilot resync on project cohort change**

Find the `update_project` endpoint (around line 883) and add the resync call after a successful update:

```python
@app.put("/api/projects/{project_id}", response_model=models.ProjectResponse)
async def update_project(
    project_id: int,
    project_update: models.ProjectUpdate,
    session: AsyncSession = Depends(db.get_session),
):
    """Update a project. Triggers ChromaDB resync if allowed_cohort_id changes."""
    try:
        # Fetch current state before update
        current = await db.get_project(session, project_id)
        if not current:
            raise HTTPException(status_code=404, detail="Project not found")

        old_cohort_id = current.allowed_cohort_id
        updated_project = await db.update_project(session, project_id, project_update)
        logger.info(f"Project updated: ID={project_id}")

        # If cohort changed, trigger CraftPilot resync asynchronously
        new_cohort_id = updated_project.allowed_cohort_id
        if old_cohort_id != new_cohort_id:
            import httpx, asyncio
            craftpilot_url = "http://127.0.0.1:8000/api/resync-project-annotations"
            internal_token = os.getenv("INTERNAL_API_TOKEN", "")
            payload = {
                "project_name": updated_project.name,
                "allowed_cohort_id": new_cohort_id,
            }
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        craftpilot_url,
                        json=payload,
                        headers={"X-Internal-Token": internal_token},
                    )
                    resp.raise_for_status()
                    logger.info(
                        f"ChromaDB resync triggered for project '{updated_project.name}': "
                        f"cohort {old_cohort_id} → {new_cohort_id}"
                    )
            except Exception as e:
                logger.error(f"ChromaDB resync failed for project {project_id}: {e}")
                # Do not fail the project update — resync can be retried

        return updated_project
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

Also add `httpx` to requirements if not present:

```bash
cd /opt/video_elicitation_annotation_tool && grep -q "httpx" backend/requirements.txt || echo "httpx" >> backend/requirements.txt
```

- [ ] **Step 3: Verify the endpoint is reachable (manual test)**

```bash
# Get a valid JWT from the video elicitation tool first, then:
curl -s http://127.0.0.1:8005/api/cohorts/managed \
  -H "Authorization: Bearer <JWT>" | python3 -m json.tool
```

Expected: JSON array (may be empty `[]` if no teacher roles exist in test env)

- [ ] **Step 4: Commit**

```bash
cd /opt/video_elicitation_annotation_tool
git add backend/main.py backend/requirements.txt
git commit -m "feat: add /api/cohorts/managed and auto-resync on project cohort change"
```

---

## Task 9: Video Elicitation UI — cohort selector + silo banner

**Files:**
- Modify: `/opt/video_elicitation_annotation_tool/js/app.js`

- [ ] **Step 1: Fetch managed cohorts and store in app state**

In `js/app.js`, find where `state` is initialised (near the top) and add:

```javascript
managedCohorts: [],        // {cohort_id, cohort_name}[] from /api/cohorts/managed
siloContactEmail: null,    // from JWT payload.silo_contact_email
```

Add a function to load cohorts after auth (call it after JWT is verified):

```javascript
async function loadManagedCohorts() {
    try {
        const jwt = state.token;  // existing JWT storage — adjust name to match codebase
        const resp = await fetch('/api/cohorts/managed', {
            headers: { 'Authorization': `Bearer ${jwt}` }
        });
        if (resp.ok) {
            state.managedCohorts = await resp.json();
        }
    } catch (e) {
        console.warn('Could not load managed cohorts:', e);
    }
}
```

- [ ] **Step 2: Add cohort selector to project create/edit modal**

Find the function that renders the project modal (search for `createProject` or the modal HTML). After the description field, add:

```javascript
function renderCohortSelector(selectedCohortId) {
    const contactEmail = state.siloContactEmail || 'your Moodle administrator';
    if (state.managedCohorts.length === 0) {
        return `<div class="silo-notice">
            <p>You are not currently assigned as a teacher in any cohort-enrolled course.
            If your work should be protected from other organisations' search results in
            CraftPilot, please contact
            <a href="mailto:${contactEmail}">${contactEmail}</a>
            to have the correct role assigned. Until then, your annotations will be
            visible to all authenticated users.</p>
        </div>`;
    }
    const options = state.managedCohorts.map(c =>
        `<option value="${c.cohort_id}" ${selectedCohortId === c.cohort_id ? 'selected' : ''}>
            ${c.cohort_name} only
        </option>`
    ).join('');
    return `<label for="project-cohort">Visibility</label>
        <select id="project-cohort" name="allowed_cohort_id">
            <option value="" ${!selectedCohortId ? 'selected' : ''}>
                Open access — visible to all authenticated CraftPilot users
            </option>
            ${options}
        </select>`;
}
```

When reading the form on submit, include:

```javascript
const cohortSelect = document.getElementById('project-cohort');
const allowed_cohort_id = cohortSelect && cohortSelect.value
    ? parseInt(cohortSelect.value, 10)
    : null;
// include allowed_cohort_id in the PUT/POST body
```

- [ ] **Step 3: Add the dismissible silo banner**

Add after page load, before rendering projects:

```javascript
function showSiloBannerIfNeeded(projects) {
    const dismissed = localStorage.getItem('craftpilot_silo_banner_dismissed');
    if (dismissed) return;
    if (state.managedCohorts.length === 0) return;

    const unsecured = projects.filter(p => p.allowed_cohort_id == null);
    if (unsecured.length === 0) return;

    const banner = document.createElement('div');
    banner.className = 'silo-banner';
    banner.innerHTML = `
        <p>You have <strong>${unsecured.length} project(s)</strong> whose annotations are
        currently visible to all authenticated CraftPilot users. If this content contains
        proprietary knowledge, open each project's settings to assign it to a cohort.</p>
        <button id="silo-banner-dismiss">Don't show me again</button>
    `;
    document.querySelector('.projects-list').prepend(banner);
    document.getElementById('silo-banner-dismiss').addEventListener('click', () => {
        localStorage.setItem('craftpilot_silo_banner_dismissed', '1');
        banner.remove();
    });
}
```

Call `showSiloBannerIfNeeded(projects)` after projects are loaded.

- [ ] **Step 4: Commit**

```bash
cd /opt/video_elicitation_annotation_tool
git add js/app.js
git commit -m "feat: add cohort selector to project modal and silo awareness banner"
```

---

## Task 10: Moodle videoelicit plugin — admin settings + JWT silo_contact_email

**Files:**
- Modify: `/var/www/html/public/local/videoelicit/settings.php`
- Modify: `/var/www/html/public/local/videoelicit/lang/en/local_videoelicit.php`
- Modify: `/var/www/html/public/local/videoelicit/classes/jwt_helper.php`
- Modify: `/var/www/html/public/local/videoelicit/index.php`

- [ ] **Step 1: Add settings**

In `settings.php`, before the closing `$ADMIN->add(...)` line, add:

```php
    // Knowledge Silo section
    $settings->add(new admin_setting_heading(
        'local_videoelicit/silo_header',
        get_string('settings_silo_header', 'local_videoelicit'),
        get_string('settings_silo_header_desc', 'local_videoelicit')
    ));

    $settings->add(new admin_setting_configtext(
        'local_videoelicit/silo_contact_email',
        get_string('settings_silo_contact_email', 'local_videoelicit'),
        get_string('settings_silo_contact_email_desc', 'local_videoelicit'),
        '',
        PARAM_EMAIL
    ));
```

- [ ] **Step 2: Add lang strings**

In `lang/en/local_videoelicit.php`, append:

```php
$string['settings_silo_header']            = 'Knowledge Silo';
$string['settings_silo_header_desc']       = 'Controls who can see elicitation content in the CraftPilot RAG.';
$string['settings_silo_contact_email']     = 'Silo contact email';
$string['settings_silo_contact_email_desc'] = 'Displayed to experts who have no cohort assigned. Leave blank to show "your Moodle administrator".';
```

- [ ] **Step 3: Embed `silo_contact_email` in JWT payload**

In `classes/jwt_helper.php`, find `$payload = json_encode([...])` in `create_token` and add the field:

```php
    public static function create_token($userid, $username, $contextid, $roles, $expires_minutes = 60, $silo_contact_email = '') {
        // ... existing code ...
        $payload = json_encode([
            'userid'             => $userid,
            'username'           => $username,
            'contextid'          => $contextid,
            'roles'              => $roles,
            'exp'                => time() + ($expires_minutes * 60),
            'iat'                => time(),
            'silo_contact_email' => $silo_contact_email,   // NEW
        ]);
```

- [ ] **Step 4: Pass setting from `index.php` to `create_token`**

In `index.php`, find the `create_token` call and update:

```php
$silo_contact_email = get_config('local_videoelicit', 'silo_contact_email') ?: '';
$jwt_token = jwt_helper::create_token(
    $USER->id,
    $USER->username,
    $context->id,
    $roles,
    60,
    $silo_contact_email    // NEW
);
```

- [ ] **Step 5: Verify setting appears in Moodle admin**

Navigate to: Site Administration → Plugins → Local plugins → Video Elicitation Tool

Expected: "Knowledge Silo" section visible with the email field.

- [ ] **Step 6: Commit**

```bash
cd /var/www/html/public/local/videoelicit
git add settings.php lang/en/local_videoelicit.php classes/jwt_helper.php index.php
git commit -m "feat: add Knowledge Silo admin settings and embed silo_contact_email in JWT"
```

---

## Task 11: Moodle craftpilot plugin — `chat_proxy.php` user_id cross-check

**Files:**
- Modify: `/var/www/html/public/local/craftpilot/chat_proxy.php`

- [ ] **Step 1: Add the cross-check**

In `chat_proxy.php`, find the block after `confirm_sesskey` passes (around line 51-56). Add immediately after the sesskey check:

```php
// Validate that the user_id in the body matches the authenticated session.
// This prevents a logged-in user from claiming another user's identity.
$incoming_user_id = isset($data['user_id']) ? (int)$data['user_id'] : 0;
if ($incoming_user_id !== (int)$USER->id) {
    http_response_code(403);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'user_id mismatch']);
    exit;
}
```

The full block should now read:

```php
// Validate the sesskey included in the JSON body.
$incoming_sesskey = $data['sesskey'] ?? '';
if (!confirm_sesskey($incoming_sesskey)) {
    http_response_code(403);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'Invalid session key']);
    exit;
}

// Validate that the user_id in the body matches the authenticated session.
$incoming_user_id = isset($data['user_id']) ? (int)$data['user_id'] : 0;
if ($incoming_user_id !== (int)$USER->id) {
    http_response_code(403);
    header('Content-Type: application/json');
    echo json_encode(['error' => 'user_id mismatch']);
    exit;
}
```

- [ ] **Step 2: Verify no syntax errors**

```bash
php -l /var/www/html/public/local/craftpilot/chat_proxy.php
```

Expected: `No syntax errors detected`

- [ ] **Step 3: Commit**

```bash
cd /var/www/html/public/local/craftpilot
git add chat_proxy.php
git commit -m "feat: cross-check user_id against Moodle session in chat_proxy.php"
```

---

## Task 12: Moodle craftpilot plugin — JS passes `user_id`

**Files:**
- Modify: `/var/www/html/public/local/craftpilot/amd/src/chat_interface.js`

- [ ] **Step 1: Add `user_id` to the POST payload**

In `chat_interface.js`, find the `payload` object (around line 1089):

```javascript
    const payload = {
        message: userMessage,
        conversation_thread_id: state.currentConvId,
        is_first_message: isFirstMessage,
        sesskey: (window.M && window.M.cfg) ? window.M.cfg.sesskey : '',
    };
```

Add `user_id`:

```javascript
    const payload = {
        message: userMessage,
        conversation_thread_id: state.currentConvId,
        is_first_message: isFirstMessage,
        sesskey:  (window.M && window.M.cfg) ? window.M.cfg.sesskey  : '',
        user_id:  (window.M && window.M.cfg) ? window.M.cfg.userid   : 0,   // NEW
    };
```

`M.cfg.userid` is a standard Moodle global — always set for authenticated users.

- [ ] **Step 2: Rebuild the AMD bundle**

```bash
cd /var/www/html/public && grunt amd --root=local/craftpilot 2>/dev/null \
  || npx grunt amd --root=local/craftpilot
```

If Grunt is not available, copy the source to the build manually (dev environments often skip the build step):

```bash
cp /var/www/html/public/local/craftpilot/amd/src/chat_interface.js \
   /var/www/html/public/local/craftpilot/amd/build/chat_interface.min.js
```

- [ ] **Step 3: Verify payload in browser DevTools**

Open the CraftPilot chat widget, open DevTools → Network, send a message. In the request to `chat_proxy.php`, verify the JSON body contains `"user_id": <your_moodle_user_id>`.

- [ ] **Step 4: Commit**

```bash
cd /var/www/html/public/local/craftpilot
git add amd/src/chat_interface.js amd/build/chat_interface.min.js
git commit -m "feat: include user_id in CraftPilot chat POST payload for silo enforcement"
```

---

## Post-Deploy Migration Step

After all tasks are deployed, run a full annotation re-sync so existing ChromaDB documents get `cohort_id` and `open_access` metadata:

```bash
curl -s -X POST http://127.0.0.1:8000/api/sync-annotations \
  -H "Content-Type: application/json" \
  -H "X-Internal-Token: $INTERNAL_API_TOKEN" \
  -d '{"use_extended": true, "clear_existing": true}'
```

This is a one-time operation. All existing annotations will get `cohort_id = -1` and `open_access = True` (open mode) since their projects have `allowed_cohort_id = NULL`. Experts can then update their projects' cohort settings via the UI, which triggers automatic resync.

---

## Self-Review Checklist

- [x] SiloService queries `mdl_cohort_members` for cohort, `mdl_user_enrolments + mdl_enrol` for courses ✓
- [x] `build_cohort_filter` correctly handles empty cohort list (open-only filter) ✓
- [x] All six retrieve methods in `rag_service.py` are updated (Task 3 step 4) ✓
- [x] Course filtering added to `similarity_search_all_courses` ✓
- [x] `user_id` guard in route returns 403 before any pipeline execution ✓
- [x] Resync endpoint deletes before re-ingesting — avoids duplicate docs ✓
- [x] SQLite migration is idempotent (checks column existence before ALTER) ✓
- [x] `silo_contact_email` flows: Moodle setting → JWT payload → JS state → UI ✓
- [x] `chat_proxy.php` cross-check uses `(int)` cast on both sides — no type coercion surprise ✓
- [x] Post-deploy resync documented ✓
