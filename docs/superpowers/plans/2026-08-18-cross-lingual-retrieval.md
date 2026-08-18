# Cross-Lingual Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** any query, in any language, retrieves at French-corpus quality against the French-only vector store, and the assistant answers back in the query's language.

**Architecture:** A new `detect_and_translate_query` node runs first in the pipeline. A local, near-zero-cost language ID (`py3langid`) checks the raw query; French queries pass through unchanged (no LLM call, no latency added). Non-French queries get one LLM call that translates to French, producing `search_query` — which every retrieval node (`retrieve_initial`, `refine_query_prf`, `retrieve_final_dual`) then embeds instead of the raw query, so the rest of the pipeline (already tuned for French) runs unmodified. `query_language` also flips one line of the system prompt so the LLM answers in the same language as the question.

**Tech Stack:** Python 3.12 (conda env `moodle_backend`) · LangChain · `py3langid` (new dependency, pip-installed manually, no requirements.txt in this repo) · pytest

**Spec:** `docs/superpowers/specs/2026-08-18-cross-lingual-retrieval-design.md`

## Global Constraints

- `py3langid` must be installed in the `moodle_backend` conda env before Task 3's tests can pass: `/root/miniconda3/envs/moodle_backend/bin/python -m pip install py3langid`
- Use `py3langid.LanguageIdentifier.from_modelstring(py3langid.model, norm_probs=True).classify(text)` — NOT the bare module-level `py3langid.classify()`, which returns unnormalized log-probabilities, not a usable `[0, 1]` confidence
- Every failure path (langid unavailable, low confidence, short query, translation call failure) MUST degrade to `query_language="fr"` / `search_query=<original>` — never worse than the pipeline's current behavior. This is a hard invariant, test it explicitly in every task that touches it.
- `_build_messages`'s `<query>` tag in the user-facing prompt always uses the **original** `messages[-1].content`, never `search_query` — the LLM must see what the apprentice actually typed. Only the system-prompt language rule changes.
- Do not touch `MAX_RERANK_CANDIDATES` (rag_service.py) or reranking latency behavior — out of scope, already constrained by the 2-core CPU budget.
- All tests run from `/opt/craftpilot_backend/` with `/root/miniconda3/envs/moodle_backend/bin/python -m pytest` (system `python3` has an incompatible `langchain` install and cannot import this codebase)
- Never commit `.env` or secrets
- Frequent small commits — one per task minimum
- Branch: `feature/cross-lingual-retrieval` (already created, one commit in: eval script + spec)

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `config/settings.py` | Add `enable_cross_lingual_detection`, `langid_confidence_threshold`, `min_langid_chars` to `RAGConfig` |
| Modify | `core/types.py` | Add `query_language`, `search_query` to `ConversationState` |
| Modify | `services/rag_service.py` | New `_initialize_langid`, new `detect_and_translate_query` node; `retrieve_initial`/`refine_query_prf`/`retrieve_final_dual` read `search_query`; `_build_messages` swaps the language rule |
| Create | `tests/test_language_detection.py` | Unit tests for `detect_and_translate_query` and the language-rule swap |
| Modify | `pipeline.py` | Call `detect_and_translate_query` first in `stream_response`; status event; kill-switch check; add to `_build_conversation_graph` functions list |
| Modify | `tests/test_pipeline_integration.py` | Update `test_stream_response_does_not_short_circuit_in_domain` to mock the new node |
| Modify | `eval/09_cross_lingual_eval.py` | Add Config D (translate-first pipeline) |
| Create | `eval/fixtures/ground_truth_ambiguous.json` | Short/ambiguous French utterances — confirms the confidence gate doesn't over-trigger |

---

## Task 1: Config flags

**Files:**
- Modify: `config/settings.py`

**Interfaces:**
- Produces: `RAGConfig.enable_cross_lingual_detection: bool`, `RAGConfig.langid_confidence_threshold: float`, `RAGConfig.min_langid_chars: int`

- [ ] **Step 1: Add the fields**

In `config/settings.py`, inside `RAGConfig` (after `remote_reranker_score_threshold: float = 0.1`, line 59):

```python
    # Cross-lingual query handling — see docs/superpowers/specs/2026-08-18-cross-lingual-retrieval-design.md
    enable_cross_lingual_detection: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CROSS_LINGUAL_DETECTION", "true").lower() == "true"
    )
    # py3langid confidence below this defaults to "fr" rather than guessing —
    # biased toward the safe direction (never spuriously mistranslate real French).
    langid_confidence_threshold: float = 0.5
    # Queries shorter than this (chars) default to "fr" — langid is unreliable on short text.
    min_langid_chars: int = 12
```

- [ ] **Step 2: Verify with a smoke import**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -c "
from config.settings import RAGConfig
c = RAGConfig()
print(c.enable_cross_lingual_detection, c.langid_confidence_threshold, c.min_langid_chars)
"
```

Expected: `True 0.5 12`

- [ ] **Step 3: Commit**

```bash
cd /opt/craftpilot_backend
git add config/settings.py
git commit -m "feat: add cross-lingual detection config flags"
```

---

## Task 2: `ConversationState` fields

**Files:**
- Modify: `core/types.py`

**Interfaces:**
- Produces: `ConversationState.query_language: Optional[str]`, `ConversationState.search_query: Optional[str]`

- [ ] **Step 1: Add the fields**

In `core/types.py`, append to `ConversationState` (after `enrolled_course_ids`):

```python
    query_language: Optional[str]          # NEW — ISO code detected for the raw query; "fr" is default/fallback
    search_query: Optional[str]            # NEW — French text used for embedding/retrieval (set by detect_and_translate_query)
```

- [ ] **Step 2: Verify with a smoke import**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -c "
from core.types import ConversationState
print('query_language' in ConversationState.__annotations__)
print('search_query' in ConversationState.__annotations__)
"
```

Expected: `True` twice

- [ ] **Step 3: Commit**

```bash
cd /opt/craftpilot_backend
git add core/types.py
git commit -m "feat: add query_language and search_query to ConversationState"
```

---

## Task 3: `detect_and_translate_query` node

**Files:**
- Modify: `services/rag_service.py`
- Test: `tests/test_language_detection.py`

**Interfaces:**
- Consumes: `RAGConfig.enable_cross_lingual_detection` / `.langid_confidence_threshold` / `.min_langid_chars` (Task 1), `ConversationState.query_language` / `.search_query` (Task 2)
- Produces: `RAGService._initialize_langid(self)`, `RAGService.detect_and_translate_query(self, state: ConversationState) -> Dict[str, Any]` returning `{"query_language": str, "search_query": str}`

- [ ] **Step 1: Write the failing tests**

Create `/opt/craftpilot_backend/tests/test_language_detection.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -m pytest tests/test_language_detection.py -v 2>&1 | head -30
```

Expected: `AttributeError: 'RAGService' object has no attribute 'detect_and_translate_query'`

- [ ] **Step 3: Install `py3langid` in the conda env**

```bash
/root/miniconda3/envs/moodle_backend/bin/python -m pip install py3langid
```

- [ ] **Step 4: Implement `_initialize_langid` and `detect_and_translate_query`**

In `services/rag_service.py`, add the initializer near `_initialize_cross_encoder` (same file, method order doesn't matter — place it right after `_initialize_cross_encoder`, around line 260+):

```python
    def _initialize_langid(self):
        """Load py3langid with a normalized-probability identifier.

        The bare module-level `py3langid.classify()` returns unnormalized
        log-probabilities, not a usable [0, 1] confidence — the
        LanguageIdentifier instance with norm_probs=True is required for the
        confidence threshold in detect_and_translate_query to mean anything.
        """
        try:
            import py3langid as langid
            identifier = langid.LanguageIdentifier.from_modelstring(
                langid.model, norm_probs=True
            )
            logger.info("py3langid initialized (normalized probabilities)")
            return identifier
        except Exception as e:
            logger.error(f"py3langid initialization failed: {e} — cross-lingual detection disabled")
            return None
```

Wire it into `__init__` (after `self.cross_encoder = self._initialize_cross_encoder()`, line 57):

```python
        self.cross_encoder = self._initialize_cross_encoder()
        self._langid = self._initialize_langid()
```

Add the node method (place it right before `retrieve_initial`, e.g. just above line 921's PRF section comment):

```python
    @traceable(name="detect_and_translate_query", run_type="chain")
    def detect_and_translate_query(self, state: ConversationState) -> Dict[str, Any]:
        """Pipeline step 0 — language detection + French translation.

        French queries (the common case) pass through untouched with zero LLM
        calls. Non-French queries get one LLM translation call so every
        downstream retrieval node can keep embedding French text — the corpus
        and refine_query_prf's prompt are both French-only, so this reuses
        that already-tuned pipeline instead of asking it to also handle
        translation.

        Every failure path (langid unavailable, low confidence, short query,
        translation error) degrades to {"query_language": "fr",
        "search_query": <original>} — i.e. today's existing behavior.
        """
        original_query = str(state["messages"][-1].content)

        if self._langid is None:
            return {"query_language": "fr", "search_query": original_query}

        lang, confidence = self._langid.classify(original_query)

        if (
            lang == "fr"
            or confidence < self.config.langid_confidence_threshold
            or len(original_query) < self.config.min_langid_chars
        ):
            return {"query_language": "fr", "search_query": original_query}

        search_query = original_query
        try:
            translate_prompt = (
                "Traduis la question suivante en français, en conservant tout son sens "
                "technique et son intention.\n\n"
                f'Question originale ({lang}) :\n"{original_query}"\n\n'
                "Réponds avec UNIQUEMENT la traduction française, sans explication."
            )
            response = self.llm.invoke(translate_prompt)
            if isinstance(response.content, str):
                translated = response.content.strip()
            elif isinstance(response.content, list):
                translated = " ".join(str(item) for item in response.content).strip()
            else:
                translated = str(response.content).strip()

            if translated:
                search_query = translated
                logger.info(f"detect_and_translate_query: [{lang}] '{original_query}' -> '{translated}'")
            else:
                logger.warning("detect_and_translate_query: empty translation — using original query")
        except Exception as e:
            logger.error(f"detect_and_translate_query: translation failed: {e} — using original query")

        # query_language is trusted independently of translation success — a
        # failed translation shouldn't also force a French-language answer to
        # a question we know was asked in another language.
        return {"query_language": lang, "search_query": search_query}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -m pytest tests/test_language_detection.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 6: Commit**

```bash
cd /opt/craftpilot_backend
git add services/rag_service.py tests/test_language_detection.py
git commit -m "feat: add detect_and_translate_query pipeline node"
```

---

## Task 4: Wire `search_query` into retrieval nodes

**Files:**
- Modify: `services/rag_service.py`
- Test: `tests/test_language_detection.py` (append)

**Interfaces:**
- Consumes: `ConversationState.search_query` (Task 2/3)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_language_detection.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -m pytest tests/test_language_detection.py -v -k "search_query or refine_query_prf_grounds or retrieve_final_dual_falls"
```

Expected: FAIL — assertions comparing against the raw English message instead of the French `search_query`

- [ ] **Step 3: Update `retrieve_initial`**

In `services/rag_service.py`, `retrieve_initial` (around line 955), change:

```python
        query = str(state.get("messages")[-1].content)
```
to:
```python
        query = state.get("search_query") or str(state.get("messages")[-1].content)
```

- [ ] **Step 4: Update `refine_query_prf`**

Around line 1000, change:
```python
        original_query = str(state.get("messages")[-1].content)
```
to:
```python
        original_query = state.get("search_query") or str(state.get("messages")[-1].content)
```

- [ ] **Step 5: Update `retrieve_final_dual`**

Around line 1056, change:
```python
        refined_query = state.get("refined_query") or str(state.get("messages")[-1].content)
```
to:
```python
        refined_query = (
            state.get("refined_query")
            or state.get("search_query")
            or str(state.get("messages")[-1].content)
        )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -m pytest tests/test_language_detection.py -v
```

Expected: all 11 tests PASS

- [ ] **Step 7: Commit**

```bash
cd /opt/craftpilot_backend
git add services/rag_service.py tests/test_language_detection.py
git commit -m "feat: retrieval nodes embed search_query instead of raw message"
```

---

## Task 5: Answer-language switch in `_build_messages`

**Files:**
- Modify: `services/rag_service.py`
- Test: `tests/test_language_detection.py` (append)

**Interfaces:**
- Consumes: `ConversationState.query_language` (Task 2/3)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_language_detection.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -m pytest tests/test_language_detection.py -v -k build_messages
```

Expected: `test_build_messages_swaps_rule_for_non_french_query` FAILs (rule never swaps today); the other three should already pass (confirms the `<query>` tag and default-French behavior are untouched by this task).

- [ ] **Step 3: Implement the swap in `_build_messages`**

In `services/rag_service.py`, `_build_messages` (starting line 125), insert the rule swap right before the `if self.system_prompt and self.user_template:` check (line 147):

```python
        query_language = state.get("query_language")
        if query_language and query_language != "fr":
            system_prompt = self.system_prompt.replace(
                "- Répondez TOUJOURS en français correct et soigné, sans fautes d'orthographe ni de grammaire.\n",
                "- Répondez TOUJOURS dans la même langue que la question de l'apprenti "
                "(ci-dessous), avec une orthographe et une grammaire soignées.\n",
            )
        else:
            system_prompt = self.system_prompt
```

Then change the return statement on line 153 from:
```python
            return [SystemMessage(content=self.system_prompt), HumanMessage(content=user_text)]
```
to:
```python
            return [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -m pytest tests/test_language_detection.py -v
```

Expected: all 15 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /opt/craftpilot_backend
git add services/rag_service.py tests/test_language_detection.py
git commit -m "feat: answer in the query's language instead of always French"
```

---

## Task 6: Pipeline wiring

**Files:**
- Modify: `pipeline.py`
- Modify: `tests/test_pipeline_integration.py`

**Interfaces:**
- Consumes: `RAGService.detect_and_translate_query` (Task 3), `RAGConfig.enable_cross_lingual_detection` (Task 1)
- Produces: `stream_response` runs `detect_and_translate_query` first; emits `{"event": "status", "data": "Traduction de la question…"}` only on the non-French branch

- [ ] **Step 1: Update the existing test that will break**

`test_stream_response_does_not_short_circuit_in_domain` in `tests/test_pipeline_integration.py` fully mocks `pipeline.rag_service` — without mocking the new node, `state.update(MagicMock())` will raise `TypeError`. Update it (around line 134, before the existing `pipeline.rag_service.retrieve_initial = ...` line):

```python
    pipeline.rag_service.detect_and_translate_query = MagicMock(return_value={
        "query_language": "fr", "search_query": "Comment souffler le verre ?",
    })
    pipeline.rag_service.retrieve_initial = MagicMock(return_value={
        "context": [], "video_metadata": None, "refined_query": None,
        "hypothetical_document": None, "enhanced_query": None, "query_variants": [],
    })
```

(leave the rest of that test unchanged)

- [ ] **Step 2: Write the new failing tests**

Append to `tests/test_pipeline_integration.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -m pytest tests/test_pipeline_integration.py -v -k "detect_and_translate or kill_switch or skips_translation_status"
```

Expected: FAIL — `detect_and_translate_query` is never called by `stream_response` yet

- [ ] **Step 4: Wire the node into `stream_response`**

In `pipeline.py`, after the initial `state` dict is built (after line 416) and before the `# --- PRF step 1: initial retrieval ---` comment (line 418), insert:

```python
            # --- Step 0: language detection + translation ---
            if self.rag_service.config.enable_cross_lingual_detection:
                result = await asyncio.to_thread(self.rag_service.detect_and_translate_query, state)
                state.update(result)
                if state.get("query_language") and state["query_language"] != "fr":
                    yield json.dumps({"event": "status", "data": "Traduction de la question…"}) + "\n"
```

Also add the two new fields to the initial `state` dict literal (in the block starting at line 403), so they exist even when the kill-switch is off:

```python
            state: Dict[str, Any] = {
                "messages": [HumanMessage(content=message)],
                "selected_domain": selected_domain,
                "course_id": course_id,
                "context": [],
                "video_metadata": None,
                "refined_query": None,
                "hypothetical_document": None,
                "enhanced_query": None,
                "query_variants": [],
                "route": None,
                "user_cohort_ids": user_cohort_ids,
                "enrolled_course_ids": enrolled_course_ids,
                "query_language": None,           # NEW
                "search_query": None,              # NEW
            }
```

- [ ] **Step 5: Add the node to `_build_conversation_graph`'s function list**

In `pipeline.py`, `_build_conversation_graph` (line 131-139), add it as the first entry:

```python
            return self.graph_service.build_conversation_graph(
                functions=[
                    "detect_and_translate_query",
                    "retrieve_initial",
                    "refine_query_prf",
                    "retrieve_final_dual",
                    "rerank",
                    "generate",
                ]
            ).compile_graph()
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -m pytest tests/test_pipeline_integration.py -v
```

Expected: all tests PASS (existing + 3 new)

- [ ] **Step 7: Run the full test suite**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -m pytest tests/ -v
```

Expected: all tests PASS, no regressions in unrelated test files

- [ ] **Step 8: Commit**

```bash
cd /opt/craftpilot_backend
git add pipeline.py tests/test_pipeline_integration.py
git commit -m "feat: wire detect_and_translate_query into stream_response with kill-switch"
```

---

## Task 7: Eval validation — Config D + ambiguous-query fixture

**Files:**
- Modify: `eval/09_cross_lingual_eval.py`
- Create: `eval/fixtures/ground_truth_ambiguous.json`

**Interfaces:**
- Consumes: `RAGService.detect_and_translate_query` (Task 3)
- Produces: `eval/results/config_d_en_results.json`

This task validates the *numbers*, not unit-level correctness (already covered by Tasks 3-6's tests). It requires real API credentials, so — same constraint as the original eval run — it must be executed with `/root/miniconda3/envs/moodle_backend/bin/python`, which now works since Task 3 installed `py3langid` there and the ACL grant gives read access to the whole env.

- [ ] **Step 1: Create the ambiguous-query fixture**

Create `/opt/craftpilot_backend/eval/fixtures/ground_truth_ambiguous.json`:

```json
[
  {"qid": "amb_1", "query": "ok", "expected_language": "fr"},
  {"qid": "amb_2", "query": "merci", "expected_language": "fr"},
  {"qid": "amb_3", "query": "flamme", "expected_language": "fr"},
  {"qid": "amb_4", "query": "verre", "expected_language": "fr"},
  {"qid": "amb_5", "query": "oui", "expected_language": "fr"},
  {"qid": "amb_6", "query": "biseau ?", "expected_language": "fr"},
  {"qid": "amb_7", "query": "et la molette ?", "expected_language": "fr"},
  {"qid": "amb_8", "query": "pourquoi ça casse", "expected_language": "fr"},
  {"qid": "amb_9", "query": "comment ça marche", "expected_language": "fr"},
  {"qid": "amb_10", "query": "combien de temps", "expected_language": "fr"}
]
```

- [ ] **Step 2: Add Config D to the eval script**

In `eval/09_cross_lingual_eval.py`, add after `run_config_b`:

```python
def run_config_d(rag, query):
    """Config D: translate-first — detect_and_translate_query -> retrieve_initial
    -> refine_query_prf -> retrieve_final_dual."""
    state = make_state(query)

    s0 = rag.detect_and_translate_query(state)
    state0 = {**state, **s0}

    s1 = rag.retrieve_initial(state0)
    initial_context = s1.get('context', [])

    state2 = {**state0, **s1}
    s2 = rag.refine_query_prf(state2)
    refined_query = s2.get('refined_query', query)

    state3 = {**state2, **s2}
    s3 = rag.retrieve_final_dual(state3)
    final_docs = s3.get('context', [])

    return final_docs, refined_query, initial_context
```

In `main()`, after the Config B run/save block, add:

```python
    results_d = run_config('D', rag, ground_truth, lambda rag, q: run_config_d(rag, q))
    with open(os.path.join(RESULTS_DIR, 'config_d_en_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_d, f, ensure_ascii=False, indent=2)
    print("Saved config_d_en_results.json")
```

And extend the summary printout — after the existing `row("EN - B (PRF)", en_b)` line, add:

```python
    en_d = results_d['aggregate']
    print(row("EN - D (translate)", en_d))
```

- [ ] **Step 3: Run the eval**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python eval/09_cross_lingual_eval.py 2>&1 | tail -60
```

Success criterion from the spec: **EN-D MAP should approach FR-A's 0.667** (today's EN-B/PRF-only reaches 0.303). If it lands meaningfully short of that, treat it as a signal to revisit `langid_confidence_threshold` / the translation prompt before merging — not a blocker to fix in this same task, but flag it in the PR description either way.

- [ ] **Step 4: Spot-check the ambiguous-query fixture doesn't over-trigger translation**

```bash
cd /opt/craftpilot_backend && /root/miniconda3/envs/moodle_backend/bin/python -c "
import json, sys
sys.path.insert(0, '.')
from config.settings import ConfigurationManager
from services.rag_service import RAGService
from langchain_core.messages import HumanMessage

rag = RAGService(ConfigurationManager())
fixtures = json.load(open('eval/fixtures/ground_truth_ambiguous.json'))
wrong = 0
for item in fixtures:
    state = {'messages': [HumanMessage(content=item['query'])]}
    result = rag.detect_and_translate_query(state)
    ok = result['query_language'] == item['expected_language']
    if not ok:
        wrong += 1
    print(f\"{'OK ' if ok else 'BAD'} [{item['qid']}] '{item['query']}' -> {result['query_language']}\")
print(f'{wrong}/{len(fixtures)} misclassified')
"
```

Expected: `0/10 misclassified` (or close — if several ambiguous French utterances misclassify, lower `langid_confidence_threshold` or raise `min_langid_chars` in Task 1's config before merging)

- [ ] **Step 5: Commit**

```bash
cd /opt/craftpilot_backend
git add eval/09_cross_lingual_eval.py eval/fixtures/ground_truth_ambiguous.json eval/results/config_d_en_results.json
git commit -m "test: add Config D translate-first eval and ambiguous-query fixture"
```

---

## Self-Review Checklist

- [x] `py3langid` uses the normalized-probability `LanguageIdentifier` instance, not the raw module function — Task 3 Step 4 ✓
- [x] Every failure path (langid unavailable, low confidence, short query, translation error) degrades to `query_language="fr"` / `search_query=original` — covered by 5 of the 7 tests in Task 3 ✓
- [x] `search_query` never leaks into the `<query>` tag shown to the LLM — explicit test in Task 5 Step 1 ✓
- [x] All three retrieval nodes (`retrieve_initial`, `refine_query_prf`, `retrieve_final_dual`) read `search_query` with a fallback chain to the raw message — Task 4 ✓
- [x] Kill-switch (`enable_cross_lingual_detection=False`) skips the node entirely, not just the translation call — Task 6 Step 4 + explicit test ✓
- [x] Status event only fires on the non-French branch — explicit test in Task 6 ✓
- [x] Existing test `test_stream_response_does_not_short_circuit_in_domain` updated so it doesn't break from the new node — Task 6 Step 1 ✓
- [x] `MAX_RERANK_CANDIDATES` / rerank latency path untouched — no task modifies it ✓
- [x] Eval validates the actual numeric goal (EN-D MAP ≈ FR-A MAP), not just unit-level correctness — Task 7 ✓
