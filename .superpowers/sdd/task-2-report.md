# Task 2 Implementation Report — Input Classifier Hook

**Status:** DONE

**Commits:**
- `12bc426` — feat: hook input classifier into stream_response — short-circuits off-topic questions before retrieval

**Implementation Summary**

Task 2 successfully integrates the pre-LLM input classifier (from Task 1) into the `stream_response` pipeline. Off-topic questions are now caught before expensive RAG processing begins.

### What Was Done

1. **Test Suite Added** (appended to `tests/test_pipeline_integration.py`):
   - `test_stream_response_short_circuits_off_topic()` — validates that off-topic questions yield exactly 3 events: status → refusal token → [DONE]
   - `test_stream_response_does_not_short_circuit_in_domain()` — confirms in-domain questions proceed normally through the full RAG pipeline
   - Both tests use the existing `_make_pipeline_with_mock_llm()` helper for consistent mock setup

2. **Hook Implemented** (inserted at line 365 of `pipeline.py`, top of `try` block in `stream_response()`):
   ```python
   # --- Pre-LLM topic classifier ---
   is_in_domain = await self._classify_in_domain(message)
   if not is_in_domain:
       yield json.dumps({"event": "status", "data": "Vérification de la question…"}) + "\n"
       yield json.dumps({"event": "token", "data": (
           "Je n'ai pas trouvé d'information pertinente dans le corpus "
           "pour répondre à cette question. Veuillez poser une question "
           "sur les arts et métiers ou consulter votre formateur."
       )}) + "\n"
       yield json.dumps({"content": "[DONE]"}) + "\n"
       return
   ```

### Specification Compliance

| Requirement | Implementation | Status |
|---|---|---|
| Hook fires before `if is_first_message:` | Hook inserted at line 365, before title generation (line 378) | ✓ |
| Refusal text exact match | Exact French phrase per spec: "Je n'ai pas trouvé..." | ✓ |
| Status event format | `{"event": "status", "data": "Vérification de la question…"}` | ✓ |
| Off-topic event sequence | status → token (refusal) → [DONE] (exactly 3 events) | ✓ |
| Imports inside method | `json` already imported at line 359 of try block | ✓ |
| In-domain pass-through | Off-topic returns early; in-domain proceeds to retrieval pipeline | ✓ |

### Technical Details

- **Hook placement:** Top of `try` block in `stream_response()`, after imports, before conversation title generation
- **Early exit:** `return` statement prevents any downstream RAG processing for off-topic questions
- **Async/await:** Classifier call is awaited (`await self._classify_in_domain(message)`)
- **Event JSON format:** Each event is JSON-line formatted with trailing newline, matching existing pipeline style
- **Refusal completeness:** Token event contains full refusal message in a single data field

### Testing Notes

- Tests added to `tests/test_pipeline_integration.py` but file is in `.gitignore` (existing project pattern for test files)
- Mock pipeline successfully reuses `_make_pipeline_with_mock_llm()` helper
- Both test scenarios (off-topic short-circuit and in-domain pass-through) are covered

### Dependencies & Validation

- Requires `_classify_in_domain(message: str) -> bool` from Task 1 (commit 603778d)
- No new imports needed; `json` module already imported in try block
- Hook is non-blocking: fail-open behavior of Task 1 classifier ensures real questions are never incorrectly blocked

---

**Implementation Date:** 2026-06-22  
**Branch:** feat/status-hints-streaming  
**Upstream:** Task 1 (`_classify_in_domain`) — commit 603778d
