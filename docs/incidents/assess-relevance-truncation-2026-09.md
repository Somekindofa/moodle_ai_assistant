# Relevance Classifier Misjudged Well-Matched Content as Insufficient — Fixed (September 2026)

Symptom: the CraftPilot chat returned its canned "no relevant information"
fallback (`RAGService.INSUFFICIENT_CONTEXT_MESSAGE`) even when retrieval
had found and reranked exactly the right content with high confidence
(rerank score 0.92+). Reproduced live via a real Moodle course/page save —
see `docs/PLAYWRIGHT_DEBUGGING.md` for how that test rig works.

Found while validating the cross-lingual ingestion-translation feature
(`docs/superpowers/specs/2026-08-18-cross-lingual-retrieval-design.md`):
an English and a Greek page about glassblowing furnace safety were saved,
correctly translated to French at ingestion time, and retrieved with a
top rerank score of 0.924 for a matching French query — yet the chat still
answered "not found."

## Root Cause and Fix

Fix is in `services/rag_service.py`.

| Root cause | Fix |
|-----------|-----|
| `assess_relevance` — the LLM classifier that runs *after* retrieval/rerank to decide SUFFICIENT / AMBIGUOUS / INSUFFICIENT — built its prompt from `doc.page_content[:300]`, a **300-character** preview per document. `SemanticChunker` (`services/course_rag_service.py`) targets **~1600 chars** per chunk (`TARGET_TOKENS=400`). Any chunk near that target size has most of its content — potentially the specific fact the query asked about — silently cut from what the classifier sees, with no signal to the LLM that anything was truncated. | Added `RELEVANCE_PREVIEW_CHARS = 1600`, matching the chunker's target size, with a comment cross-referencing `course_rag_service.TARGET_TOKENS` so the two don't drift apart again if either changes. |

## Evidence

The two real ingested chunks that triggered this (course 108, a throwaway
test course) were 672 and 646 characters — both **within** the chunker's
normal target size, not pathologically long. In both, the specific fact
needed to answer a two-part question ("what temperature range, and why
must apprentices never work alone near the glory hole") sat past the old
300-char cutoff:

- English-sourced chunk: preview cut off mid-sentence at "...températures"
  — the temperature figures (`1090 à 1150 degrés Celsius`, offset 304) and
  the entire glory-hole safety rule were both outside the window.
- Greek-sourced chunk: temperature figures just barely made it in (offset
  260), but the glory-hole rule was still entirely cut off.

**Backend log signature** (`/tmp/craftpilot_backend.log`) — this is the
tell to look for if this class of bug recurs: after a good rerank, only
**one** `chat/completions` call appears (the `assess_relevance` judgment)
and the request ends there with the deterministic fallback, instead of the
**two** calls (`assess_relevance` + the actual `generate` call) that show
up on a real answer.

```
# Broken (before fix) — one call, then nothing:
rerank (remote): 5 candidates → 5 passed threshold=0.1 (top score=0.924, ...)
httpx: POST .../chat/completions "200 OK"
POST /api/chat HTTP/1.1 200 OK          # ← ends here, fallback message sent

# Fixed — two calls:
rerank (remote): 5 candidates → 5 passed threshold=0.1 (top score=0.930, ...)
httpx: POST .../chat/completions "200 OK"   # assess_relevance
httpx: POST .../chat/completions "200 OK"   # generate — the real answer
```

## Verification

- New test `test_assess_relevance_prompt_includes_answer_past_300_chars`
  (`tests/test_pipeline_integration.py`) — constructs a >300-char document
  with a marker string past the old cutoff, deliberately absent from the
  query text (an earlier draft of this test accidentally put the same
  phrase in both the query and the document, which made the assertion
  pass regardless of truncation — worth remembering if writing a similar
  test). Fails on the pre-fix code, passes after.
- Full existing suite re-run clean: no regressions (a handful of
  pre-existing unrelated failures — missing test env vars, missing
  `respx` dependency — confirmed identical via `git stash` before/after).
- Live re-verification after restarting `craftpilot-backend.service`: the
  same French query against the same course now returns a real, correctly
  grounded answer (temperature range + full glory-hole safety explanation)
  instead of the fallback.

## Key invariant to preserve

`RELEVANCE_PREVIEW_CHARS` (`rag_service.py`) must stay `>=` the chunker's
effective max chunk size (`TARGET_TOKENS` in `course_rag_service.py`,
currently ~1600 chars). If `TARGET_TOKENS` is ever raised, bump
`RELEVANCE_PREVIEW_CHARS` to match — otherwise this bug comes back for
whatever fraction of chunks land near the new, larger target size.
