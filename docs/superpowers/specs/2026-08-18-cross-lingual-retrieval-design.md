# Cross-Lingual Retrieval — Design Spec
**Date:** 2026-08-18
**Status:** Draft, pending review

---

## Problem

CraftPilot's vector store (`bge_multilingual_gemma2` embeddings) and PRF pipeline are tuned exclusively around French. Both the corpus and every hardcoded prompt in `refine_query_prf` assume the query is French. There is currently no handling — deliberate or accidental — for a query asked in another language.

**Measured impact** (`eval/09_cross_lingual_eval.py`, 33 non-adversarial glassblowing queries hand-translated to English, same `relevant_annotation_ids`/`relevant_course_sources` as the French ground truth):

| Config | MAP (FR) | MAP (EN) |
|---|---|---|
| A — raw `similarity_search` | 0.6667 | 0.2273 |
| B — PRF (`retrieve_initial` → `refine_query_prf` → `retrieve_final_dual`) | 0.6364 | 0.3030 |

Two findings:
1. Raw cross-lingual embedding retrieval is far weaker than same-language retrieval (0.227 vs 0.667 MAP) despite the embedding model's multilingual claims.
2. `refine_query_prf`'s hardcoded-French prompt implicitly retranslates non-French queries as a side effect of "grounding the query in corpus vocabulary" — but it does so using whatever (often wrong-topic) documents the weak initial retrieval handed it, so the result is an unreliable partial rescue (0.303), not a fix. `refine_query_prf` also fully replaces the query for final retrieval with no fallback to the original.

**Goal:** any query, in any language, should retrieve at French-corpus quality, and the assistant should answer back in the query's language.

---

## Chosen Approach

**Approach A — translate-to-French front node**, gated by a near-zero-cost local language check so the (large majority) French traffic pays no latency or behavior change.

Rejected alternatives:
- **Widen the retrieval net + rely on the reranker, no translation** — `bge-reranker-v2-m3` is multilingual, but `MAX_RERANK_CANDIDATES=5` is capped specifically because reranking takes ~5s/pair on this 2-core box (pipeline.py:36); widening the candidate set enough to compensate for weak cross-lingual recall breaks the latency budget. Doesn't address answer-language either.
- **Ensemble (translated + raw query in parallel)** — hedges against bad translations, but adds real complexity for a failure mode that's already rarer and cheaper to catch than PRF's current hallucination problem. Noted as a v2 candidate, not v1.

---

## Data Flow

```
Before:
  messages[-1].content ──> retrieve_initial ──> refine_query_prf ──> retrieve_final_dual ──> rerank ──> generate
                              (embeds raw query,                        (embeds refined_query,
                               any language)                             hardcoded-French prompt)

After:
  messages[-1].content ──> detect_and_translate_query ──> retrieve_initial ──> refine_query_prf ──> retrieve_final_dual ──> rerank ──> generate
                              │                                 (embeds search_query)   (embeds search_query)
                              ├─ query_language: "fr" (default/fallback)
                              └─ search_query: French version used by every retrieval node
                                                                                                              generate reads
                                                                                                              query_language to
                                                                                                              pick the answer-
                                                                                                              language rule
```

French queries (the common case) pass through `detect_and_translate_query` with zero LLM calls and zero change to today's behavior.

---

## Section 1 — `detect_and_translate_query` node

New method on `RAGService`, new first step in `stream_response` (pipeline.py) and first entry in the `functions=[...]` list passed to `_build_conversation_graph` (pipeline.py:132).

**Library:** `py3langid` (pure Python, no compiled model file to fetch separately, sub-millisecond inference, returns ISO 639-1 code + confidence). Needs `pip install py3langid` in the `moodle_backend` conda env — an operational step, not something I can run myself (same permission boundary as `.env`).

**Logic:**
```python
def detect_and_translate_query(self, state: ConversationState) -> Dict[str, Any]:
    original_query = str(state["messages"][-1].content)

    if self._langid is None:  # import/load failed at __init__ — fail safe
        return {"query_language": "fr", "search_query": original_query}

    lang, confidence = self._langid.classify(original_query)
    if lang == "fr" or confidence < LANGID_CONFIDENCE_THRESHOLD or len(original_query) < MIN_LANGID_CHARS:
        return {"query_language": "fr", "search_query": original_query}

    try:
        translate_prompt = (
            "Traduis la question suivante en français, en conservant tout son sens "
            "technique et son intention.\n\n"
            f'Question originale ({lang}) :\n"{original_query}"\n\n'
            "Réponds avec UNIQUEMENT la traduction française, sans explication."
        )
        response = self.llm.invoke(translate_prompt)
        translated = _extract_text(response).strip()
        search_query = translated if translated else original_query
    except Exception as e:
        logger.error(f"detect_and_translate_query: translation failed: {e} — using original query")
        search_query = original_query

    return {"query_language": lang, "search_query": search_query}
```

`LANGID_CONFIDENCE_THRESHOLD` and `MIN_LANGID_CHARS` (~15) default to conservative values that bias toward "fr" — a missed non-French query degrades to today's baseline; a French query misflagged as non-French only costs one harmless near-no-op translation call.

**`core/types.py` — `ConversationState` gains:**
```python
query_language: Optional[str]   # ISO code detected for the raw query; "fr" is the default/fallback
search_query: Optional[str]     # French text actually used for embedding/retrieval
```

---

## Section 2 — Retrieval nodes read `search_query`

`retrieve_initial`, `refine_query_prf`, `retrieve_final_dual` (and `retrieve_with_hyde`/`generate_hypothetical_document` if kept) change their query source from:
```python
str(state.get("messages")[-1].content)
```
to:
```python
state.get("search_query") or str(state.get("messages")[-1].content)
```
One-line change per call site. `refine_query_prf`'s corpus-grounding logic is otherwise untouched — it keeps operating on French input exactly as it does today (0.636 MAP), which is the entire point: reuse the pipeline that already works, instead of asking it to also handle translation.

---

## Section 3 — Answer-language switch

`self.system_prompt` (rag_service.py:67-98) is built once at `__init__` and hardcodes `"Répondez TOUJOURS en français correct et soigné..."` (line 72). `_build_messages` (line 125) is where it's turned into a `SystemMessage` per request — that's the injection point.

When `state.get("query_language")` is non-French, that one rule line is swapped for a language-agnostic instruction ("always answer in the same language as the apprentice's question") before constructing the `SystemMessage`. Every other instruction in the prompt (structure rules, the mandatory "Pour aller plus loin" A/B/C section, markdown rules) is left untouched — those govern the model's *behavior*, not its output language, and the underlying LLM follows French meta-instructions to produce non-French output without issue.

**Explicit v1 scope boundary:** a few other strings stay hardcoded French regardless of `query_language`:
- The pre-LLM off-topic refusal (pipeline.py:388-392) — it runs *before* `detect_and_translate_query`, so it can't know the language yet.
- The "no relevant info in corpus" fallback embedded in the system prompt (line 77-79).

Result: a non-French user gets a correctly-localized answer when the RAG pipeline succeeds, but may occasionally see a French refusal/fallback message when it doesn't. Acceptable v1 boundary; flagged rather than silently left ambiguous.

---

## Section 4 — Status event & kill-switch

One new line in `stream_response`, emitted only on the non-French branch:
```python
yield json.dumps({"event": "status", "data": "Traduction de la question…"}) + "\n"
```

**Kill-switch:** `enable_cross_lingual_detection: bool` in `config/settings.py`, checked once in `pipeline.py` before calling `detect_and_translate_query`. Lets this be disabled in production instantly (no redeploy) if `py3langid` or the translation step misbehaves — this touches a live pipeline serving real apprentices.

---

## Section 5 — Failure Modes

| Scenario | Behaviour |
|---|---|
| `py3langid` not installed / fails to load at startup | Node is a no-op: `query_language="fr"`, `search_query=original`. Identical to today's pipeline. |
| Short or low-confidence query | Defaults to `"fr"` — never spuriously translates real French traffic. |
| Detected non-French, but LLM translation call fails/empty | `query_language` keeps the detected code (answer still localizes); `search_query` falls back to the raw original (today's weaker cross-lingual embedding path, not a crash). |
| Query actually non-French but misclassified as French | Flows through unchanged — no worse than today's baseline for every query. |
| `enable_cross_lingual_detection=False` | Node skipped entirely; pipeline behaves exactly as before this feature existed. |

**Guiding rule:** every failure path degrades toward today's existing (already-shipped) behavior, never toward something worse.

---

## Section 6 — Testing / Eval Plan

1. Extend `eval/09_cross_lingual_eval.py` with **Config D**: `detect_and_translate_query → retrieve_initial → refine_query_prf → retrieve_final_dual`, run against `fixtures/ground_truth_en.json`. **Success criterion: EN-D MAP should approach FR-A's 0.667** (today's EN-B/PRF-only reaches 0.303).
2. New fixture: ~10 short/ambiguous French utterances (`"ok"`, `"merci"`, `"flamme"`) to confirm the confidence/length gate doesn't trigger spurious translation calls on real French traffic.
3. Answer-language is a generation-quality property, not a retrieval metric — the eval harness never calls `generate`. Verify separately with a handful of English queries through the *full* streamed pipeline, checking the response text's language (manual or a small script) before rollout.

---

## Out of Scope (v1)

- Approach C (parallel raw-query ensemble alongside the translated query) — noted as a v2 hardening step if Config D's numbers don't fully close the gap.
- Localizing the hardcoded-French off-topic refusal and "no relevant info" fallback strings (Section 3).
- Localizing the transient status-hint UI text (spinner labels stay French for all languages, consistent with existing UI chrome).
- Fixing `refine_query_prf`'s -0.030 MAP regression on French-language queries — a separate, unrelated finding from this eval, not caused by and not fixed by this design.
