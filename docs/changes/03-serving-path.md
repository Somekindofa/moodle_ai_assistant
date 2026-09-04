# 03 — Serving path: user-facing messages, MMR fetch_k, orphan-label safety
**Agent 3 · 2026-09-04 · area: pipeline.py, services/rag_service.py, api/models.py**

Four independent defects on the query-serving path. Nothing in the retrieval
logic itself changed: two of these are about what the learner is shown when the
system declines to answer, one is a mis-sized MMR parameter, one is a latent
data-integrity hazard made visible.

---

## Task 1 — "unknown" leaked to learners

### Problem

`_build_ambiguous_clarification` (`pipeline.py`) builds the AMBIGUOUS
clarifying question by listing the topics actually retrieved:

```python
topic = doc.metadata.get("project_name") or doc.metadata.get("craft")
```

`AnnotationIngestRequest.project_name` (`api/models.py`) defaults to the
literal string `"unknown"` for annotations with no project, so `"unknown"` is a
value genuinely stored in ChromaDB metadata — not a missing key that `or` would
skip past. A real user was shown:

> Votre question peut correspondre à plusieurs sujets du corpus (**unknown**).
> Pourriez-vous préciser votre demande ?

— a topic that does not exist, phrased as if it did.

The same guard already existed twenty lines away in the same repo:
`ResyncProjectRequest.reject_placeholder_project` refuses `"unknown"` as a
resync target and documents at length why it is never a real project. The
clarification builder just never got it.

### What changed

- `api/models.py` now owns the placeholder concept, next to the field that
  produces it: `PLACEHOLDER_PROJECT_NAMES` (frozenset) and
  `is_placeholder_project_name(value)`. The helper takes `Any` (ChromaDB
  metadata is not schema-checked on read), treats `None`/blank as a
  placeholder, and never raises.
- `reject_placeholder_project` now reads from that constant instead of
  re-spelling `"unknown"`, so the two guards cannot drift apart. Behaviour is
  unchanged for `"unknown"` and blanks; it additionally rejects
  `none/null/n/a/na/-`, which widens a guard that protects a destructive
  delete — the safe direction.
- `pipeline._usable_topic(doc)` (new) applies the guard **per candidate**, so a
  document tagged `project_name="unknown", craft="glassblowing"` still
  contributes `glassblowing` — the placeholder suppresses the placeholder, not
  the document.
- When filtering leaves nothing nameable, the builder returns the generic
  "your question isn't specific enough" wording rather than an empty
  parenthesis. `"( )"` would be a worse lie than saying nothing.

### Why this approach

The alternative — filtering inline in `_build_ambiguous_clarification` with a
literal `!= "unknown"` — would have been three characters shorter and would have
recreated exactly the drift that caused the bug: two copies of the same rule,
one of which someone forgets. The constant lives beside the field that emits the
placeholder, which is the only place a future third caller would think to look.

### Evidence

See the Task 1 block under **Evidence** below: the exact reported string no
longer appears, the craft fallback still works, and real project names are
listed unchanged.

---

## Task 2 — refusal/clarification language

### Problem

The answer path already follows the learner's language: `query_language` is set
by `detect_and_translate_query`, and `_build_messages` swaps the system prompt's
"always answer in French" rule for "answer in the learner's language". Three
user-visible strings never went through the LLM and so never followed it:

1. the pre-LLM topic classifier's rejection (`pipeline.py`, inside
   `stream_response`),
2. `RAGService.INSUFFICIENT_CONTEXT_MESSAGE`, emitted at the INSUFFICIENT gate,
3. `_build_ambiguous_clarification`'s two strings.

Net effect: a Greek learner got Greek when retrieval worked and French exactly
when it did not — i.e. at the moment they most needed to understand the reply.

### What changed

- `services/rag_service.py` gains a module-level `USER_MESSAGES` table keyed by
  message id then ISO 639-1 code, plus `localized_message(key, language,
  **fields)`.
- `RAGService.INSUFFICIENT_CONTEXT_MESSAGE` is now
  `USER_MESSAGES["insufficient_context"]["fr"]` — **still a plain `str`**, see
  the constraint below. `RAGService.insufficient_context_message(language)`
  reaches the other languages.
- `RAGService.detect_query_language(text)` exposes the existing py3langid gate
  (`translation_service.decide_translation`) on its own, with no LLM call.
- `stream_response` detects the language once, before the topic classifier, and
  uses it for all three refusal paths.
- `_build_messages`: for a non-French query, the French refusal sentence baked
  into the system prompt is replaced with the localized one, so the "answer in
  the learner's language" rule and the literal sentence stop contradicting each
  other.

### The `rag_service.py:166` constraint (checked before touching it)

`INSUFFICIENT_CONTEXT_MESSAGE` is interpolated into `self.system_prompt` with an
f-string during `__init__`:

```python
"Si le contexte est insuffisant ..., répondez UNIQUEMENT : "
f"\"{self.INSUFFICIENT_CONTEXT_MESSAGE}\" "
```

That usage needs a plain string, and the surrounding prompt is French regardless
of the learner's language. So the attribute was left as a `str` holding the
French text, and localization was added *beside* it rather than by changing its
type. `insufficient_context_message()` returns
`self.INSUFFICIENT_CONTEXT_MESSAGE` for French/None specifically so an
instance-level override still wins — which is what
`tests/test_pipeline_integration.py:1175` relies on, and which keeps the French
path byte-for-byte what it was.

### Why a static table, not an LLM call

- A refusal is already the slow, disappointing path. A network round trip is the
  wrong thing to add to it.
- `translation_service.translate_to_french` returns `None` on failure. A failed
  translation of a refusal degrades to *no message at all* — the one output that
  must never be missing. A static string cannot fail.
- The strings are fixed and few. There is nothing to generate.

### Which languages, and what happens outside them

`fr`, `en`, `el` only. Those are what this deployment is actually known to
serve: French (UI, corpus, all prompts), and English + Greek, both attested in
`eval/fixtures/xling_annotations_seed.json`,
`eval/fixtures/xling_course_chunks_seed.json`, and the GR-Glassblowing course
(course 109). I deliberately did not add speculative languages: unverified
translations are content someone has to maintain and nobody can check.

Any other detected language falls back to **English** — the learner
demonstrably did not write French, so English is the better guess than French —
and logs a `WARNING` naming the ISO code. A genuine coverage gap therefore
appears in `/tmp/craftpilot_backend.log` as a specific, actionable line, and the
table gets extended from data instead of guesswork.

### One deliberate asymmetry

The language detection in `stream_response` is **not** gated on
`enable_cross_lingual_detection`. That kill-switch governs LLM *translation* of
queries; nothing on this path translates anything, and `decide_translation` is a
local sub-millisecond call. This also means the three refusal paths agree with
each other about the language of a given question even with the switch off,
which they would not if two read `state["query_language"]` and one did its own
detection. `refusal_language = state.get("query_language") or detected_language`
— with the switch on, the two are computed by the same function from the same
text and never disagree.

---

## Task 3 — MMR fetch_k

### Problem

```python
results = self.vector_store.max_marginal_relevance_search(query, k=k, **kwargs)
```

No `fetch_k`, so langchain's flat default of 20 applied — by accident, not by
design. `fetch_k` is MMR's candidate pool: it fetches `fetch_k` nearest
neighbours and then greedily picks `k` of them trading relevance against
redundancy. Two consequences:

- against the 16-document annotation collection, **every** query logged
  `Number of requested results 20 is greater than number of elements in index
  16, updating n_results = 16`
  (`chromadb/segment/impl/vector/local_persistent_hnsw.py:424`);
- at `k=15` (the configured `similarity_search_k`), a 20-document pool left MMR
  essentially nothing to diversify over — quietly, with no warning at all. That
  half was the more interesting one.

### What changed

`RAGService.MMR_FETCH_K_MULTIPLIER = 5`, `MMR_FETCH_K_CAP = 50`, and

```python
fetch_k = min(max(k * 5, k), 50)          # then clamped to the live collection size
```

via `_mmr_fetch_k(k)` / `_collection_count()`, passed explicitly to
`max_marginal_relevance_search`.

### Why these numbers

- **5×** mirrors langchain's own default ratio (`k=4`, `fetch_k=20`) and the
  usual dense-retrieval rule of thumb of 4–5×. At `fetch_k == k`, MMR
  degenerates into plain similarity search; 5× is enough headroom for the
  redundancy penalty to actually choose something.
- **Cap of 50** matters only once the corpus grows. `fetch_k` is how many full
  3584-dim embeddings Chroma materialises and returns per query, so 5 × 15 = 75
  of them to discard 60 is real bandwidth for no measurable diversity gain.
- **Clamp to collection size** is what actually silences the warning. Chroma
  compares the requested `n_results` against the *whole segment*, not against
  the cohort-filtered subset (`local_persistent_hnsw.py:421-425`), so the filter
  cannot be relied on to keep the number down.
- Small collection (16 docs): every `k` clamps to 16 — no warning, MMR sees the
  whole collection, which is the best it can do.
  Large collection: 25 / 50 / 50 for `k` = 5 / 10 / 15.
- `_collection_count()` returns `None` on any failure or a non-`int` (a mocked
  vector store, as `tests/test_cohort_filter.py` builds), and the clamp is
  simply skipped. It cannot break a query.

---

## Task 4 — orphaned-label limitation

### Problem (root cause already established, not re-investigated)

hnswlib allocates a monotonically increasing label per element ever added and
never reuses one; deleting a document frees its id in Chroma's metadata segment
but not its label in the vector segment. A segment whose label high-water mark
far exceeds its live count can fail to reload with `Cannot return the results in
a contigious 2D array` — and when it does, every vector query in that process
returns nothing, silently, for the life of the process. Unfixed upstream
(chroma-core/chroma#2620; PR #2621 closed unmerged; no chroma-hnswlib release
since 0.7.6).

The annotation collection reached 381 allocated labels against 16 addressable
documents; a later dedupe deleted ~365 and left the labels behind.
`stable_document_id` defused it — the sync is now an upsert in place that
allocates no new labels — so this is **latent, not active**.

`_clear_annotation_documents` does `get(where={"type": "video_annotation"})`
then `delete(ids=...)`. By construction it only ever touches ids Chroma can
still address. Orphaned labels have no id left to name them, are invisible to
`get()`, and survive the call untouched. It cannot be the repair tool, and
reaching for it as one adds a fresh round of deletes on top of the problem.

### What changed

Nothing was dropped, mutated, or written. The limitation was made **stated** and
**detectable**:

1. **Honest docstring** on `_clear_annotation_documents` saying plainly what it
   cannot do, that calling it on a broken collection will not help, and that the
   only remedy is an offline rebuild with the backend stopped (the local
   `PersistentClient` is not process-safe).
2. **`_hnsw_label_stats()`** — returns `(allocated_labels, live_documents)` or
   `None`. Chroma exposes no public API for the label high-water mark, so it
   reads `_total_elements_added` off the persistent-HNSW segment.
3. **`warn_if_hnsw_labels_orphaned(context)`** — logs at `ERROR` with the
   numbers and the consequence when `orphaned >= 32` **and**
   `allocated >= 2 × live`. Called at the end of `_clear_annotation_documents`
   and once from `pipeline._auto_sync_annotations`.

### Why it cannot crash or slow down startup

- It **peeks** at the segment manager's already-instantiated segment
  (`manager.segment_cache[VECTOR].get(id)` → `manager._instances.get(...)`) and
  never calls `get_segment()`, which would *build* one — building a persistent
  HNSW segment reads the metadata pickle and reloads the index from disk. If the
  vector segment has not been used yet, this returns `None` rather than doing
  I/O to find out. Verified below: before any query it returns `None`; after a
  normal query it returns real numbers.
- Every step is inside one `try/except` degrading to `None`. These are private,
  version-pinned chromadb internals (0.6.3); an upgrade that moves them must
  cost the diagnostic, never raise on a query or startup path. Verified against
  a `MagicMock` vector store: returns `None`, no exception.
- When it does run, it reads two in-memory values plus one SQLite `COUNT`.

### Threshold choice, and one honest caveat

Ratio rather than a bare difference: a handful of orphans is normal churn. The
absolute floor of 32 stops a tiny collection (2 live, 5 allocated) crying wolf.
The production shape — 384 allocated / 16 live — clears both by a wide margin.

Caveat, documented in the code: `_total_elements_added` only advances when a
write batch is flushed into the HNSW index, while `count()` includes the pending
batch. A freshly seeded 16-document collection therefore reads `0 allocated / 16
live`. That is correct for a *leak detector* — the signal is allocated running
far ahead of live — but it is not an accounting audit, and `allocated < live` is
normal rather than a second fault.

---

## Files touched

| File | Lines/functions | What changed |
|---|---|---|
| `api/models.py` | new `PLACEHOLDER_PROJECT_NAMES`, `is_placeholder_project_name()` | Placeholder project names get one shared definition, beside the field that emits `"unknown"`. |
| `api/models.py` | `ResyncProjectRequest.reject_placeholder_project` | Reads the shared constant instead of an inline `== "unknown"`; error message generalized to name the actual value. |
| `pipeline.py` | imports | `+ localized_message` from `services.rag_service`; `+ is_placeholder_project_name` from `api.models`. |
| `pipeline.py` | new `_usable_topic()` | Per-candidate placeholder filter; craft still used when project_name is a placeholder. |
| `pipeline.py` | `_build_ambiguous_clarification()` | Takes `language`; filters placeholders; both strings come from `USER_MESSAGES`. |
| `pipeline.py` | `_auto_sync_annotations()` | Calls `warn_if_hnsw_labels_orphaned("startup annotation sync")`. |
| `pipeline.py` | `stream_response()` | Detects query language once up front; off-topic / INSUFFICIENT / AMBIGUOUS refusals all follow it. French INSUFFICIENT still reads the attribute directly. |
| `services/rag_service.py` | new `USER_MESSAGES`, `USER_MESSAGE_FALLBACK_LANG`, `localized_message()` | Static translation table (fr/en/el) + a never-raising accessor. |
| `services/rag_service.py` | `INSUFFICIENT_CONTEXT_MESSAGE` | Now sourced from the table's `fr` entry; still a plain `str` for the `__init__` f-string. |
| `services/rag_service.py` | new `insufficient_context_message()`, `detect_query_language()` | Localized refusal accessor; LLM-free language detection reusable before pipeline step 0. |
| `services/rag_service.py` | `_build_messages()` | For non-French queries, swaps the French refusal sentence inside the system prompt for the localized one. |
| `services/rag_service.py` | new `MMR_FETCH_K_MULTIPLIER`, `MMR_FETCH_K_CAP`, `_collection_count()`, `_mmr_fetch_k()` | Explicit, clamped MMR candidate pool. |
| `services/rag_service.py` | `similarity_search()` | Passes `fetch_k=self._mmr_fetch_k(k)`. |
| `services/rag_service.py` | new `HNSW_ORPHAN_MIN_LABELS`, `HNSW_ORPHAN_RATIO`, `_hnsw_label_stats()`, `warn_if_hnsw_labels_orphaned()` | Cheap, non-forcing, never-raising orphan-label detector. |
| `services/rag_service.py` | `_clear_annotation_documents()` | Honest docstring about what it cannot do; runs the detector afterwards. |

Diffstat:

```
 api/models.py           |  41 ++++-
 pipeline.py             | 117 ++++++++++++---
 services/rag_service.py | 389 ++++++++++++++++++++++++++++++++++++++++++++++--
 3 files changed, 515 insertions(+), 32 deletions(-)
```

---

## Evidence

All commands run as `claude-runner`, `PYTHONNOUSERSITE=1`, conda env
`moodle_backend`. Nothing touched the production ChromaDB collection; the
Chroma-backed checks below run against scratch collections under
`/tmp/claude-1000/.../scratchpad/`.

### CRLF preserved (checked before and after every edit)

```
$ file pipeline.py services/rag_service.py api/models.py
pipeline.py:             Python script, Unicode text, UTF-8 text executable, with CRLF line terminators
services/rag_service.py: Python script, Unicode text, UTF-8 text executable, with CRLF line terminators
api/models.py:           Python script, Unicode text, UTF-8 text executable, with CRLF line terminators

$ # count LF bytes not preceded by CR
pipeline.py loneLF 0
services/rag_service.py loneLF 0
api/models.py loneLF 0
```

(The diffstat above — 515 insertions — is the other half of that proof: a silent
CRLF→LF conversion would have rewritten all 2 500+ lines.)

### Compile

```
$ PYTHONNOUSERSITE=1 python3 -m py_compile pipeline.py services/rag_service.py api/models.py
COMPILE_OK
```

### Targeted tests

Baseline first, to establish what already fails. `git show HEAD:` copies of the
three files were symlink-overlaid into a scratch directory so the running
backend's sources were never reverted:

```
$ cd <scratch>/pristine && pytest tests/test_pipeline_integration.py tests/test_resync_guardrails.py \
      tests/test_cohort_filter.py tests/test_language_detection.py tests/test_merge_dedup.py -q
FAILED tests/test_pipeline_integration.py::test_hyde_generates_document - ope...
1 failed, 92 passed in 9.28s
```

After the change, same five files:

```
FAILED tests/test_pipeline_integration.py::test_hyde_generates_document - ope...
1 failed, 92 passed in 9.15s
```

Identical. The one failure is environmental and pre-existing — that test builds
a real `RAGService`, whose `_initialize_embeddings` needs `INFOMANIAK_API_KEY`
from `.env`, which this account cannot read:

```
WARNING  config.settings:settings.py:122 Missing required environment variables: INFOMANIAK_API_KEY, INFOMANIAK_PRODUCT_ID, LANGSMITH_API_KEY
ERROR    services.rag_service: Failed to initialize embeddings: The api_key client option must be set ...
openai.OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable
```

Widened run, final state:

```
$ pytest tests/test_pipeline_integration.py tests/test_resync_guardrails.py tests/test_cohort_filter.py \
      tests/test_language_detection.py tests/test_merge_dedup.py tests/test_document_ids.py \
      tests/test_ingest_annotation_silo.py -q
FAILED tests/test_pipeline_integration.py::test_hyde_generates_document - ope...
1 failed, 104 passed in 9.39s
```

**One regression was caught and fixed during this work.** Routing the
INSUFFICIENT emission through `self.rag_service.insufficient_context_message()`
broke `test_stream_response_insufficient_relevance_skips_videos_and_generation`:
that test mocks `rag_service` wholesale and overrides the
`INSUFFICIENT_CONTEXT_MESSAGE` *attribute*, so a method call returned a
`MagicMock` and `json.dumps` raised — which in the real system would abort the
whole stream and leave the learner with nothing. The French path now reads the
attribute directly (see Task 2), which restores the old contract exactly and
makes the failure mode impossible for the common case.

### Task 1 — placeholder filtering

```
is_placeholder_project_name samples:
   'unknown'                -> True
   'UNKNOWN'                -> True
   ' Unknown '              -> True
   ''                       -> True
   '   '                    -> True
   None                     -> True
   'n/a'                    -> True
   '-'                      -> True
   0                        -> False
   'LV Rivetage 2026'       -> False
   'glassblowing'           -> False

The exact case that reached a real user (project_name='unknown', no craft):
   OLD behaviour would have said: 'plusieurs sujets du corpus (unknown)'
   NEW: Votre question n'est pas assez précise pour que je trouve une réponse fiable dans le corpus. Pourriez-vous la reformuler avec plus de détails ?

Placeholder project but a real craft tag -> craft is used, not dropped:
    Votre question peut correspondre à plusieurs sujets du corpus (glassblowing, glovemaking). Pourriez-vous préciser votre demande ?

Real project names still listed unchanged:
    Votre question peut correspondre à plusieurs sujets du corpus (LV Rivetage 2026, Biseau 2026). Pourriez-vous préciser votre demande ?

No metadata at all -> generic wording, never an empty parenthesis:
    Votre question n'est pas assez précise pour que je trouve une réponse fiable dans le corpus. Pourriez-vous la reformuler avec plus de détails ?

   assertions OK
```

### Task 2 — localized refusals

```
--- fr ---
  off_topic          : Je n'ai pas trouvé d'information pertinente dans le corpus pour répondre à cette question. Veuillez poser une question sur les arts et métiers ou consulter votre formateur.
  insufficient       : Je n'ai pas trouvé d'information pertinente dans le corpus pour répondre à cette question. Veuillez reformuler ou consulter votre formateur.
  ambiguous (topics) : Votre question peut correspondre à plusieurs sujets du corpus (glassblowing, glovemaking). Pourriez-vous préciser votre demande ?
  ambiguous (generic): Votre question n'est pas assez précise pour que je trouve une réponse fiable dans le corpus. Pourriez-vous la reformuler avec plus de détails ?

--- en ---
  off_topic          : I could not find any relevant information in the corpus to answer this question. Please ask a question about the crafts and trades, or ask your trainer.
  insufficient       : I could not find any relevant information in the corpus to answer this question. Please rephrase it, or ask your trainer.
  ambiguous (topics) : Your question could match several topics in the corpus (glassblowing, glovemaking). Could you be more specific?
  ambiguous (generic): Your question is not specific enough for me to find a reliable answer in the corpus. Could you rephrase it with more detail?

--- el ---
  off_topic          : Δεν βρήκα σχετικές πληροφορίες στο σώμα κειμένων για να απαντήσω σε αυτή την ερώτηση. Παρακαλώ κάντε μια ερώτηση σχετική με τις τέχνες και τα επαγγέλματα ή απευθυνθείτε στον εκπαιδευτή σας.
  insufficient       : Δεν βρήκα σχετικές πληροφορίες στο σώμα κειμένων για να απαντήσω σε αυτή την ερώτηση. Παρακαλώ αναδιατυπώστε την ή απευθυνθείτε στον εκπαιδευτή σας.
  ambiguous (topics) : Η ερώτησή σας μπορεί να αντιστοιχεί σε περισσότερα θέματα του σώματος κειμένων (glassblowing, glovemaking). Μπορείτε να τη διατυπώσετε πιο συγκεκριμένα;
  ambiguous (generic): Η ερώτησή σας δεν είναι αρκετά συγκεκριμένη ώστε να βρω μια αξιόπιστη απάντηση στο σώμα κειμένων. Μπορείτε να την αναδιατυπώσετε με περισσότερες λεπτομέρειες;

--- unknown language 'de' -> English fallback + a warning naming it ---
   I could not find any relevant information in the corpus to answer this question. Please ask a question about the crafts and trades, or ask your trainer.

--- None / '' -> French, matching the query_language default ---
   Je n'ai pas trouvé d'information pertinente dans le corpus pour répondre à cette question. Veuillez reformuler ou consulter votre formateur.

   assertions OK

--- INSUFFICIENT_CONTEXT_MESSAGE is still a plain str (system prompt interpolates it) ---
   type: str
   accessor: fr/None -> instance override, el -> Greek table entry  OK
   system-prompt interpolation still works: "Je n'ai pas trouvé d'information pertinente dans le corpus ...
```

The `de` line above also emitted:

```
WARNING services.rag_service: localized_message: no 'de' translation for 'off_topic' — falling back to 'en'. Add 'de' to USER_MESSAGES if learners are writing in it.
```

### Task 3 — the warning is gone, and fetch_k is now deliberate

Against a scratch collection holding exactly 16 documents (same size as
production's annotation collection):

```
collection size: 16
   k=5    -> fetch_k=16
   k=10   -> fetch_k=16
   k=15   -> fetch_k=16
   k=100  -> fetch_k=16

   OLD call max_marginal_relevance_search(query, k=15)   chroma warnings:
      'Number of requested results 20 is greater than number of elements in index 16, updating n_results = 16'
   NEW call similarity_search(query, k=15) [fetch_k clamped] chroma warnings:
      (none)
   assertions OK

   mocked vector_store: _collection_count -> None | _mmr_fetch_k(5) -> 25 (5x5, no clamp available)
```

Both calls ran against the same live collection in the same process, with a log
handler attached to
`chromadb.segment.impl.vector.local_persistent_hnsw` — the old form reproduces
the production warning verbatim, the new form emits nothing.

### Task 4 — leak reproduced, then detected

The historical failure shape was reproduced in a scratch collection by minting a
fresh random UUID per document per simulated restart (24 restarts × 16 docs),
then deleting all but the last 16 — 384 allocated labels against 16 live
documents, against production's reported 381/16:

```
before any vector query — segment not loaded, so the check does nothing:
   _hnsw_label_stats() -> None
   warn_if_hnsw_labels_orphaned() -> None

after a normal query — segment loaded:
   _hnsw_label_stats() (allocated, live) -> (384, 16)

   warn_if_hnsw_labels_orphaned('clearing annotation documents') logs:
ERROR services.rag_service: HNSW label leak in collection 'scratch_coll' after clearing annotation documents: 384 labels allocated for only 16 live documents (368 orphaned). A segment in this state can fail to reload with 'Cannot return the results in a contigious 2D array', after which every vector query in the process silently returns nothing. Deleting documents cannot reclaim these labels (see _clear_annotation_documents); the only fix is to rebuild the collection offline with the backend stopped.
   -> (384, 16)

healthy collection (16 allocated / 16 live) must stay quiet:
   _hnsw_label_stats() -> (0, 16)
   warn_if_hnsw_labels_orphaned() -> (0, 16)

mocked / broken vector store must degrade to None, never raise:
    None None

_clear_annotation_documents CANNOT reclaim those labels — demonstrated:
   allocated/live before clear: (384, 16)
   allocated/live after  clear: (384, 0)
   -> live documents dropped, allocated label high-water mark did not move.
```

That last block is the direct demonstration of the docstring's claim: clearing
every addressable annotation document took `live` from 16 to 0 and left
`allocated` at 384. The purge is not a repair.

---

## How to revert

No commits were made; everything is in the working tree.

```bash
cd /opt/craftpilot_backend
git checkout -- pipeline.py services/rag_service.py api/models.py
rm docs/changes/03-serving-path.md
```

`git status` must show `services/course_rag_service.py` still modified after
this — that file belongs to Agent 2 and must not be reverted.

The changes are independent; a single task can be backed out on its own:

- Task 1 only — revert `api/models.py` and `_usable_topic`/the filter loop in
  `_build_ambiguous_clarification`.
- Task 2 only — revert `USER_MESSAGES`/`localized_message`/`detect_query_language`
  and the three refusal call sites.
- Task 3 only — drop `fetch_k=self._mmr_fetch_k(k)` from `similarity_search`.
  The warning returns; nothing else changes.
- Task 4 only — remove the two `warn_if_hnsw_labels_orphaned` call sites. The
  helpers are inert if nothing calls them.

**The backend was not restarted, so none of this is live yet.** All four
changes take effect on the next restart.

---

## Known limits / not done

- **Not restarted, not observed in production.** Everything above is offline
  evidence against scratch collections and fabricated inputs. No end-to-end
  Greek query was put through the running backend.

- **Verified by reading only:**
  - `_build_messages`'s system-prompt refusal swap. The unit check confirms
    `insufficient_context_message("el")` returns the Greek string and that
    `str.replace` of the French sentence is exact, but no LLM call was made, so
    the model's actual behaviour with a Greek refusal sentence in an otherwise
    French system prompt is unverified.
  - That `detect_query_language` returns `"el"` for real Greek input. It calls
    `translation_service.decide_translation`, which is already covered by
    `tests/test_language_detection.py` (all passing), but I did not add a
    Greek-input assertion of my own.

- **`services/course_rag_service.py:583` has the identical `fetch_k` defect** —
  `collection.max_marginal_relevance_search(query, k=k)` with no `fetch_k`, so
  every per-course collection query carries the same accidental default 20 and
  the same warning whenever a course collection holds fewer than 20 chunks. That
  file belongs to Agent 2 and was not touched. **Recommend porting
  `_mmr_fetch_k`'s logic there**, or importing it.

- **`tests/test_pipeline_integration.py` was not updated** (not my area). Two
  points a test owner may want:
  - `test_stream_response_insufficient_relevance_skips_videos_and_generation`
    (line ~1175) still passes, but only because the French path reads
    `INSUFFICIENT_CONTEXT_MESSAGE` directly. It does not cover the localized
    path. Adding `mock_rag_service.insufficient_context_message = lambda lang:
    ...` and a non-French case would.
  - There is no test for `_build_ambiguous_clarification` at all. The offline
    checks in this document are not a substitute for one.

- **`test_hyde_generates_document` still fails**, before and after, for want of
  `.env` credentials. Not addressed.

- **The orphan-label detector is bounded by design.** It reports `None`
  (= "no information") whenever the vector segment is not already loaded, and
  `allocated` lags a pending write batch (see the caveat under Task 4). It
  cannot prove a collection is healthy — only shout when one clearly is not.

- **The production collection was not inspected.** No `chroma.sqlite3` read, no
  second `PersistentClient` against `./chroma_langchain_db`. Whether the live
  collection currently has orphaned labels is therefore still unknown; the
  startup check will answer that on the next restart, in the log.

- **Nothing needs root.** No sudo command is pending for the user.
