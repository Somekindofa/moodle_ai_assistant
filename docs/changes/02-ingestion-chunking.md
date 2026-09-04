# 02 — Ingestion: heading hierarchy + breadcrumb translation
**Agent 2 · 2026-09-04 · area: services/course_rag_service.py**

## Problems

**Bug A — a missing `<h1>` fabricated a false heading root.** `SemanticChunker`
tracked the heading breadcrumb in a plain `List[str]` and trimmed it *by list
position*: `heading_stack[:] = heading_stack[: level - 1]`. Position and heading
level are only the same thing when a document starts at `<h1>` and never skips a
level. On a page whose first heading is an `<h2>`, `level - 1 == 1` preserves
index 0, so the first `<h2>` becomes a permanent breadcrumb root and every later
`<h2>` is falsely nested beneath it — `Το φυσοκάλαμο > Το πόντιλ`, where the
punty is not part of the blowpipe; they are sibling tools. Because `_emit`
prepends the breadcrumb to the chunk text *before embedding*, the fabricated
parent goes into the vector, not just the citation line. The same trimming
expression appeared three times (HTML walker, PDF chunker, DOCX chunker), so all
three source formats were affected. Authoring an `<h1>` into every page is a
per-page workaround; every pre-existing course still carries the defect.

**Bug B — breadcrumbs were re-translated per chunk, inconsistently and wrongly.**
Ingestion translates non-French modules to French and stores the translated text.
The breadcrumb was already baked into `page_content`, so each chunk's translation
call handed the model a title followed by a body — and the model rewrote the
title from the body it had just read. One Greek `<h1>`,
`Ανόπτηση, ψυχρή κατεργασία και ελαττώματα` (annealing, cold working and
defects), came back as five different French roots across five chunks of one
page — `Découpage` / `Torsade` / `Tronçonnage` / `Affûtage` / `Torsion` — none of
which means annealing. Measured against the live index (read-only query of
`chroma_langchain_db/chroma.sqlite3`): **323 of 4,749** `(module, heading_path)`
groups have chunks that disagree on their own baked-in breadcrumb, and **all 323
carry a `source_language` tag** — i.e. every affected group is one that went
through ingestion translation. The breadcrumb's whole purpose is to be a stable
hierarchy signal; a per-chunk-variable breadcrumb is noise injected into every
vector of the page.

## Files touched

| File | Lines/functions | What changed |
|---|---|---|
| `services/course_rag_service.py` | new module-level `_push_heading`, `_breadcrumb` | Heading stack is now `List[Tuple[int, str]]`, trimmed by *level* (`pop while top.level >= new.level`), never by list position. Handles skipped levels, pages starting at h2/h3, and repeated same-level siblings uniformly. |
| `services/course_rag_service.py` | `SemanticChunker._walk_soup` (HTML) | Uses `_push_heading` / `_breadcrumb` instead of `heading_stack[:level-1]` + `" > ".join(...)`. |
| `services/course_rag_service.py` | `SemanticChunker.chunk_pdf` | Same substitution (the font-size level heuristic itself is unchanged). |
| `services/course_rag_service.py` | `SemanticChunker.chunk_docx` | Same substitution. |
| `services/course_rag_service.py` | new module-level `_build_heading_translation_prompt` | A title-only translation prompt, separate from `translation_service.build_chunk_translation_prompt`. |
| `services/course_rag_service.py` | new `CourseRAGService._split_breadcrumb` | Splits stored chunk text into `(breadcrumb, body)` using `metadata["heading_path"]`. Returns `("", text)` when the text does not start with the breadcrumb, so anything not produced by `SemanticChunker._emit` behaves exactly as before. |
| `services/course_rag_service.py` | new `CourseRAGService._translate_heading_path` | One LLM call per **distinct heading segment**, memoised in a caller-owned cache. |
| `services/course_rag_service.py` | new `CourseRAGService._reattach_breadcrumb` | Glues the translated breadcrumb back onto the translated body. |
| `services/course_rag_service.py` | `CourseRAGService._translate_chunks_if_needed` | Splits the breadcrumb off, translates the **body only**, re-attaches the once-translated breadcrumb. Cache is per module. Body-translation failure still falls back to the whole original chunk (unchanged behaviour). |
| `services/course_rag_service.py` | `CourseRAGService.backfill_translations` | Same split/re-attach, with one cache for the whole run so every chunk in a collection agrees on a given heading. |
| `tests/test_course_heading_hierarchy.py` | **new file** (12 tests) | Offline coverage for both bugs. Note: `tests/` is gitignored in this repo — needs `git add -f`. |

CRLF line endings preserved (verified with `file` before and after; the whole
file is 917 lines, 917 CRLF, 0 lone CR). Diff is `182 insertions(+), 23
deletions(-)`, not a whole-file rewrite.

## Design decision on breadcrumb translation

**Chosen: translate each distinct heading segment once, then reuse it.**
Not "per distinct full path" — per *segment*. A page produces both `A` (for text
directly under A) and `A > B` (for text under B); if the two paths were
translated as whole strings, the same heading `A` could still come out two
different ways within one page, which is the exact defect being fixed. Caching
by segment makes "the same heading always renders the same French string" an
invariant, and it composes: a heading shared between a leaf and an ancestor path
is identical in both. A failed or empty translation caches the *original*
segment, so a transient API error degrades to a stable untranslated breadcrumb
rather than one that flips language halfway down a page.

The second half of the fix matters as much as the caching: the body is now
translated **without** the breadcrumb attached, with a dedicated title-only
prompt used for the heading. The observed corruption was not random LLM
variance — the model was handed `title + body` under a prompt that says
"conserve la structure (y compris un éventuel titre de section en début de
texte)" and dutifully produced a title matching the body's topic. Removing the
body from the heading call removes the mechanism, not just the symptom.

**Rejected: not translating the breadcrumb at all.** It would also be consistent,
and it is true that queries are translated to French before embedding, so a
Greek breadcrumb would mostly stop contributing. But "mostly stop contributing"
is not free: the breadcrumb is *inside* the embedded text, so leaving it in the
source language makes every chunk of a non-French page a bilingual string, and
that out-of-language prefix perturbs the vector of every chunk on the page in
the same direction. It trades a wrong signal for a dead-weight one, and it
throws away design principle #2 of the chunker (hierarchy-aware retrieval)
exactly for the courses that need it most — the non-French ones. Translating
once costs one extra LLM call per distinct heading per module (typically a
handful) against the tens or hundreds of body calls that module already makes,
so the cost argument does not favour skipping it either.

**Not done, deliberately:** `_build_heading_translation_prompt` logically belongs
next to the other prompt builders in `services/translation_service.py`, but that
file is outside this agent's ownership, so the prompt lives in
`course_rag_service.py` for now. Worth moving in a follow-up.

## Evidence it works

### Bug A — old vs new algorithm on the same HTML

Run offline against the real `_push_heading` / `_breadcrumb` (old algorithm
reimplemented verbatim from the pre-fix source for comparison):

```
=== no <h1> — two sibling tools (the reported Greek page)
  OLD: 'Το φυσοκάλαμο'                         NEW: 'Το φυσοκάλαμο'
  OLD: 'Το φυσοκάλαμο > Το πόντιλ'             NEW: 'Το πόντιλ'            <-- FIXED
  OLD: 'Το φυσοκάλαμο > Το πόντιλ > Χρήση'     NEW: 'Το πόντιλ > Χρήση'    <-- FIXED

=== <h1> present — normal nesting
  OLD: 'Εργαλεία'                              NEW: 'Εργαλεία'
  OLD: 'Εργαλεία > Το φυσοκάλαμο'              NEW: 'Εργαλεία > Το φυσοκάλαμο'
  OLD: 'Εργαλεία > Το πόντιλ'                  NEW: 'Εργαλεία > Το πόντιλ'

=== skipped level h1 -> h3
  OLD: 'Outils'                                NEW: 'Outils'
  OLD: 'Outils > Types de cannes'              NEW: 'Outils > Types de cannes'

=== page starting at h3
  OLD: 'Recuit'                                NEW: 'Recuit'
  OLD: 'Recuit > Defauts'                      NEW: 'Defauts'              <-- FIXED

=== deep then shallow: h1 h2 h3 h2 h1
  OLD: 'A'                                     NEW: 'A'
  OLD: 'A > B'                                 NEW: 'A > B'
  OLD: 'A > B > C'                             NEW: 'A > B > C'
  OLD: 'A > D'                                 NEW: 'A > D'
  OLD: 'E'                                     NEW: 'E'
```

Pages with a proper `<h1>` and pages that skip a level are unchanged — this is a
strict repair, not a re-shaping of correct breadcrumbs.

### Bug B — measured in the live index (read-only)

```
(module, heading_path) groups with a substantive heading: 4749
  ... whose chunks disagree on the baked-in breadcrumb:    323
  ... of those, groups tagged as translated at ingest:     323

Example course_54 module 713
  metadata heading_path: Areas of interest or concern and possible issues and challenges
   -> Axes d'intérêt ou de préoccupation et problèmes et défis possibles
   -> Axes d'intérêt ou de préoccupation et problèmes et défis potentiels
   -> Axes d'intérêt ou de préoccupation et problèmes et défis éventuels
   -> Axes d’intérêt ou de préoccupation et problèmes et défis possibles
   -> Axes d’intérêt ou de préoccupation et problèmes et défis potentiels
   -> Domaines d'intérêt ou de préoccupation et enjeux et défis possibles
   -> Domaines d'intérêt ou de préoccupation et problèmes et défis potentiels
   -> Domaines d'intérêt ou de préoccupation et problèmes et défis éventuels
```

One English heading, eight French renderings inside a single module. This is the
defect the fix removes; it is *not* evidence that the fix has repaired the stored
data (see Rollout).

### Test suite

```
cd /opt/craftpilot_backend
PYTHONNOUSERSITE=1 /root/miniconda3/envs/moodle_backend/bin/python -m pytest \
  tests/test_course_heading_hierarchy.py -v
```

```
tests/test_course_heading_hierarchy.py::test_page_without_h1_keeps_h2_siblings_as_siblings PASSED
tests/test_course_heading_hierarchy.py::test_h1_present_nests_h2_under_it PASSED
tests/test_course_heading_hierarchy.py::test_skipped_level_does_not_invent_a_missing_parent PASSED
tests/test_course_heading_hierarchy.py::test_deeper_then_shallower_pops_back_to_the_right_ancestor PASSED
tests/test_course_heading_hierarchy.py::test_page_starting_at_h3_has_no_fabricated_root PASSED
tests/test_course_heading_hierarchy.py::test_breadcrumb_is_prepended_to_the_embedded_text PASSED
tests/test_course_heading_hierarchy.py::test_one_breadcrumb_translation_is_reused_by_every_chunk_of_the_page PASSED
tests/test_course_heading_hierarchy.py::test_body_is_translated_without_the_breadcrumb_glued_on PASSED
tests/test_course_heading_hierarchy.py::test_same_heading_renders_identically_as_leaf_and_as_ancestor PASSED
tests/test_course_heading_hierarchy.py::test_heading_path_metadata_stays_in_the_source_language PASSED
tests/test_course_heading_hierarchy.py::test_body_translation_failure_keeps_the_whole_chunk_original PASSED
tests/test_course_heading_hierarchy.py::test_chunk_without_a_baked_in_breadcrumb_is_translated_as_one_blob PASSED
============================== 12 passed in 4.50s ==============================
```

`test_one_breadcrumb_translation_is_reused_by_every_chunk_of_the_page` uses a
stub LLM that deliberately returns a *different* French title on every call
(`Découpage` / `Torsade` / `Tronçonnage` / `Affûtage` / `Torsion` — the five real
observed outputs). It asserts that all five chunks still come out with the same
root, and that exactly one heading-translation call was made.

No regressions in the pre-existing tests for this area:

```
PYTHONNOUSERSITE=1 /root/miniconda3/envs/moodle_backend/bin/python -m pytest \
  tests/test_course_translation.py tests/test_course_backfill.py \
  tests/test_course_silo.py tests/test_document_ids.py -q
→ 29 passed in 16.14s
```

(The full suite was deliberately not run: importing the pipeline constructs live
services and the run has been SIGKILLed before.)

### Byte-level checks

```
$ PYTHONNOUSERSITE=1 python3 -m py_compile services/course_rag_service.py   # clean
$ file services/course_rag_service.py
services/course_rag_service.py: Python script, Unicode text, UTF-8 text executable, with CRLF line terminators
$ git diff --stat -- services/course_rag_service.py
 services/course_rag_service.py | 205 +++++++++++++++++++++++++++++++-----
 1 file changed, 182 insertions(+), 23 deletions(-)
```

## Rollout required (cannot be done from here)

**The code fix changes nothing about already-indexed content.** All 11,671
existing course chunks keep the breadcrumbs they were embedded with. Only
newly-ingested or re-ingested modules get the corrected behaviour.

1. **Restart the backend** so the new `course_rag_service.py` is loaded.
   Not done here on purpose — another agent and a live user depend on the
   running service.
   ```bash
   # as root, when the service is safe to bounce
   /root/miniconda3/envs/moodle_backend/bin/uvicorn server:app --host 0.0.0.0 --port 8000
   ```

2. **Re-ingest the corpus** from the Moodle admin page
   `https://aimove.minesparis.psl.eu/local/craftpilot/reingest_all.php`.
   This requires `moodle/site:config`. The account available to this agent
   (`claude_runner`, uid 293) is **not** a site admin — admins are uids
   2, 3, 19, 22, 23, 280 — so this step must be performed by one of them.

3. **`content_hash` does NOT block the re-ingest — checked.** The hash skip
   lives only in `plugin/classes/observer.php::course_module_updated`
   (line 81: `if ($existing && $existing->content_hash === $hash) { return; }`),
   which is the *incremental* save path. `reingest_all.php` takes a different
   path: its Step 1 does `$DB->delete_records('local_craftpilot_cm_index')`,
   wiping the hash table outright, and its per-module loop then calls
   `delete_module()` + `ingest_module()` **unconditionally** — it never reads
   `content_hash` at all, only writes a fresh one afterwards. So a chunker
   change does take effect via that page.
   The corollary is that "just re-save each page in Moodle" is *not* a valid
   workaround: that route goes through the observer and will be skipped as
   unchanged. The admin page is the way.

4. **Expect the re-ingest to exceed its own time limit.** `reingest_all.php`
   sets `set_time_limit(300)` (5 minutes). The corpus is 375 modules /
   ~11,671 chunks, of which ~8,970 carry a `source_language` tag, i.e. go
   through an LLM translation call — now one call per chunk body plus a few
   per module for headings. That is hours of API time, not minutes. Options,
   in order of preference:
   - raise `set_time_limit()` in `reingest_all.php` (and PHP-FPM's
     `request_terminate_timeout` / Apache `ProxyTimeout` for that path) before
     running it; **or**
   - run it repeatedly — the page is idempotent per module (it deletes then
     re-ingests), so successive runs make progress, but each run restarts from
     module 1 and re-does everything, so this is only viable if the limit is
     also raised; **or**
   - add a CLI equivalent under `plugin/cli/` that loops the same logic without
     a web time limit. There is currently no such script (`plugin/cli/` holds
     only `migrate_from_mod.php`). This would be the robust answer and is *not*
     implemented here — `plugin/` is outside this agent's file ownership.

5. **Take a Chroma backup first.** There are already snapshots at
   `chroma_backup_20260903-180736/` and `chroma_backup_20260903-180819/`;
   make a fresh one before a full re-ingest.

6. **Verify afterwards** by re-running the two read-only measurements above:
   the "chunks disagree on the baked-in breadcrumb" count should drop toward 0
   for re-ingested courses, and a spot check of a no-`<h1>` page should show
   sibling headings as siblings rather than nested.

## How to revert

The change is confined to one file plus one new test file, with no schema, API
or config change, and no data migration.

```bash
cd /opt/craftpilot_backend
git checkout -- services/course_rag_service.py     # reverts both fixes
rm tests/test_course_heading_hierarchy.py          # optional
# then restart the backend
```

A pre-edit copy is also at
`/tmp/claude-1000/-home-claude-runner/c8258934-98a5-4964-bf53-a5a553424cd1/scratchpad/course_rag_service.py.bak`
(session-scoped, do not rely on it long-term).

Reverting the code does **not** un-do a re-ingest that has already run; restore
the Chroma directory from a backup for that.

## Known limits / not done

- **No live LLM verification.** `.env` is ACL-blocked for `claude-runner`, so no
  Infomaniak call could be made from here. Every translation assertion in this
  report is about *structure and stability* (one call per heading, identical
  result reused everywhere, breadcrumb kept out of the body prompt), proven with
  a stub LLM. **The semantic quality of the new title-only prompt is unverified**
  — that `Ανόπτηση…` now comes back as `Recuit…` rather than `Découpage` is the
  expectation, not a measured fact. A bare title carries less context than a
  title-plus-body, and it is possible a short heading translates *worse* in
  isolation for some inputs; what is guaranteed is that whatever it produces is
  now the same for every chunk of the page. Worth spot-checking on one Greek
  module right after the first re-ingest.
- **No end-to-end retrieval eval.** Whether corrected breadcrumbs measurably
  improve MAP is untested. `eval/09_cross_lingual_eval.py` exists and could be
  re-run post-re-ingest, but per CLAUDE.md it is noisy (`llm_temperature = 0.4`)
  and needs more than one run to trust a delta.
- **PDF heading levels are still heuristic.** `chunk_pdf` now uses the correct
  stack semantics, but the level it feeds in still comes from the font-size
  formula `round(2 - avg_size/heading_threshold)` clamped to 1..3, which is
  coarse (it effectively only ever yields 1 or 2 for typical documents). Bug A
  is fixed there; the level *estimate* was left alone as out of scope.
- **Backfill path improved but untested against live data.** The same
  split/re-attach now applies in `backfill_translations`, covered by the
  existing `tests/test_course_backfill.py` (29 passed) — but that path was not
  exercised against a real collection from here.
- **`_build_heading_translation_prompt` is in the wrong file.** It belongs in
  `services/translation_service.py` with its siblings; left where it is because
  that file is outside this agent's ownership.
- **`tests/` is gitignored** (`.gitignore:8 tests/`, `:14 *test*`) while the
  existing test files are tracked, so the new test needs
  `git add -f tests/test_course_heading_hierarchy.py` to be committed.
- **Nothing was committed, pushed, or restarted**, per the task constraints.
