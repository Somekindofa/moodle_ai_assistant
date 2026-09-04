# Change records — 2026-09-04

Three agents worked in parallel on a strict file partition, so no two touched the
same file. Each wrote its own record below. **If a bug appears in this area,
start here** — every record lists the exact files and functions touched, the
evidence gathered, and how to revert.

| # | Area | Files owned | Record |
|---|---|---|---|
| 01 | Video elicitation annotation tool | `/opt/video_elicitation_annotation_tool/js/app.js`, `css/styles.css` | [01-annotation-tool.md](01-annotation-tool.md) |
| 02 | Course ingestion / chunking | `services/course_rag_service.py` | [02-ingestion-chunking.md](02-ingestion-chunking.md) |
| 03 | Query serving path | `pipeline.py`, `services/rag_service.py`, `api/models.py` | [03-serving-path.md](03-serving-path.md) |

## Cross-cutting change made outside the partition

`services/course_rag_service.py::similarity_search` had the **same implicit
`fetch_k` defect** that record 03 fixed in `RAGService` — MMR left `fetch_k` to
langchain's default of 20 regardless of `k` or of how many chunks the course
holds. Agent 3 found it but could not cross the ownership line into agent 2's
file, so it was ported afterwards: `fetch_k = min(max(k * 5, k), 50, len(data['ids'])) or 1`,
reusing the `collection.get()` result already fetched one line above so the
count costs nothing. **`RAGService.MMR_FETCH_K_MULTIPLIER` / `MMR_FETCH_K_CAP`
and this expression must be kept in step.**

## What is NOT yet live

Nothing here has been loaded. **All backend changes take effect on the next
`sudo systemctl restart craftpilot-backend`.** The annotation tool serves from
disk, so its changes appear on the next hard refresh.

## Rollout still required (cannot be done from this account)

Record 02's chunker fix **repairs nothing already indexed** — all 11,671 existing
chunks keep their bad breadcrumbs until re-ingested.

- Re-ingest is `/local/craftpilot/reingest_all.php`, which needs
  `moodle/site:config`. `claude_runner` (uid 293) is not a site admin.
- Re-saving a page in Moodle is **not** a workaround: that path goes through
  `observer.php::course_module_updated`, which skips on unchanged
  `content_hash`. A code-only change does not alter the hash.
- `reingest_all.php` sets `set_time_limit(300)` but the corpus needs ~8,970 LLM
  translation calls — hours, not five minutes. It also wipes
  `local_craftpilot_cm_index` at step 1, so a timeout leaves the corpus
  partially indexed. **Raise the timeouts (or add a CLI entry point) and back up
  `chroma_langchain_db/` before running it.**

## Line endings — read before editing these files

`pipeline.py`, `services/rag_service.py`, `services/course_rag_service.py`,
`api/models.py` and `CLAUDE.md` are **CRLF**. An editor that silently converts
them to LF turns a 179-line diff into 3,425 lines and destroys `git blame`. This
happened once already today and had to be undone before committing. Check with
`file <path>` before and after editing.

## Known-unverified items

Each record has its own "Known limits" section; the ones most likely to bite:

- **01** — the "create custom domain" path inside the craft gate was verified by
  reading only (no Moodle JWT locally, so the button was hidden). Not tested
  inside the Moodle iframe or against a real video file.
- **02** — no live LLM call was possible (`.env` is ACL-blocked), so the
  translation fix is proven *structurally* with a stub. That a Greek heading now
  yields a correct French root is an expectation, not a measurement. What is
  guaranteed is that it is now identical across every chunk of the page.
  Spot-check one Greek module after the first re-ingest.
- **03** — whether the *production* Chroma collection currently holds orphaned
  HNSW labels is still unknown; the new startup check answers it in the log on
  the next restart.

---

## Post-restart verification (2026-09-04 17:22–17:27, production)

Backend restarted 17:22:03 with `Environment=PYTHONNOUSERSITE=1` now pinned via
`/etc/systemd/system/craftpilot-backend.service.d/pythonnousersite.conf`.
Clean startup, 16 annotations synced, no errors.

**Record 03, open question resolved — the production index has NO orphaned labels.**
Measured offline from a copy of the vector segment (never the live files). Parsing
`header.bin` with the correct layout (one leading `uint32`, then `size_t` fields):

```
max_elements          = 1000
cur_element_count     = 0      <-- allocated HNSW labels
size_data_per_element = 14476  ( = 132 + 3584*4 + 8, matches dim 3584 )
maxlevel              = -1
enterpoint_node       = -1
```

`maxlevel = -1` and a 0-byte `link_lists.bin` mean the persisted HNSW graph is
**empty**: all 16 documents are still in Chroma's brute-force buffer, because the
collection has no metadata overrides so the default `hnsw:batch_size=100` applies
and nothing has been flushed into the graph yet. So `allocated 0 / live 16 /
orphaned 0` — yesterday's drop-and-rebuild is holding, and `stable_document_id`
upserts are allocating no new labels.

**Nuance worth keeping:** the orphan failure mode is currently *unreachable*
rather than *fixed upstream* — the graph is empty, so it cannot be corrupted. The
risk returns once the collection grows past the batch threshold and elements are
flushed into the graph. That is exactly what record 03's startup check is for.
Note that check logs at DEBUG when healthy, so at INFO level "no orphans" and
"could not run" look identical — read it as informative only when it WARNs.

**A caution for anyone repeating this measurement:** `length.bin` is a
capacity-sized (1000 x int32) buffer whose unused slots hold uninitialised bytes.
Counting its non-zero entries yields a plausible-looking but completely wrong
"982 orphaned". `header.bin` is the only trustworthy source, and a directory path
cannot be passed to `hnswlib.load_index` — Chroma persists across four files,
not hnswlib's single-file format.

### Functional checks through the Moodle UI (as `claude_runner`, course 109)

| Check | Result |
|---|---|
| Craft inference still fires | `Inferred craft 'glassblowing' from course 109 (category 25)` |
| Correct video card (no regression) | `Loic_biseauOblique.mov.mp4`, annotations 16 -> 2, zero glovemaking |
| Record 03 `fetch_k` | The `requested 20 > index 16` warning is **gone** |
| Record 03 refusal language | Greek off-topic question answered **in Greek**: "Δεν βρήκα σχετικές πληροφορίες στο σώμα κειμένων…" |

**One warning remains by design:** `requested 4 > index 2` comes from
`course_rag_service._search_with_embedding`, which passes `n_results=k` to a raw
`collection.query()`. Clamping it would need a count query per course per query —
a real per-request cost to silence a benign warning that Chroma already handles
by clamping. Left alone deliberately.

**Still unverified** (unchanged from the records above): records 01 and 02's
limits, and the whole of record 02's effect, which cannot appear until a
re-ingest is run.
