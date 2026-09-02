# CraftPilot Backend — Technical Reference

## Runtime

- **Entry point**: `server.py` — FastAPI application on `0.0.0.0:8000`
- **Start**: `/root/miniconda3/envs/moodle_backend/bin/uvicorn server:app --host 0.0.0.0 --port 8000`
- **Conda env**: `moodle_backend`
- **Logs**: `/tmp/craftpilot_backend.log`
- **LLM**: `mistral3` via [Infomaniak AI Tools](https://developer.infomaniak.com/docs/api/post/2/ai/%7Bproduct_id%7D/openai/v1/chat/completions) — OpenAI-compatible, `ChatOpenAI` with `openai_api_base = https://api.infomaniak.com/2/ai/{INFOMANIAK_PRODUCT_ID}/openai/v1`. Valid IDs (from `GET /2/ai/{id}/openai/v1/models`): `mistral3`, `llama3`, `qwen3`, `swiss-ai/Apertus-70B-Instruct-2509`. `mixtral` is **not** a valid ID — produces garbled output.
- **Embeddings**: `bge_multilingual_gemma2` via Infomaniak (3584-dim, SOTA FR-MTEB) — `OpenAIEmbeddings` with the same base URL. Replaced `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768-dim, 128-token truncation) in March 2026.
- **Vector store**: ChromaDB, persisted at `./chroma_langchain_db`
- **Credentials**: `INFOMANIAK_API_KEY` and `INFOMANIAK_PRODUCT_ID` in `.env`. Product ID retrieved via `GET https://api.infomaniak.com/1/ai`.

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/chat` | Streaming chat endpoint — body includes `conversation_thread_id`, `message`, `selected_domain?`, `course_id?` |
| POST | `/api/ingest-course-module` | Ingest a Moodle course module into the per-course ChromaDB collection |
| DELETE | `/api/delete-course-module` | Remove all chunks for a module from its course collection |
| DELETE | `/api/delete-course` | Drop an entire per-course ChromaDB collection |
| GET | `/api/course-status/{course_id}` | Return chunk and module counts for a course collection |

---

## Architecture Overview

```
MoodleAIAssistantPipeline
├── RAGService              — video annotation collection + PRF retrieval nodes
├── CourseRAGService        — per-course collections + SemanticChunker
├── AnnotationService       — SQLite annotation database reader
├── DocumentProcessingService
├── LangChainService
└── ConversationGraphService — LangGraph state machine
```

The pipeline is a **LangGraph state machine** over `ConversationState`. The active graph is a four-node **Pseudo-Relevance Feedback (PRF)** pipeline:

```
retrieve_initial → refine_query_prf → retrieve_final_dual → generate
```

---

## RAG Pipeline: Corpus-Grounded Pseudo-Relevance Feedback

### Problem Statement

The CraftPilot system serves apprentices in niche vocational crafts — glassblowing, glove-making, nautical sealing. These domains share a structural property that makes standard retrieval strategies unreliable: a deep **vocabulary gap** separates the novice learner from the expert corpus.

A student learning glassblowing might ask *"pourquoi mon verre tombe quand je souffle"* (why does my glass fall when I blow). The expert corpus — sourced from annotated video transcripts and pedagogical documents — describes the same phenomenon as *"perte d'axialité de la paraison liée à une rotation insuffisante de la canne"* (loss of axial alignment of the parison due to insufficient mandrel rotation). These two formulations share almost no lexical overlap. Any retrieval system that treats the query as a nearest-neighbour search problem in embedding space will retrieve weakly relevant or irrelevant documents when the gap is this wide.

### Prior Approach: HyDE and Its Failure Mode

The system's original retrieval strategy was **Hypothetical Document Embeddings (HyDE)** (Gao et al., 2022). HyDE addresses the vocabulary gap by inverting the retrieval direction: rather than embedding the query and searching for similar documents, it prompts the LLM to generate a *hypothetical expert document* that would answer the query, then searches for real documents similar to that synthetic text.

HyDE is effective when the LLM has sufficient parametric knowledge of the domain to generate a plausible synthetic document in the correct technical register. For generic domains — programming, cooking, common sciences — this assumption holds. For niche vocational crafts, it does not. The LLM's knowledge of glassblowing paraison dynamics, Vendée glove-pattern construction, or marine sealant application is sparse and potentially inaccurate. When the synthesised document uses incorrect or absent technical vocabulary, it embeds into a region of the space that is systematically far from the expert corpus, and retrieval degrades to near-random.

### Active Strategy: Corpus-Grounded PRF

The replacement strategy is a dense adaptation of **Pseudo-Relevance Feedback (PRF)**, a classical IR technique with roots in the Rocchio algorithm (1971) and later probabilistic formulations (Lavrenko & Croft, 2001; Abdul-Jaleel et al., 2004). In traditional PRF, the top-k retrieved documents are assumed to be relevant; their term statistics are used to expand the query before a second retrieval pass.

The CraftPilot adaptation replaces term-frequency expansion with **LLM-mediated reformulation grounded in the corpus vocabulary**. The critical design constraint — and the key distinction from naive LLM query expansion — is that the reformulation prompt explicitly instructs the model to use vocabulary *extracted from the retrieved documents*, not vocabulary from its parametric weights:

```
"Documents récupérés (utilise leur vocabulaire technique, ne l'invente pas) :
[Document 1 — page]
...excerpt from first-pass retrieval...

Instructions :
- Réécris la requête originale en une seule phrase reformulée
- Utilise UNIQUEMENT les termes techniques que tu vois dans les documents ci-dessus
- N'invente aucun terme ; si le corpus ne contient pas la réponse, dis-le
```

This makes the LLM a **vocabulary bridge** rather than a knowledge oracle: it reads what the corpus actually says and rewrites the student's question in corpus-consistent language. The safety guarantee is that expansion terms are always attested in the retrieved documents — they cannot be hallucinated.

### Pipeline Node Specification

**Node 1: `retrieve_initial`** (`services/rag_service.py:714`)

Executes a first-pass MaxMarginal Relevance (MMR) search against both collections using the raw user query. MMR is preferred over pure cosine similarity because it penalises redundancy within the result set, ensuring that the top-k documents used for reformulation cover the corpus broadly rather than clustering around a single sub-topic.

- Video annotation collection: `moodle_assistant_collection` — global, domain-agnostic
- Course content collection: `course_{course_id}` — per-course-isolated (queried only when `course_id` is present in state)
- Results merged and deduplicated by `metadata.source`
- Output: `state["context"]` (up to 10 documents), `state["video_metadata"]`

**Node 2: `refine_query_prf`** (`services/rag_service.py:753`)

Takes the top-3 documents from `state["context"]` as corpus evidence. Constructs a structured prompt that presents the original query and corpus excerpts to the LLM and solicits a single reformulated query sentence using only attested vocabulary. Falls back to the original query if no context documents are available (i.e., the system degrades gracefully to zero-shot generation rather than failing).

- Input: `state["messages"][-1]` (original query), `state["context"][:3]`
- Output: `state["refined_query"]`
- Observed example: `"pourquoi mon verre tombe quand je souffle"` → `"Pourquoi la paraison tombe-t-elle lors du soufflage en raison d'une perte d'axialité liée à une rotation insuffisante ou à une prise trop serrée de la canne ?"`

**Node 3: `retrieve_final_dual`** (`services/rag_service.py:809`)

Repeats the dual-collection MMR search using `state["refined_query"]` instead of the original query. The refined query, now using the corpus's own technical vocabulary, retrieves the final context set for generation.

**Node 4: `generate`**

Standard RAG generation: the LLM receives the merged context and produces a pedagogically framed response in French, with explicit instructions to acknowledge uncertainty when the context is insufficient.

### Architectural Properties

- **No hallucinated expansion**: PRF grounds vocabulary expansion in the corpus, not LLM weights.
- **Graceful degradation**: if the first-pass retrieval is empty (empty collection), `refine_query_prf` passes through the original query unchanged; `retrieve_final_dual` similarly returns empty context; `generate` responds with an honest uncertainty acknowledgement.
- **Dual-source fusion**: both video annotations and course documents are retrieved at each pass. The merged context allows the generation step to synthesise across modalities — a student question about glassblowing technique can receive context drawn simultaneously from annotated expert video and from the instructor's written course pages.
- **Shared embedding model**: `CourseRAGService` reuses the `HuggingFaceEmbeddings` instance from `RAGService` (`pipeline.py:43–46`) to avoid loading the 420 MB MPNET model twice. Both collections use the same vector space, making cross-collection similarity scores directly comparable.

### Prior Art Lineage

The codebase already contained a textbook PRF implementation in `services/rag_service.py` (methods `retrieve`, `enhance_query`, `retrieve_final`), marked `[LEGACY]` and absent from the active graph. These methods were rewritten as `retrieve_initial`, `refine_query_prf`, and `retrieve_final_dual`, with the following substantive changes:

1. `retrieve_initial` adds dual-collection queries (the legacy `retrieve` only queried the annotation collection).
2. `refine_query_prf` tightens the prompt to *forbid* fabricating technical terms (the legacy `enhance_query` allowed free LLM elaboration).
3. `retrieve_final_dual` mirrors `retrieve_initial`'s dual-collection architecture with the refined query.

---

## Cross-Lingual Query Detection (August 2026)

### Feature

`detect_and_translate_query` (`services/rag_service.py:974`) is pipeline
step 0, run before `retrieve_initial`. Non-French queries — detected via
`py3langid`, gated by `langid_confidence_threshold` and `min_langid_chars`
in `config/settings.py` — get one LLM translation call into French before
entering the PRF pipeline, so every downstream node keeps embedding French
text against the French-only corpus. French queries pass through with zero
LLM calls. Every failure path (langid unavailable, low confidence, short
query, translation error) degrades to French pass-through — i.e. today's
pre-fix behavior, never a hard failure.

### Running the cross-lingual eval

```bash
cd /opt/craftpilot_backend
PYTHONNOUSERSITE=1 /root/miniconda3/envs/moodle_backend/bin/python eval/09_cross_lingual_eval.py
```

- `PYTHONNOUSERSITE=1` is required — without it, user-site-packages can
  shadow the conda env's correct package versions.
- Must use the conda env's own interpreter, not plain `python3` — the
  system Python has an incompatible `langchain` and can't import this
  codebase at all.
- Compares Config A (raw), B (PRF only), D (translate-first + PRF) MAP on
  the same 33 EN queries against the FR baseline (`04_evaluate_retrieval.py`
  results on disk).

### Eval debugging pattern: diff per-query, not just aggregate MAP

When two eval configs show a suspicious aggregate delta,
`eval/results/config_*_results.json` stores full per-query detail
(`retrieved_sources`, `initial_context_sources`, `refined_query`) — diff
these directly instead of trusting the summary table. A −0.06 MAP
regression between two configs turned out to trace to only 5/33 queries; 4
were a non-deterministic rank-order tie-break between two near-duplicate
corpus videos (`biseauDroit` vs `biseauOblique`), not a logic bug.
`llm_temperature = 0.4` (`config/settings.py:44`) means both the
translation and PRF-refinement LLM calls are non-deterministic, so this
eval is inherently noisy — rerun before trusting a single MAP delta,
especially when the ground-truth set concentrates several queries on
near-duplicate documents.

---

## Course Content Ingestion Pipeline

### Moodle Event Observer

Ingestion is triggered by Moodle's native event system, not by polling. The observer (`classes/observer.php`) registers callbacks for four events:

| Event | Handler | Trigger |
|-------|---------|---------|
| `\core\event\course_module_created` | `course_module_created` | Teacher adds a Page, Label, or Resource to a course |
| `\core\event\course_module_updated` | `course_module_updated` | Teacher saves changes to an existing module |
| `\core\event\course_module_deleted` | `course_module_deleted` | Teacher removes a module |
| `\core\event\course_deleted` | `course_deleted` | Admin deletes an entire course |

All observers use `'internal' => false`, which defers execution until *after* the enclosing Moodle database transaction commits. This is the correct trigger point for ingestion: it guarantees that the content being extracted from the database is the committed, stable version.

**Content hash deduplication.** The `craftpilot_cm_index` table stores the MD5 hash of each module's extracted content alongside its timestamp. On `course_module_updated` events, the observer recomputes the hash and skips the backend call if the content has not changed — for example, when a teacher saves a page without modifying its text (a common Moodle action that triggers the `updated` event regardless).

### Semantic Chunking (`services/course_rag_service.py`)

#### Design Principles

Fixed-size chunking (e.g., RecursiveCharacterTextSplitter with a uniform token window) is inadequate for structured pedagogical content. Course pages and uploaded documents are typically organised around headings that delineate conceptually distinct sub-topics. Splitting at arbitrary character boundaries severs the relationship between a heading and its explanatory paragraphs, and discards the hierarchical context that distinguishes, for instance, "Tools > Blowpipe" from "Tools > Gaffer's block". The semantic chunker respects document structure by:

1. Using heading boundaries as hard split points.
2. Prepending the full heading breadcrumb to each chunk's text before embedding, so that vector similarity searches are hierarchy-aware.
3. Using target token budget (~400 tokens, ~1600 characters) within each section as a soft split point for long paragraphs.
4. Enforcing a minimum chunk size (50 tokens) to prevent near-empty chunks from polluting the collection.

#### HTML Chunking (BeautifulSoup DOM walker)

Source: Moodle Page and Label modules, which store content as Moodle-formatted HTML.

Algorithm (`SemanticChunker._walk_soup`):

```
State: heading_stack: List[str], buffer: List[str], chunks: List[Document]

For each element in soup.descendants:
    If element.name in {h1, h2, h3, h4, h5, h6}:
        flush(force=True)          ← close current section
        level = int(tag[1])
        heading_stack[:] = heading_stack[:level-1]   ← trim to parent level
        heading_stack.append(element.get_text())     ← push new heading
    Elif element.name in {p, li, td, th, dd, dt, blockquote}:
        buffer.append(element.get_text())
        if _approx_tokens(" ".join(buffer)) >= 400:
            flush(force=True)

flush(force=True)   ← emit remaining buffer

flush():
    text = " ".join(buffer).strip()
    if _approx_tokens(text) < MIN_TOKENS: return   ← too small, keep accumulating
    heading_path = " > ".join(heading_stack)
    full_text = f"{heading_path}\n\n{text}"        ← prepend breadcrumb
    emit Document(page_content=full_text, metadata={..., "heading_path": heading_path})
```

The breadcrumb prepend (e.g., `"Outils > La canne > Types de cannes\n\n..."`) encodes the document hierarchy directly into the embedded text. This means that a query for "types of blowpipes" retrieves chunks under the "La canne > Types de cannes" heading with higher similarity than identically-worded text appearing under an unrelated heading.

#### PDF Chunking (PyMuPDF font-size heuristic)

Source: Moodle Resource modules containing PDF files.

PDFs lack semantic markup; heading detection relies on a typographic heuristic derived from the document's own font-size distribution:

```
all_sizes = [span.size for all spans in all pages]
body_size = median(all_sizes)
heading_threshold = body_size * 1.15   ← 15% larger than body = heading

For each page → block → line:
    avg_size = mean(span sizes in line)
    is_bold  = any(span.flags & 0b10000)
    if avg_size >= heading_threshold OR (is_bold AND len(line) < 80):
        flush(force=True)
        approximate level from size ratio
        update heading_stack
    else:
        buffer.append(line_text)
```

The 1.15× threshold was determined empirically to correctly classify section titles in typical instructional PDF documents (lecture slides, technical guides) while ignoring inline emphasis. The bold+short condition catches heading-like spans rendered at body size with bold weight (common in some authoring tools).

#### DOCX Chunking (python-docx paragraph styles)

Source: Moodle Resource modules containing DOCX files.

DOCX documents carry explicit semantic markup via paragraph styles. The chunker maps `Heading N` paragraph styles directly to heading levels without any heuristic:

```
For each para in doc.paragraphs:
    if para.style.name.startswith("Heading"):
        level = int(re.search(r"(\d+)", style_name).group(1))
        flush(force=True)
        heading_stack[:] = heading_stack[:level-1]
        heading_stack.append(para.text)
    else:
        buffer.append(para.text)

For each table in doc.tables:
    For each row:
        buffer.append("Row: " + " | ".join(cell.text for cell in row.cells))
```

Tables are linearised to pipe-delimited rows. This preserves tabular content (e.g., comparison tables of tool properties, assessment rubrics) in a form that the embedding model can process without requiring vision capabilities.

### Per-Course Collection Isolation

Each Moodle course's content is indexed into a dedicated ChromaDB collection named `course_{course_id}`. All retrievals from course content use this collection exclusively, enforcing a hard isolation boundary at the vector-store level.

Isolation rationale: in a multi-course Moodle deployment, an apprentice working in a glassblowing course should not retrieve chunks from an unrelated carpentry course, even if a query happens to be lexically similar across domains. Per-course collections make cross-contamination structurally impossible without adding runtime filtering overhead.

The `CourseRAGService._collections` dictionary caches open collection handles to avoid repeated Chroma client initialisation. Collections are opened lazily on first access and remain cached for the lifetime of the process.

---

## Moodle Frontend Plugin — Critical Identity Note

The running Moodle plugin is **`local_craftpilot`** (a "local" plugin), NOT `mod_craftpilot` (an "activity module" plugin). This caused an entire debugging session to be wasted applying fixes to the wrong file.

**How to confirm which plugin is running**: Open the browser Network tab → find any AJAX request → check the `methodname` field. The correct plugin uses `local_craftpilot_*` methods (e.g. `local_craftpilot_manage_conversations`). The wrong file used `mod_craftpilot_*`.

| | Correct (running) | Wrong (dead) |
|---|---|---|
| Plugin type | `local` | `mod` |
| Path | `/var/www/html/public/local/craftpilot/` | `/var/www/html/public/mod/craftpilot/` |
| Git repo | this repo, `plugin/` (formerly `moodle-local-craftpilot`, now archived) | `moodle-plugin-ai` — **archived, read-only** |
| JS source | `amd/src/chat_interface.js` (1296 lines) | `amd/src/chat_interface.js` (1409 lines) |
| Build command | `cd /var/www/html/public/local/craftpilot && npx grunt babel` | (irrelevant) |
| Proxy URL | `/local/craftpilot/chat_proxy.php` | `/mod/craftpilot/chat_proxy.php` |
| Status | serving all traffic | 0 course instances, empty tables, no traffic since 2026-03-08 |

**Always edit the `local/craftpilot` file.** After any JS edit:
```bash
cd /var/www/html/public/local/craftpilot && npx grunt babel
php /var/www/html/admin/cli/purge_caches.php
```

### Repo history: split (2026-08-03), then re-merged (2026-08-06)

The Moodle plugin now lives **inside this repo**, at `plugin/`. It was a
separate private repo (`moodle-local-craftpilot`) for three days; that repo
had only 3 commits and was never independently released, so the split just
added a two-PR-per-feature tax with no offsetting benefit. It's now archived
on GitHub (not deleted) — its history was merged into `plugin/` here via
`git merge -s ours --no-commit --allow-unrelated-histories` + `git read-tree
--prefix=plugin/`.

**To see the plugin's pre-merge commits**, plain `git log plugin/` won't
show them — this wasn't a real `git subtree` (not installed, no sudo to add
it), so git's pathspec history-simplification doesn't trace the directory
rename across the merge. The original commits are still fully preserved
and reachable, just via a different command:
```bash
git log plugin-merge-2026-08-06 --oneline   # tag on the pre-merge tip
```

**The live path is a synced copy, not a git checkout.**
`/var/www/html/public/local/craftpilot` is **not** version-controlled
itself — same pattern as `local_videoelicit` below. Edit `plugin/` in this
repo, then sync and commit:

```bash
rsync -a --delete --exclude=node_modules --exclude=.claude \
  plugin/ /var/www/html/public/local/craftpilot/
php /var/www/html/admin/cli/purge_caches.php
git add plugin/ && git commit -m "..."
```

Two consequences worth knowing:

- **`plugin/amd/build/*.min.js` is committed on purpose.** Moodle serves it
  directly with no deploy-time build step, and
  `plugin/amd/build/dompurify.min.js` is vendored from `node_modules` with
  no `amd/src` counterpart — `grunt babel` will not regenerate it. Losing it
  breaks XSS sanitization.
- **The gitleaks hook now applies repo-wide.** The old plugin-only
  `.githooks/pre-commit` merged in under `plugin/.githooks/` — this repo's
  root hook (`core.hooksPath`) is what actually runs; the nested one is
  inert, kept only because it's part of the preserved history.

⚠️ **A second, older `local_craftpilot/` copy (no `plugin/` prefix) was
removed in the same session** (was added by PR #19 on 2026-07-30, dead
weight even before the 2026-08-03 split — see git history if you need it).
Don't confuse it with the current `plugin/` folder if you're reading old
commits.

The old preprod/staging workflow (feature branch → preprod → PR review →
promote-to-prod script) was deliberately retired in favor of editing the
live server's git checkouts directly and restarting the affected service —
see `/var/www/html/public/DEV_HANDOFF.md`. The old deploy/promote scripts
still exist at `/home/claude-runner/preprod-migration/deploy/` and the
preprod VM itself hasn't been torn down, but neither is part of the current
process; nothing runs them anymore.

---

## Historical incident notes

- Conversation isolation bugs (Mar 2026) — see `docs/incidents/conversation-isolation-2026-03.md`
- Video streaming perf bugs (Apr 2026) — see `docs/incidents/video-streaming-perf-2026-04.md`
- Security fixes summary — see `docs/incidents/security-fixes.md`
- Relevance classifier truncation bug (Sep 2026) — well-matched, correctly-translated course content misjudged as "insufficient" because `assess_relevance`'s 300-char document preview was 5x shorter than an actual chunk — see `docs/incidents/assess-relevance-truncation-2026-09.md`

---

## Dev tooling — browser-driven debugging against the live Moodle instance

To test something end-to-end for real (a teacher saving a page, the
observer firing, translation, ingestion, CraftPilot chat retrieval) rather
than via a script or unit test, drive a real browser at
`https://aimove.minesparis.psl.eu` with the Playwright MCP tool. This hits
several real gotchas on a fresh session — an expired local TLS cert that's
fine in production but breaks local browser navigation, a stale browser
process lock after reconnecting, the auto-mode permission classifier
silently blocking clicks even after verbal user approval, and Moodle
rendering some "buttons" as `<input type=submit>` that role/CSS selectors
miss. See `docs/PLAYWRIGHT_DEBUGGING.md` for the full writeup and a
quick-start checklist — read it before burning time rediscovering any of
this.

---

## Moodle Theme — Primary Navigation Invisible Bug

### Symptom
Nav items in the primary navigation bar are **clickable but have no visible text**. The items exist in the DOM and respond to clicks, but appear transparent.

### Root Cause
`core/moremenu` (the Moodle component that renders primary nav items) is compiled with `opacity: 0` by default and relies on JavaScript adding the `.observed` class to become visible (`opacity: 1`). If that JS fails silently — or the element is initialised in an unusual layout context — the nav stays invisible.

The compiled rule in the theme CSS cache:
```css
.moremenu { opacity: 0; height: 60px }
.moremenu.observed { opacity: 1 }
```

### Permanent Fix
Add this override to the theme's **custom SCSS** field (Moodle admin → Site administration → Appearance → Themes → Almondb → Raw SCSS):
```css
.primary-navigation .moremenu { opacity: 1 !important; }
```
Or apply it via the database:
```sql
UPDATE mdl_config_plugins
SET value = CONCAT(value, '\n\n.primary-navigation .moremenu { opacity: 1 !important; }\n')
WHERE plugin='theme_almondb' AND name='scss';
```
Then purge caches: `php /var/www/html/admin/cli/purge_caches.php`

### Key Files
- **Template** (frontpage nav): `/var/www/html/public/theme/almondb/templates/frontpage/header3.mustache`
  - Must use `{{> core/moremenu}}` inside `{{#primarymoremenu}}<div class="primary-navigation">` to render real Moodle nav items (Site Administration, Video Elicitation Tool, etc.)
  - A backup of the original (pre-Feb-2026) version is at `header3.mustache.original` — **do not restore it**: it uses `{{{frontpagenavlink}}}` which renders the theme demo links, not the real Moodle navigation
- **Theme CSS cache**: `/var/www/moodledata/localcache/theme/<themerev>/almondb/css/all_<cssrev>.css`
- **Custom SCSS in DB**: `mdl_config_plugins` where `plugin='theme_almondb'` and `name='scss'`

### What NOT to Do
- Do not restore `header3.mustache.original` — it replaces the real Moodle nav with demo `frontpagenavlink` items
- Do not remove `{{> core/moremenu}}` from the template — that is the correct nav renderer
- After any template edit, always run `php /var/www/html/admin/cli/purge_caches.php`

---

## Dependencies Added

```
pymupdf         ← PDF text extraction via block-level dict API (fitz.open)
python-docx     ← DOCX paragraph and table iteration
beautifulsoup4  ← HTML DOM parsing for Page/Label content
```

Install: `/root/miniconda3/envs/moodle_backend/bin/pip install pymupdf python-docx beautifulsoup4`

---

## Security Architecture (March 2026)

### Internal API Token (Moodle ↔ Backend Authentication)

All backend endpoints except `/api/health` and `/api/status` require a shared secret header:

```
X-Internal-Token: <token>
```

The token lives in two places that must stay in sync:
- **Backend**: `INTERNAL_API_TOKEN` in `/opt/craftpilot_backend/.env`
- **Moodle**: `local_craftpilot / internal_api_token` in `mdl_config_plugins` (set via Site Administration → Plugins → Local plugins → CraftPilot, or with `set_config('internal_api_token', '...', 'local_craftpilot')`)

The middleware is in `server.py` (`require_internal_token`). PHP callers that send the header:
- `classes/backend_client.php` — reads via `get_config('local_craftpilot', 'internal_api_token')`
- `chat_proxy.php` — same

If you rotate the token, update both places and restart the backend (`uvicorn`).

### CSRF Protection on `chat_proxy.php`

`chat_proxy.php` validates `sesskey` from the JSON request body using `confirm_sesskey()`. The JS sends `sesskey: M.cfg.sesskey` inside the POST payload. `require_sesskey()` cannot be used directly because PHP's `$_POST` is empty for raw JSON bodies.

### CORS

`server.py` allows only `http://127.0.0.1` and `http://localhost` as origins, with `allow_credentials=False`. The backend is intentionally internal-only and never reachable from the public internet.

### Web-exposed dotfiles — fixed 2026-08-03

Until August 2026, `https://aimove.minesparis.psl.eu/mod/craftpilot/.git/config`
and `/.claude/settings.local.json` returned **HTTP 200 to the public internet**.
Apache's stock `<Files ".ht*">` rule matches only `.htaccess`/`.htpasswd` — it
does not cover `.git`, `.env`, or `.claude`.

The fix is a server-scope block in `/etc/httpd/conf/httpd.conf`:

```apache
<DirectoryMatch "/\.(git|svn|hg|bzr|claude|github)(/|$)">
    Require all denied
</DirectoryMatch>
<FilesMatch "^\.(git.*|env.*|claude.*|npmrc|pypirc)$">
    Require all denied
</FilesMatch>
```

Two invariants:

1. **Server scope, not inside a `<VirtualHost>`.** Three vhosts serve
   `/var/www/html/public` across ports 80 and 443; a vhost-scoped rule left
   port 80 exposed.
2. **`<DirectoryMatch>`, not just `<FilesMatch>`.** `.git/index` plus the loose
   objects under `.git/objects/` are enough to reconstruct a full source tree,
   so blocking `.git/config` alone accomplishes nothing.

This matters more now that `local/craftpilot` is itself a git checkout. Verify
after any Apache change — both must return 403:

```bash
for s in http https; do curl -s -o /dev/null -w "$s -> %{http_code}\n" -k \
  "$s://aimove.minesparis.psl.eu/local/craftpilot/.git/config"; done
```

The `X-Internal-Token` hardcoded in `moodle-ssl.conf` was checked against all
git history and every plugin file on 2026-08-03 — not present, no rotation
needed.

### Input Validation

`api/models.py` enforces:
- `ChatRequest.message`: 1–4000 characters
- `conversation_thread_id`: ≤ 255 characters
- `course_id`: ≤ 20 characters

### Video Path Allowlist

`api/routes.py` (`stream_video`) resolves the file path with `Path.resolve()` and then checks it starts with one of the permitted directories:
- `/opt/video_elicitation_annotation_tool`
- `/var/www/html`
- `/tmp`

Requests to paths outside these directories return HTTP 403.

### Secrets in `.env`

`.env` is gitignored. It contains:

| Key | Purpose |
|-----|---------|
| `INFOMANIAK_API_KEY` | LLM + embedding API |
| `INFOMANIAK_PRODUCT_ID` | Infomaniak product ID |
| `LANGSMITH_API_KEY` | LangSmith tracing (renamed from `LANGCHAIN_API_KEY` in May 2026; the legacy name is still accepted by the SDK as a fallback but is deprecated) |
| `LANGSMITH_TRACING` / `LANGSMITH_ENDPOINT` / `LANGSMITH_PROJECT` | Tracing flag, region endpoint, and project name (not secrets) |
| `MOODLE_DB_PASSWORD` | MySQL `moodleuser` password (used by `routes.py` `/annotations-dashboard` and `export_to_owncloud.py`) |
| `INTERNAL_API_TOKEN` | Shared secret for Moodle → backend auth |

Never hardcode these values in source files. The startup log prints only key names, not values.

### XSS Protection (Frontend)

All LLM output rendered via `bubble.innerHTML` is first passed through:
1. `stripThinkTags()` — removes `<think>…</think>` blocks
2. `renderMarkdown()` — runs `marked.parse()` then `DOMPurify.sanitize()`

DOMPurify is bundled as `amd/build/dompurify.min.js` (DOMPurify 3.3.3, AMD-compatible). Source: `node_modules/dompurify/dist/purify.min.js`. After any DOMPurify update: copy `purify.min.js` to `amd/build/dompurify.min.js`, rebuild with `npx grunt babel`, then purge caches.

### Operational Security — Handover Notes (May 2026)

The host is configured so that AI coding assistants (Claude Code, Copilot, etc.) cannot read secrets even if the model misbehaves. If you inherit this box, here is how it works and what not to break.

**Run AI tools as `claude-runner`, never as root.**

```bash
sudo -u claude-runner -i
claude     # or whatever CLI you use
```

The `claude-runner` unix user (uid 1000) has its password locked. Its home is `/home/claude-runner`, including a `.claude/` config carried over from the original setup.

**What the cage looks like:**

- `.env` files are mode `600`:
  - `/opt/craftpilot_backend/.env` — owned `root:root` (this service runs as root)
  - `/opt/video_elicitation_annotation_tool/.env` — owned `apache:apache`
- POSIX ACLs grant `claude-runner` `rwX` on the two project trees so it can edit source, AND explicitly `---` (no access) on each `.env` file. The kernel refuses every read attempt — `cat`, `python open()`, `dd`, `/proc/<pid>/environ` — they all return `Permission denied`.
- `/root` is `550` and `/root/.ssh` is `700`, so `claude-runner` cannot traverse into them.
- Default ACLs on the project trees mean new files inherit the grant. **If you ever add a new secret-bearing file** (a credential JSON, a `.pem`, etc.) inside these trees, immediately revoke the AI's access:
  ```bash
  sudo setfacl -m u:claude-runner:--- <newfile>
  ```

**Verifying the cage still holds**, e.g. after restoring from a backup that lost ACLs:

```bash
sudo -u claude-runner cat /opt/craftpilot_backend/.env   # must say "Permission denied"
sudo -u claude-runner head -1 /opt/craftpilot_backend/app.py  # must succeed
```

If the first command succeeds, ACLs are gone and the cage is open. Re-apply with:

```bash
sudo setfacl -R -m u:claude-runner:rwX /opt/craftpilot_backend
sudo setfacl -R -d -m u:claude-runner:rwX /opt/craftpilot_backend
sudo setfacl -m u:claude-runner:--- /opt/craftpilot_backend/.env
```

(Same pattern for `/opt/video_elicitation_annotation_tool`.)

**Pre-commit hook (gitleaks).**

Installed at `.git/hooks/pre-commit` in both `craftpilot_backend` and `video_elicitation_annotation_tool`. It runs `gitleaks protect --staged` and rejects commits that introduce key-shaped strings.

Caveats:
- Per-clone, not pushed with the repo. If you re-clone or another developer joins, reinstall it. Or migrate to the `pre-commit` framework (`.pre-commit-config.yaml`) so it lives in the repo.
- Bypassable with `git commit --no-verify` — by design, but use sparingly.
- Tunable via `.gitleaksignore` or inline `# gitleaks:allow` comments for false positives (UUIDs, hash fixtures).
- `gitleaks` binary lives at `/usr/local/bin/gitleaks` (v8.30.1 as of install). Update with the standard release tarball from `gitleaks/gitleaks` on GitHub.

**One-time history scrub (May 2026).**

The `INFOMANIAK_API_KEY` was previously hardcoded in three eval scripts and committed to `main` and `feat-astream-events-refactor`. After rotation, `git-filter-repo` rewrote all branches to replace the leaked value with `REDACTED_OLD_INFOMANIAK_KEY` and the affected branches were force-pushed. **Old PR refs (`refs/pull/N/head`) on GitHub still contain the leaked value** — GitHub does not rewrite these on force-push. The leaked key is dead (rotated) so this is purely cosmetic; if you want to scrub PR refs too, file a GitHub Support ticket.

**No automated audits run on this box.** No cron, no GitHub Action, nothing that consumes API tokens periodically. The protections above are static (filesystem perms + commit-time hook). If you want continuous monitoring, you'll have to add it yourself.

---

### Removed: Fireworks.ai (March 2026)

The `local_craftpilot_keys` DB table (per-user Fireworks API keys) was dropped in plugin version `2026031200`. Associated code removed:
- `credential_service.php` — replaced with `get_internal_api_token()`
- `classes/external/get_user_credentials.php` — now a no-op session check
- Fireworks heading in `settings.php` and lang strings

The `local_craftpilot_get_user_credentials` AJAX method still exists in `services.php` (used by JS as a pre-stream session check) but no longer returns any credentials.

---

## Status Hints Streaming — Architecture (June 2026)

### Problem: PHP-FPM buffering → 504 timeout

The original `chat_proxy.php` acted as a cURL proxy, streaming the entire backend response through PHP. This hit two compounding problems:

1. **`ProxyIOBufferSize 1048576`** (1 MB default) in Apache's `mod_proxy_fcgi` caused the FastCGI (PHP-FPM) response to buffer in Apache before it was forwarded to the browser. Status events never reached the user in real-time.
2. **`ProxyTimeout`** inside a `<VirtualHost>` does NOT apply to `SetHandler proxy:fcgi://...` connections — only the global default (300 s) applies. The pipeline takes 3+ minutes, so 504 timeouts were inevitable regardless of `ProxyTimeout 7200` in the VirtualHost.

### Fix: 307 redirect architecture

`chat_proxy.php` now validates the Moodle session and `sesskey`, then issues a **307 redirect** to `/craftpilot-api/chat`. The browser re-POSTs with the same body directly to Apache's HTTP `ProxyPass`, bypassing PHP-FPM entirely.

```
Browser → POST /local/craftpilot/chat_proxy.php
                │  (validates session + sesskey)
                │  307 →
Browser → POST /craftpilot-api/chat   (Apache ProxyPass → uvicorn:8000)
```

Key details:
- Apache's `<Location /craftpilot-api/>` injects `X-Internal-Token` for all requests to that path, including browser-followed redirects.
- `ProxyPass /craftpilot-api/ ... flushpackets=on` forwards each JSON-line chunk to the browser immediately.
- `Timeout 7200` and `ProxyTimeout 7200` inside the VirtualHost DO apply to HTTP ProxyPass — the 3-minute pipeline is safe.
- The request body (`sesskey`, `message`, etc.) is preserved verbatim through the 307. The backend ignores `sesskey` (only Apache's `X-Internal-Token` matters for backend auth).

### Problem: Apertus 70B outputs empty response

`swiss-ai/Apertus-70B-Instruct-2509` is a **reasoning model** that generates internal `<think>...</think>` blocks before its visible answer. The system prompt previously contained:

```
"- Ne produisez JAMAIS de balises <think> ni de raisonnement interne visible."
```

Telling a reasoning model to suppress `<think>` blocks causes it to produce nothing visible after the think block — the entire generation is internal, and `stream_generate`'s filter strips it all, yielding 0 tokens. The browser receives `[DONE]` with `fullResponse = ""`.

**Diagnosis signal**: In the backend log, when this happens, the `httpx` LLM API `"200 OK"` and the uvicorn request completion appear at the **same second** — the generator exhausted immediately after the LLM responded because no tokens passed the filter.

### Fix: remove the `<think>` suppression rule

Removed from `system_prompt` in `services/rag_service.py`. The backend's `stream_generate` already filters think blocks server-side — let the model think freely and output a visible answer, then strip the thinking before forwarding tokens.

Added to `stream_generate`:
- **Token counter**: logs `"N LLM chunks → M tokens yielded (think block filtered)"` after every generation.
- **Safety fallback**: if N > 0 but M = 0 (all tokens filtered), logs an error and yields a French fallback message rather than silently returning empty.

---

## Security Issues — Fixed

See `docs/incidents/security-fixes.md` for the full table (12 items, CRITICAL → LOW, with fix locations).
