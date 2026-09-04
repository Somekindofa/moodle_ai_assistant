# Troubleshooting Guide

This file documents known failure modes, their symptoms, diagnosis steps, and fixes.

---

## 1. Moodle admin JavaScript completely broken ("More" button unresponsive, dropdowns dead)

**Symptom:** All JavaScript on Moodle pages stops working. The secondary navigation "More" dropdown doesn't respond to clicks. Any JS-driven UI (modals, AJAX, etc.) is broken sitewide.

**Browser console error:**
```
/lib/requirejs.php/...core/first.js:NNNNN Uncaught SyntaxError: Cannot use import statement outside a module
require.min.js:5 Uncaught Error: No define call for core/first
```

**Root cause:** Moodle's RequireJS AMD loader bundles `local_craftpilot/chat_interface` into `core/first.js`. If `amd/build/chat_interface.min.js` contains raw ES6 `import` statements (i.e. the Babel build didn't run or failed silently), RequireJS crashes on every page load.

The `.min.js` file being identical in size to the source `.js` is the tell: Babel produced no output, or the source was copied over the build.

**Diagnosis:**
```bash
# Check if source and build are identical (they should NOT be)
diff /var/www/html/public/local/craftpilot/amd/src/chat_interface.js \
     /var/www/html/public/local/craftpilot/amd/build/chat_interface.min.js
# If output is empty → files are identical → Babel build is broken

# Check the build file starts with define() not import
head -3 /var/www/html/public/local/craftpilot/amd/build/chat_interface.min.js

# Verify the live bundle served to the browser is clean
curl -sk "https://aimove.minesparis.psl.eu/lib/requirejs.php/$(grep jsrev <<< "$(curl -sk https://aimove.minesparis.psl.eu/)" | grep -o '[0-9]\{10\}' | head -1)/core/first.js" | grep -c "^import"
# Should return 0
```

**Fix:**
```bash
# 1. Re-run the Babel build
cd /var/www/html/public/local/craftpilot
node_modules/.bin/grunt babel:dist

# 2. Verify the build now starts with define()
head -2 amd/build/chat_interface.min.js
# Expected: define(["exports", "core/ajax", ...], function(...) {

# 3. Purge Moodle localcache so the bundle is regenerated
find /var/www/moodledata/localcache -name "*.js" -delete

# 4. Hard-refresh the browser (Ctrl+Shift+R)
```

**After any edit to chat_interface.js, always re-run the Babel build before testing in the browser.**

---

## 2. Moodle cache purge (when admin UI works)

**Via admin UI:** Site administration → Development → Purge all caches

**Via CLI (when UI is broken, ask user to run — no sudo access):**
```
! sudo -u apache php /var/www/html/admin/cli/purge_caches.php
```

**Via filesystem (purge only JS localcache):**
```bash
find /var/www/moodledata/localcache -name "*.js" -delete
```

---

## 3. Large video uploads failing (disk full / temp overflow)

**Symptom:** Video upload fails partway through; error may mention disk space or temp file.

**Root cause:** Python's `tempfile` defaults to `/tmp` which is a dedicated 2 GB filesystem. Large video uploads spooled to `/tmp` overflow it.

**Fix (already applied in backend/main.py):** At startup, temp dir is redirected to `/var/video_uploads/.tmp` (198 GB partition):
```python
tmp_override = Path(os.getenv("TMPDIR_OVERRIDE", "/var/video_uploads/.tmp"))
tmp_override.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(tmp_override)
os.environ["TMPDIR"] = str(tmp_override)
```

**Diagnosis:**
```bash
df -h /tmp /var/video_uploads
# /tmp should show ~2 GB total; /var is 198 GB
```

---

## 4. SSL certificate expired (aimove.minesparis.psl.eu)

**Symptom:** Browser shows `ERR_CERT_DATE_INVALID`. Playwright/automation tools can't reach the site.

**Certificate info:**
- Type: Sectigo wildcard (`*.minesparis.psl.eu`)
- Files: `/etc/httpd/certs/wildcard.crt`, `/etc/httpd/certs/wildcard.key`, `/etc/httpd/certs/sectigo.crt`
- Expired: March 13, 2024 — renewal must go through IT/Mines Paris sysadmin team

**Workaround (to access admin without renewing):** In Chrome/Edge click "Advanced → Proceed"; in Firefox "Accept the Risk and Continue".

**Diagnosis:**
```bash
echo | openssl s_client -connect aimove.minesparis.psl.eu:443 -servername aimove.minesparis.psl.eu 2>/dev/null \
  | openssl x509 -noout -dates
```

**To reach the site from scripts (ignore cert):**
```bash
curl -sk https://aimove.minesparis.psl.eu/...
```

---

## 5. Plugin version mismatch blocking Moodle admin

**Symptom:** Moodle shows "upgrade required" banner or blocks access to admin after plugin file changes.

**Diagnosis:**
```bash
# Compare version.php vs DB
grep 'version' /var/www/html/public/local/videoelicit/version.php
mysql -u moodleuser -p<DB_PASS> -h localhost moodle \
  -e "SELECT plugin, name, value FROM mdl_config_plugins WHERE plugin='local_videoelicit' AND name='version';"
```

**Fix:** Version must match. Either bump `version.php` and run the Moodle upgrade, or revert the version number to match the DB value.

---

## 6. Wrong video card shown (retrieval returns an off-craft clip)

**Symptom:** A question asked inside a course returns a video source from a completely
different craft — e.g. a glovemaking clip answering a glassblowing bevel question.

**Do not reach for the rerank threshold first.** The wrong clips routinely score *higher*
than the right one, so no threshold separates them. Confirmed 2026-09-04 on the query
`Πώς κατασκευάζεται το λοξό φάλτσο στον τροχό;` (course 109):

```
rerank 0.9620  video_annotation 2 montage fourchette index droit.mp4#63_raw   (glovemaking)
rerank 0.9603  video_annotation 1 pouce droit.mp4#64_raw                      (glovemaking)
rerank 0.7747  video_annotation Loic_biseauOblique.mov.mp4#23_raw             (CORRECT)
```

The only thing that works is excluding the wrong craft **before** scoring.

**Two independent causes. Check both — fixing one alone changes nothing.**

### (a) Retrieval was craft-blind unless a domain button was clicked

`build_cohort_filter(..., craft=...)` originally got a craft only from
`DOMAIN_MAP[selected_domain]`. A student asking from inside a course, without touching the
domain selector, got no craft filter at all.

Fixed: `pipeline.stream_response` now infers the craft from the course's category
(`elif course_id:` loop over `DOMAIN_MAP` + `SiloService.get_course_ids_by_category`), passes
it in state as `domain_craft` (`core/types.py`), and both `retrieve_initial` and
`retrieve_final_dual` in `services/rag_service.py` fall back to it when no domain is selected.

Craft only — it deliberately does **not** narrow `enrolled_course_ids`, so course-content
retrieval keeps its cross-course reach.

**Confirm it is active:**
```bash
grep "Inferred craft" /tmp/craftpilot_backend.log | tail -5
# Expected: Inferred craft 'glassblowing' from course 109 (category 25)
```
No line means either a domain button *was* clicked (fine — that path wins), no `course_id`
was sent, or the course's category is not in `DOMAIN_MAP` (`services/rag_service.py:123`).

### (b) The `craft` column itself is wrong in Moodle

The filter is only as good as the labels. On 2026-09-04, 4 of 16 annotations were tagged
`glassblowing` despite plainly glovemaking transcripts ("mon tissu de gants", "monter la
fourchette"). The tell: annotation 57, on the *same video*, was correctly tagged.

**Diagnose — read the craft against the transcript, not against the filename:**
```bash
PW=$(grep -oP "dbpass\s*=\s*'\K[^']+" /var/www/html/config.php)
mysql -t -u moodleuser -p"$PW" moodle -e "
SELECT a.id, LEFT(v.filename,40) AS video, a.craft, LEFT(a.transcription,60) AS transcript
FROM mdl_local_videoelicit_annotations a
JOIN mdl_local_videoelicit_videos v ON a.videoid = v.id
WHERE a.transcription IS NOT NULL AND a.transcription <> ''
ORDER BY a.craft, a.id;"
```

**Fix — correct the rows, then restart:**
```bash
mysql -u moodleuser -p"$PW" moodle -e "
UPDATE mdl_local_videoelicit_annotations SET craft='<correct>' WHERE id IN (...) AND craft='<wrong>';"
sudo systemctl restart craftpilot-backend
```
Guard the UPDATE on the old value so re-running is a no-op, and capture the current values
as revert statements first. The restart is what propagates the change: the startup sync
upserts on `annotation::<source>` (`stable_document_id`), an id independent of `craft`, so
metadata is updated in place with no duplicates.

**Verify in ChromaDB — read a *copy*, never the live file:**
```bash
# reading chroma.sqlite3 while the backend is writing returns a stale/partial view
cp /opt/craftpilot_backend/chroma_langchain_db/chroma.sqlite3 /tmp/chroma_read.sqlite3
```
then join `embeddings` → `embedding_metadata` on `id`, keyed by `e.embedding_id`.

### Root cause of (b) — STILL OPEN

`/opt/video_elicitation_annotation_tool/js/app.js:816`:
```js
state.craft = localStorage.getItem('craft') || 'glassblowing';
```
Every annotator who never opens the craft dropdown silently files their work as
glassblowing. **Expect new mislabelled annotations until this is addressed.** The fix needs
a product decision (block submission with no craft? force an explicit first choice? infer
from the project?), so it was deliberately left alone.

### Verifying the whole path

The backend rejects unauthenticated calls (`X-Internal-Token`, `server.py:70`) and that
token lives in `.env`, which the `claude_runner` account cannot read by design — so `curl`
against `/api/chat` returns `401 Unauthorized`. Drive the Moodle chat UI instead
(`#cp-toggle` → `#cp-input` → `#cp-send`, sources land in `#cp-sources`). See
`docs/PLAYWRIGHT_DEBUGGING.md`.

Per-document rerank scores are logged by `services/reranker_service.py` — without them a
wrong video card is indistinguishable from a wrong ranking:
```bash
grep "  rerank " /tmp/craftpilot_backend.log | tail -20
```

---

## 7. Key file locations

| What | Path |
|---|---|
| CraftPilot AMD source | `/var/www/html/public/local/craftpilot/amd/src/` |
| CraftPilot AMD build | `/var/www/html/public/local/craftpilot/amd/build/` |
| CraftPilot Gruntfile | `/var/www/html/public/local/craftpilot/Gruntfile.js` |
| VideoElicit Moodle plugin | `/var/www/html/public/local/videoelicit/` |
| Moodle localcache | `/var/www/moodledata/localcache/` |
| Moodle dataroot | `/var/www/moodledata/` |
| Moodle config | `/var/www/html/config.php` |
| SSL certs | `/etc/httpd/certs/` |
| CraftPilot backend log | `/tmp/craftpilot_backend.log` |
| Video elicitation annotation tool | `/opt/video_elicitation_annotation_tool/` |
| Craft default (see §6) | `/opt/video_elicitation_annotation_tool/js/app.js:816` |
| Craft → category map | `/opt/craftpilot_backend/services/rag_service.py:123` (`DOMAIN_MAP`) |
| ChromaDB persist dir | `/opt/craftpilot_backend/chroma_langchain_db/` |
| Video upload temp dir | `/var/video_uploads/.tmp` |
| CLI cache purge | `/var/www/html/admin/cli/purge_caches.php` |
