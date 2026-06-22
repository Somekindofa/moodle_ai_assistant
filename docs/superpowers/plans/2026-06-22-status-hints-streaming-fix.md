# Status Hints Streaming Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RAG pipeline status hints ("Recherche…", "Reformulation…", etc.) visible in the chat widget by ensuring each hint is flushed to the browser before the next pipeline step blocks.

**Architecture:** Two-layer fix. Server: wrap the three synchronous pipeline calls in `asyncio.to_thread()` so the asyncio event loop — and therefore uvicorn's HTTP flushing — is never blocked between status yields. Client: convert the event-processing loop from `forEach` to `for...of` with a `requestAnimationFrame` pause after each `status` event so the browser repaints each hint even if two arrive in the same TCP segment.

**Tech Stack:** Python 3.10+ asyncio, FastAPI StreamingResponse, Moodle AMD (ES6 module compiled via Grunt/Babel), browser Fetch ReadableStream API.

## Global Constraints

- Branch: `feat/status-hints-streaming` (already created in `/opt/craftpilot_backend`)
- No new dependencies — `asyncio.to_thread` is stdlib (Python 3.9+)
- No inline `style=` attributes in JS — all visual styling goes in `styles.css` (already correct)
- No `sudo` — service restart must be done by the user via `! sudo systemctl restart craftpilot-backend`
- After any edit to `amd/src/chat_interface.js`, the full three-step deploy sequence is required (see Task 2)
- Status hint strings remain French-only

---

## File Map

| File | Role | Change type |
|---|---|---|
| `/opt/craftpilot_backend/pipeline.py` | Async generator that streams pipeline events | Modify: replace 3 blocking calls with `asyncio.to_thread`; remove `asyncio.sleep` calls |
| `/var/www/html/public/local/craftpilot/amd/src/chat_interface.js` | AMD source — compiled to `amd/build/chat_interface.min.js` | Modify: `forEach`→`for...of`+RAF; remove debug overlay; remove inline style from hint span |

---

## Task 1: Server — unblock the event loop between status yields

**Files:**
- Modify: `/opt/craftpilot_backend/pipeline.py` lines 365–395

**Interfaces:**
- `asyncio.to_thread(callable, *args)` — stdlib coroutine; awaiting it runs `callable(*args)` in a thread-pool thread and returns its result, without blocking the event loop
- `self.rag_service.retrieve_initial(state: dict) -> dict` — synchronous, ~0.5 s
- `self.rag_service.refine_query_prf(state: dict) -> dict` — synchronous, LLM call ~2–5 s
- `self.rag_service.retrieve_final_dual(state: dict) -> dict` — synchronous, ~0.5 s

- [ ] **Step 1: Replace the three blocking calls with `asyncio.to_thread`**

In `/opt/craftpilot_backend/pipeline.py`, find the PRF steps block (currently lines 365–395) and replace it with the following. The diff is surgical — only the three `retrieve_*`/`refine_*` calls and their preceding `asyncio.sleep` lines change:

```python
            # --- PRF step 1: initial retrieval ---
            yield json.dumps({"event": "status", "data": "Recherche dans la base de connaissances…"}) + "\n"
            result = await asyncio.to_thread(self.rag_service.retrieve_initial, state)
            state.update(result)

            # --- PRF step 2: corpus-grounded query refinement ---
            yield json.dumps({"event": "status", "data": "Reformulation de la question…"}) + "\n"
            result = await asyncio.to_thread(self.rag_service.refine_query_prf, state)
            state.update(result)

            # --- PRF step 3: final retrieval with refined query ---
            yield json.dumps({"event": "status", "data": "Récupération des sources pertinentes…"}) + "\n"
            result = await asyncio.to_thread(self.rag_service.retrieve_final_dual, state)
            state.update(result)

            # --- PRF step 4: cross-encoder reranking and relevance filtering ---
            if not disable_rerank:
                yield json.dumps({"event": "status", "data": "Classement des résultats…"}) + "\n"
                # Cap candidates before reranking — bge-reranker-v2-m3 takes
                # ~5 s per pair on a 2-core CPU, so keep the list tight.
                ctx = state.get("context", [])
                if len(ctx) > self.MAX_RERANK_CANDIDATES:
                    state["context"] = ctx[: self.MAX_RERANK_CANDIDATES]
                # Run synchronous cross-encoder inference in a thread so the
                # event loop remains responsive during the ~20-30 s prediction.
                result = await asyncio.to_thread(self.rag_service.rerank, state)
                state.update(result)
```

Key changes from current code:
- Lines `await asyncio.sleep(0.1)` after each status yield → **deleted**
- `result = self.rag_service.retrieve_initial(state)` → `result = await asyncio.to_thread(self.rag_service.retrieve_initial, state)`
- Same pattern for `refine_query_prf` and `retrieve_final_dual`
- The `await asyncio.sleep(0.1)` after "Classement des résultats…" → **deleted** (the existing `asyncio.to_thread(rerank)` already frees the loop)

- [ ] **Step 2: Ask the user to restart the backend**

Tell the user to run:
```
! sudo systemctl restart craftpilot-backend
```
Wait for confirmation before proceeding to Task 2.

- [ ] **Step 3: Smoke-test the server fix in isolation**

Send a message in the chat widget. Open the browser DevTools Network tab, find the `chat_proxy.php` request, and watch the response stream. Each status event should appear as a separate chunk with a visible time gap before the next one. If all five status lines appear in one burst right before generation starts, the `asyncio.to_thread` change did not take effect (check the service restarted).

- [ ] **Step 4: Commit the server change**

```bash
git -C /opt/craftpilot_backend add pipeline.py
git -C /opt/craftpilot_backend commit -m "fix(pipeline): unblock event loop between status yields via asyncio.to_thread

Wrapping retrieve_initial, refine_query_prf, and retrieve_final_dual in
asyncio.to_thread() ensures uvicorn can flush each status HTTP chunk to the
socket before the next blocking call starts. Previously, asyncio.sleep(0.1)
was used but the event loop re-blocked immediately on the sync call.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KAhxwRU92wXdUndwFV4ANd"
```

---

## Task 2: Client — force a browser repaint between status events

**Files:**
- Modify: `/var/www/html/public/local/craftpilot/amd/src/chat_interface.js`
  - `showTyping()` — line 1016: remove inline `style=` from `.cp-status-hint` span
  - `streamFromBackend()` — lines 1134–1256: convert inner loop + remove debug overlay

**Interfaces:**
- `requestAnimationFrame(callback)` — browser API; schedules `callback` before the next paint cycle. `await new Promise(resolve => requestAnimationFrame(resolve))` suspends an async function until the next frame, guaranteeing a repaint.
- `for...of` loop — unlike `forEach`, supports `await` inside the loop body
- The `read` function changes from `() => Promise` to `async () => Promise` — the outer `.then()` chain is unaffected because async functions return Promises

- [ ] **Step 1: Fix `showTyping()` — remove inline style from hint span**

Find `showTyping()` at line 1006. Change the `.cp-status-hint` span from:

```javascript
'<span class="cp-status-hint" aria-live="polite" style="margin-left:10px;font-style:italic;font-size:12px;color:#555;display:inline-block"></span>'
```

to:

```javascript
'<span class="cp-status-hint" aria-live="polite"></span>'
```

The `.cp-status-hint` rule in `styles.css` already provides `margin-left`, `font-size`, `font-style`, and `color`. The inline style was a debugging fallback and contradicts the project's no-inline-styles rule.

- [ ] **Step 2: Rewrite the `read` function inside `streamFromBackend`**

The current `read` function (lines 1134–1256) uses `.then()` chaining and a `lines.forEach` loop. Replace the entire `read` constant and the `return read()` call below it with the following async version. Everything outside this block (the `activateBubble` closure, `fullResponse`, `buf`, `reader`, `decoder`) stays unchanged.

Replace from:
```javascript
            const read = () => reader.read().then(({ done, value }) => {
```
down to (and including):
```javascript
            return read();
        })
```

With:

```javascript
            const read = async () => {
                const { done, value } = await reader.read();

                if (done) {
                    if (String(streamConvId) !== String(state.currentConvId)) {
                        finishStreaming();
                        return;
                    }
                    const b = activateBubble();
                    const cleanFinal = stripThinkTags(fullResponse);
                    b.innerHTML = renderMarkdown(cleanFinal);
                    extractFollowUpQuestions(cleanFinal);
                    const finishedSources = getConvState(streamConvId).sources;
                    saveMessage(streamConvId, 'ai', cleanFinal,
                        finishedSources.length ? JSON.stringify({sources: finishedSources}) : '');
                    finishStreaming();
                    return;
                }

                buf += decoder.decode(value, { stream: true });
                const lines = buf.split('\n');
                buf = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const ev = JSON.parse(line);

                        if (ev.event === 'conversation_title' && ev.data) {
                            const newTitle = ev.data;
                            state.currentConvTitle = newTitle;
                            updatePanelTitle(newTitle);
                            const conv = state.conversations.find(
                                c => String(c.conversation_id || c.id) === String(streamConvId)
                            );
                            if (conv) conv.title = newTitle;
                            renderConversations(state.conversations, streamConvId);
                            Ajax.call([{
                                methodname: 'local_craftpilot_manage_conversations',
                                args: { action: 'update', conversation_id: streamConvId, title: newTitle },
                            }])[0].catch(err => console.error('CraftPilot: title persist failed', err));

                        } else if (ev.event === 'status' && ev.data) {
                            const hintEl = typingEl.querySelector('.cp-status-hint');
                            if (hintEl) hintEl.textContent = ev.data;
                            // Yield one animation frame so the browser renders this
                            // hint before processing the next event in the same chunk.
                            await new Promise(resolve => requestAnimationFrame(resolve));

                        } else if (ev.event === 'video_metadata' && ev.data) {
                            const vm    = ev.data;
                            const isBVH = (vm.filename || '').toLowerCase().endsWith('.bvh');
                            addSource({
                                id:          vm.video_id || vm.bvh_id || vm.filename,
                                type:        isBVH ? 'bvh' : 'video',
                                filename:    vm.filename,
                                video_url:   isBVH ? null : vm.video_url,
                                bvh_url:     isBVH ? (vm.bvh_url || vm.video_url) : null,
                                start_time:  vm.start_time,
                                end_time:    vm.end_time,
                                duration:    vm.duration,
                                project_name: vm.project_name,
                            }, streamConvId);

                        } else if (ev.event === 'bvh_metadata' && ev.data) {
                            const bm = ev.data;
                            addSource({
                                id:          bm.bvh_id || bm.filename,
                                type:        'bvh',
                                filename:    bm.filename,
                                bvh_url:     bm.bvh_url || bm.url,
                                duration:    bm.duration,
                                frame_count: bm.frame_count,
                            }, streamConvId);

                        } else if (ev.event === 'token' && ev.data) {
                            fullResponse += ev.data;
                            const visible = stripThinkTags(fullResponse);
                            activateBubble().innerHTML = renderMarkdown(visible);
                            scrollBottom();

                        } else if (ev.event === 'message' && ev.content) {
                            ev.content.forEach((c) => { if (c.content) fullResponse += c.content; });
                            const visible = stripThinkTags(fullResponse);
                            activateBubble().innerHTML = renderMarkdown(visible);
                            scrollBottom();

                        } else if (ev.event === 'documents' && Array.isArray(ev.data)) {
                            ev.data.forEach((doc) => {
                                if (doc.type === 'video_annotation') return;
                                addSource({
                                    id:           doc.source,
                                    type:         'text',
                                    source:       doc.module_name || doc.source,
                                    content:      doc.page_content_preview,
                                    module_id:    doc.module_id,
                                    module_type:  doc.module_type,
                                    course_id:    doc.course_id,
                                    heading_path: doc.heading_path,
                                    section_name: doc.section_name,
                                }, streamConvId);
                            });

                        } else if (ev.content === '[DONE]') {
                            /* handled on stream close */

                        } else if (ev.event === 'error' || ev.type === 'error') {
                            if (msgEl && msgEl.parentNode) msgEl.parentNode.removeChild(msgEl);
                            if (typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
                            showError(ev.message || 'The AI backend returned an error.');
                            finishStreaming();
                        }
                    } catch (_) {
                        /* non-JSON lines — ignore */
                    }
                }

                return read();
            };

            return read();
        })
```

What changed vs the original:
- `const read = () => reader.read().then(...)` → `const read = async () => { const {done,value} = await reader.read(); ... }`
- `lines.forEach((line) => { ... })` → `for (const line of lines) { ... }`
- `status` handler: removed the entire `cp-dbg-overlay` block (8 lines); added `await new Promise(resolve => requestAnimationFrame(resolve))` after setting `hintEl.textContent`
- All other event handlers (`conversation_title`, `video_metadata`, `bvh_metadata`, `token`, `message`, `documents`, `error`) are character-for-character identical to the original

- [ ] **Step 3: Build the AMD bundle**

Run from the plugin root:
```bash
cd /var/www/html/public/local/craftpilot && ./node_modules/.bin/grunt babel --force
```

Expected output ends with something like:
```
Done, without errors.
```
The `--force` flag is required because the `.map` file is root-owned; it does not affect functionality.

- [ ] **Step 4: Purge the Moodle server-side JS cache**

Ask the user to run:
```
! sudo -u apache php /var/www/html/public/admin/cli/purge_caches.php
```

Alternatively, they can go to **Moodle admin → Site administration → Development → Purge all caches**. A browser hard-refresh alone is not sufficient — Moodle repacks AMD bundles server-side.

- [ ] **Step 5: Verify end-to-end**

Send a message in the Moodle CraftPilot widget and confirm:

1. The typing indicator appears immediately after send
2. The hint text inside the typing indicator reads "Recherche dans la base de connaissances…" briefly
3. It transitions to "Reformulation de la question…" (holds for a few seconds while the LLM refines the query)
4. Then "Récupération des sources pertinentes…" briefly
5. Then "Classement des résultats…" (holds ~20–30 s on this server)
6. Then "Génération de la réponse…" briefly before the first token appears
7. The typing indicator disappears cleanly when the first token arrives
8. No red overlay (`cp-dbg-overlay`) appears anywhere on the page
9. Inspect the `.cp-status-hint` span in DevTools — confirm no `style=` attribute is present

- [ ] **Step 6: Commit the client change**

```bash
git -C /opt/craftpilot_backend add docs/  # plan file only; JS lives outside the repo
git -C /opt/craftpilot_backend commit -m "fix(client): force repaint between status hints via requestAnimationFrame

- forEach -> for...of in stream read loop so we can await inside it
- await one animation frame after each status event so the browser
  renders each hint before processing the next line in the same chunk
- remove cp-dbg-overlay diagnostic block
- remove inline style= from .cp-status-hint span (css rule already covers it)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KAhxwRU92wXdUndwFV4ANd"
```

Note: `chat_interface.js` changes are not tracked in the backend git repo (it lives in the Moodle plugin directory with no VCS). The commit here records the plan doc update only. If the Moodle plugin directory ever gets a git repo, the JS change should be committed there.
