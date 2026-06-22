# Design: Fix RAG status hint streaming

**Date:** 2026-06-22  
**Branch:** `feat/status-hints-streaming`  
**Status:** Approved

---

## Problem statement

The RAG pipeline yields five status events before each step so the UI can show
an italic hint ("Recherche dans la base de connaissances…", etc.) while the
backend processes the request. Despite the events being yielded and the PHP
proxy correctly configured (output buffering disabled, `flush()` after every
`echo`), only the final status event ("Génération de la réponse…") ever reaches
the browser visibly.

---

## Root cause (confirmed)

`retrieve_initial`, `refine_query_prf`, and `retrieve_final_dual` are
**synchronous functions called directly inside an `async def` generator**.
Python's asyncio event loop is single-threaded. Calling a sync function there
freezes the entire event loop — including uvicorn's I/O layer — for the
duration of the call. Uvicorn cannot flush the previously-yielded HTTP chunk to
the OS socket while the event loop is frozen.

The two patches tried (`asyncio.sleep(0)`, `asyncio.sleep(0.1)`) both fail for
the same reason: the event loop is freed for one tick by the sleep, but the
*very next instruction* is a blocking sync call that re-freezes it — before the
I/O selector can drain uvicorn's write buffer.

`rerank` already uses `await asyncio.to_thread(...)`. That is why "Génération
de la réponse…" is the only hint that works: it is the only pipeline step that
genuinely frees the event loop.

**FastAPI docs confirm:**  
> FastAPI itself wraps every sync endpoint function in `run_in_threadpool`
> because calling a blocking function in a coroutine blocks the event loop.

**Python asyncio docs confirm:**  
> `asyncio.to_thread()` is primarily designed for IO-bound or CPU-bound
> functions that would otherwise block the event loop if run in the main thread.

---

## Solution

Two-layer fix: server-side (root cause) + client-side (safety net).

### Layer 1 — Server: `pipeline.py`

Wrap the three blocking sync calls in `asyncio.to_thread()`, matching the
pattern already used for `rerank`. Remove the now-useless `asyncio.sleep()`
calls — the `await` itself yields the event loop for the entire duration of the
thread.

```python
# OLD — freezes the event loop:
yield status_event
await asyncio.sleep(0.1)          # one useless tick
result = self.rag_service.retrieve_initial(state)   # blocks event loop

# NEW — event loop stays free during the blocking call:
yield status_event
result = await asyncio.to_thread(self.rag_service.retrieve_initial, state)
```

Apply to:
- `retrieve_initial`
- `refine_query_prf`
- `retrieve_final_dual`

`rerank` already uses `asyncio.to_thread` — no change there.

**Thread safety:** Each function takes `state` (a request-local dict) and
returns a result dict. They do not mutate shared service-level state and run
strictly sequentially (not concurrently). No locking required.

### Layer 2 — Client: `amd/src/chat_interface.js`

Even with the server fix, two events can arrive in the same TCP segment if a
step completes in less than one network RTT. The current `lines.forEach` loop
processes all lines in one synchronous JS task — the browser never repaints
between them, so only the last status text in a batch is ever rendered.

Fix: change the inner loop from `forEach` to `for...of` (so we can `await`
inside it) and after each `status` event, yield to the browser's paint
scheduler:

```javascript
for (const line of lines) {
    const ev = JSON.parse(line.trim());
    if (!ev || !ev.event) continue;

    if (ev.event === 'status') {
        const hintEl = typingEl.querySelector('.cp-status-hint');
        if (hintEl) hintEl.textContent = ev.data;
        // Yield one animation frame so the browser can render this hint
        // before we process the next event in the same chunk.
        await new Promise(resolve => requestAnimationFrame(resolve));

    } else if (ev.event === 'token') {
        // Tokens are processed immediately — no pause.
        activateBubble().textContent += ev.data;

    } // … other events unchanged
}
```

Cost: ~16 ms per status event (one animation frame). Completely imperceptible
for steps that take hundreds of milliseconds to seconds. Token throughput is
unaffected.

---

## Cleanup in the same commit

These were temporary scaffolding from debugging:

1. **Remove the diagnostic red overlay** (`cp-dbg-overlay`) — the `div`
   creation block and `dbg.textContent = ev.data` line inside the `status`
   handler in `chat_interface.js`.

2. **Remove inline `style=` from the hint `<span>`** in `showTyping()`.  
   The rule `.cp-status-hint { … }` in `styles.css` already covers all
   necessary styling. The inline attribute was added as a fallback during
   debugging and contradicts the project rule against inline styles.

---

## Files changed

| File | Location | Change |
|---|---|---|
| `pipeline.py` | `/opt/craftpilot_backend/` | Wrap 3 sync calls in `asyncio.to_thread`; remove `asyncio.sleep` calls |
| `chat_interface.js` | `amd/src/` (Moodle plugin) | `forEach` → `for...of` + RAF after status events; remove debug overlay; remove inline style |

After editing `chat_interface.js`, the standard deploy sequence applies:
1. `grunt babel --force` in the plugin root
2. Purge Moodle server-side JS cache

---

## What does not change

- PHP proxy (`chat_proxy.php`) — already correct
- CSS (`.cp-status-hint` in `styles.css`) — already correct
- The `showTyping()` DOM structure — hint `<span>` stays inside `.cp-typing`
- All other event types (`token`, `message`, `documents`, `error`, `video_metadata`) — logic unchanged

---

## Verification checklist

1. Send a prompt in the Moodle CraftPilot widget
2. The typing indicator should show each hint in sequence:
   - "Recherche dans la base de connaissances…" (briefly, ~0.5 s)
   - "Reformulation de la question…" (a few seconds while the LLM refines)
   - "Récupération des sources pertinentes…" (briefly)
   - "Classement des résultats…" (20–30 s on this server)
   - "Génération de la réponse…" (until first token)
3. No red overlay appears anywhere on the page
4. No inline `style=` attribute on the `.cp-status-hint` span (inspect DOM)
5. Typing indicator is removed cleanly on first token
6. Full response renders correctly with sources
