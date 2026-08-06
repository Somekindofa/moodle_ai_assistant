# Conversation Isolation Bugs — Fixed (March 2026)

All fixes are in `/var/www/html/public/local/craftpilot/amd/src/chat_interface.js`.

**Root causes and fixes:**

| Bug | Root cause | Fix |
|-----|-----------|-----|
| New conversation shows old history | `createConversation` only cleared DOM inside async `.then()` — a race window let old content appear | Moved all state/DOM resets (messages clear, sources clear, `showReady`) **before** the AJAX call (sync); AJAX is now fire-and-forget |
| Sources appear on wrong conversation | `addSource` wrote to a shared `state.sources` array with no ownership tracking | `addSource` now accepts an `ownerConvId` parameter; it silently drops the source if `ownerConvId !== state.currentConvId` |
| Sources lost on conversation switch | No per-conversation save/restore mechanism | Replaced `convSources` dict with `convStates` + `getConvState(id)` helper; `selectConversation` saves outgoing sources and restores incoming ones |
| Old history loads into new conversation | `loadMessages` AJAX could resolve after the user switched away | Guard at top of `.then()`: `if (String(convId) !== String(state.currentConvId)) return;` |
| Stream writes to wrong conversation | `state.currentConvId` was read live inside a long-running stream closure | Capture `const streamConvId = state.currentConvId` once at the top of `streamFromBackend`; all inner references use `streamConvId` |
| Stream response arrives after switch | No staleness check at the start of the `.then((res)=>` handler | Checks `streamConvId !== state.currentConvId`; cancels the response body and calls `finishStreaming()` |
| `clearSources()` left ghost cards visible | Empty-items path only toggled CSS classes; never cleared `dom.sourcesScroll.innerHTML` | Added `dom.sourcesScroll.innerHTML = ''` in the empty path of `setSources` |
