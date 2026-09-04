# 01 — Annotation tool: mandatory craft selection on video load
**Agent 1 · 2026-09-04 · area: /opt/video_elicitation_annotation_tool/**

## Problem
`js/app.js` seeded the active craft domain with `state.craft = localStorage.getItem('craft') || 'glassblowing'`,
and the annotation upload repeated the same fallback (`formData.append('craft', state.craft || 'glassblowing')`).
Any annotator who never opened the craft dropdown therefore filed every recording as `glassblowing` without
being asked and without any visible sign that a choice had been made on their behalf. That value is stored in
`mdl_local_videoelicit_annotations.craft` and becomes ChromaDB metadata that a CraftPilot retrieval filter
depends on, so a wrong value does not fail loudly — it silently returns another craft's videos to unrelated
questions. In production this produced 4 mislabelled rows out of 16 (ids 56, 63, 64, 76: glovemaking
transcripts tagged `glassblowing`), which is how a glassblowing question ended up sourced to a glovemaking
video. The rows were corrected by hand, but the default kept manufacturing new ones. The fix removes every
implicit craft value and replaces it with an explicit, blocking confirmation on each video load.

## Files touched
| File | Lines/functions | What changed |
|---|---|---|
| `js/app.js` | `TRANSLATIONS.en` / `TRANSLATIONS.fr` (after `addCraftError`) | Added 5 strings in both supported languages: `craftGateTitle`, `craftGateBody`, `craftGatePlaceholder`, `craftGateConfirm`, `craftGateRequired`. |
| `js/app.js` | `state` object literal | `craft: 'glassblowing'` → `craft: ''`. Added `rememberedCraft: ''` (localStorage value, pre-selection only) and `craftGateOpen: false`. |
| `js/app.js` | `initializeApp()` | Removed `state.craft = localStorage.getItem('craft') \|\| 'glassblowing'`. The stored value is now read into `state.rememberedCraft` and never becomes the active craft on its own. |
| `js/app.js` | `loadCustomCrafts()` | After appending `/api/crafts` options, calls `refreshCraftGateOptions()` if the gate is open, so custom domains that arrive late still show up in it. |
| `js/app.js` | **New**: `setAppInteractive()`, `setVideoPlayerUsable()`, `refreshCraftGateOptions()`, `openCraftGate()`, `confirmCraftSelection()`, `syncMainCraftSelector()`, `closeCraftGate()` (inserted between `showAddCraftInput()` and `createElicitControlsUI()`) | The craft gate. Builds a blocking modal with `document.createElement`, mirrors the option list from the existing `#craftSelector` (no second hardcoded list), reuses `showAddCraftInput()` for the "+" custom-domain flow, and marks every other direct child of `<body>` `inert` + `aria-hidden` while it is up. |
| `js/app.js` | `createElicitControlsUI()` | Added a disabled empty placeholder option (`data-craft-key="craftGatePlaceholder"`, so `applyLanguage()` translates it) as the first entry of `#craftSelector`; `craftSelect.value = state.craft \|\| ''` instead of `\|\| 'glassblowing'`; the `change` handler ignores the empty value. |
| `js/app.js` | `loadVideo()` | `openCraftGate()` on the first line, before any `await`, so the block is in place before the player container is unhidden. On a failed load the `catch` calls `closeCraftGate()` so a network error cannot trap the annotator. |
| `js/app.js` | `startRecording()` | Safety net: `if (!state.craft) { openCraftGate(); return; }` before `getUserMedia`. |
| `js/app.js` | `handleRecordingStop()` | `formData.append('craft', state.craft \|\| 'glassblowing')` → append only when `state.craft` is set, otherwise `console.warn`. The backend's `craft: Optional[str] = Form(None)` stores NULL, which is recoverable; a guessed value is not. |
| `css/styles.css` | appended at end of file (below the FULL SCRUB block, per repo CLAUDE.md) | New `.craft-gate`, `.craft-gate-card`, `.craft-gate-title`, `.craft-gate-body`, `.craft-gate-field`, `.craft-gate-row`, `.craft-gate-select`, `.craft-gate-add`, `.craft-gate-error`, `.craft-gate-confirm` rules using existing Studio tokens. Overlay `z-index: 4000` (above the loading overlay at 2000 and toasts at 3000). |

No other file was modified. `backend/main.py`, `backend/moodle_db.py`, `backend/database_compat.py` and
`CLAUDE.md` show as dirty in `git status` but were already modified before this work started — they are not
part of this change.

## Behaviour before → after

**Before**
- Page load: `state.craft` = stored value, or `'glassblowing'` if nothing stored. The control bar showed
  "Glassblowing" as if the annotator had picked it.
- Video load: player immediately usable; no prompt.
- Recording saved: `craft` always sent, using the assumed value when untouched.

**After**
- Page load: `state.craft` is `''`. The control-bar dropdown shows a disabled "— choose a craft domain —"
  placeholder. The stored value is kept only in `state.rememberedCraft`.
- Video load (`loadVideo()`, which also covers the Select Video modal and `loadVideoAndSegment()`): a modal
  appears immediately; every other direct child of `<body>` gets `inert` + `aria-hidden="true"`, and the
  `<video>` element is paused, loses its `controls` attribute and gets `tabindex="-1"`. `inert` removes the
  subtree from hit-testing, the tab order and keyboard handling; removing `controls` additionally kills the
  video element's own space/arrow-key handling on browsers that do not support `inert`.
- The modal has no close button, no backdrop-click dismissal and no Escape handler. Confirming with nothing
  selected shows an inline error and keeps the block. The only way through is to pick a domain (or create one
  with "+", which is hidden when `window.USER_ID` is absent, matching the control bar).
- On confirm: `state.craft` and `localStorage.craft` are set, `#craftSelector` is synced (adding the option if
  the gate created a new custom domain), the `inert` attributes and the video's `controls`/`tabindex` are
  restored, and the modal is removed.
- Recording saved: `craft` is sent only when confirmed; otherwise the field is omitted and stored as NULL.

**Decision — a remembered craft does NOT skip the prompt.** It only pre-selects the dropdown. The whole
incident came from a sticky implicit value, and the same annotator legitimately moves between crafts (that is
precisely the glovemaking case). Skipping the prompt for a remembered value would rebuild the failure mode with
extra steps: an annotator who did glassblowing yesterday and gloves today would never be asked. Pre-filling
keeps the common case to one click while still requiring the craft name to pass in front of the annotator's
eyes and be actively confirmed for each video.

## How to verify

Static:
```bash
cd /opt/video_elicitation_annotation_tool
node --check js/app.js          # must print nothing
grep -n "|| 'glassblowing'" js/app.js   # must return no matches
```

In the browser (hard refresh, `Ctrl+Shift+R` — `NoCacheStaticFiles` serves fresh, no service restart needed):
1. Open the Elicit tab, click **Select Video**, pick any video.
2. The craft modal appears at once. Try to click the video, press space, press Tab — nothing in the page
   behind the modal responds and focus never leaves the modal.
3. Click **Confirm and start** with the placeholder still selected → "Choose a craft domain to continue."
   in red, modal stays.
4. Pick a domain, click **Confirm and start** → modal disappears, video controls come back, the control-bar
   dropdown shows the confirmed domain.
5. Load a second video → the modal appears again, pre-selected with the domain confirmed in step 4.
6. Switch the UI to French and repeat: title, body, placeholder, button and error are all French.

DevTools console equivalents of the above (these are what was actually run):
```js
openCraftGate();
document.querySelector('.app-container').hasAttribute('inert');            // true
document.getElementById('videoPlayer').hasAttribute('controls');           // false
document.getElementById('videoPlayer').getAttribute('tabindex');           // "-1"
document.getElementById('craftGateConfirmBtn').click();                    // gate stays, error shown
document.getElementById('craftGateSelector').value = 'glovemaking';
document.getElementById('craftGateConfirmBtn').click();
localStorage.getItem('craft');                                             // "glovemaking"
document.getElementById('craftSelector').value;                            // "glovemaking"
Array.from(document.body.children).some(el => el.hasAttribute('inert'));   // false
```

## How to revert
Both files are tracked and were clean before this change:
```bash
cd /opt/video_elicitation_annotation_tool
git checkout -- js/app.js css/styles.css
```
(Do **not** run a bare `git checkout .` — `backend/*.py` and `CLAUDE.md` carry unrelated uncommitted work.)

To revert by hand instead, the minimal undo is:
- `js/app.js` `state`: `craft: ''` → `craft: 'glassblowing'`
- `js/app.js` `initializeApp()`: restore `state.craft = localStorage.getItem('craft') || 'glassblowing';`
- `js/app.js` `loadVideo()`: delete the `openCraftGate();` call and the `closeCraftGate();` in the `catch`
- `js/app.js` `startRecording()`: delete the `if (!state.craft) { openCraftGate(); return; }` guard

Reverting the JS alone is enough to restore old behaviour; the CSS block is inert once nothing builds a
`.craft-gate` element.

Note: annotators who have already confirmed a craft carry `localStorage.craft`. Nothing needs clearing — a
stored value can no longer become an active craft without confirmation.

## Known limits / not done
- **Verified by execution** (headless Chromium via Playwright against the live backend on 127.0.0.1:8005):
  `node --check`; the placeholder in `#craftSelector`; `state.craft` not adopting the remembered value on
  boot; `loadVideo()` opening the gate synchronously before its first `await`; `inert` + `aria-hidden` applied
  to all body children except the gate; `controls` removed and `tabindex="-1"` set on the video; confirm-with-
  nothing-selected being refused; confirm writing `localStorage.craft` and syncing `#craftSelector`; full
  release of `inert`/`controls` on close; the gate re-opening from `startRecording()` when no craft is set;
  the gate releasing after a failed video fetch; and both EN and FR string sets rendering correctly.
- **Verified by reading only:** the `showAddCraftInput()` "+" path inside the gate (it needs an authenticated
  Moodle JWT and a live `POST /api/crafts`, which the local session did not have — `window.USER_ID` was absent,
  so the button was hidden as designed). Its wiring is the same call the control bar already uses, with the
  gate's field element as the wrapper, but the create-a-new-domain-from-the-gate flow has **not** been clicked
  through end to end.
- **Not tested inside the Moodle iframe**, and not tested against a real video file — only against the
  standalone page, where `/api/videos` 401s without a JWT (expected per repo CLAUDE.md).
- **The Segment tab is not gated.** `loadSegmentPlayer()` loads a video into the small segment player without
  the gate. That path does not create annotations, so it cannot mislabel data; gating it would block the
  segmentation workflow for no benefit. Worth revisiting if segmenting ever starts writing `craft`.
- **Language cannot be changed while the gate is up** (the switcher is `inert`). The gate builds its strings
  from `currentLang` at open time and re-translates built-in option labels on every refresh, so it is always
  correct for the language in effect when it opens.
- Nothing was committed, pushed, or restarted, per instructions. The change is live on disk and served by
  `NoCacheStaticFiles` on the next hard refresh.
