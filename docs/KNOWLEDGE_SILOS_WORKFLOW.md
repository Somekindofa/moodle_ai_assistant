# Knowledge Silos — How to Use Them

CraftPilot runs on one shared Moodle platform used by multiple, sometimes
competing, companies. Without this feature, any authenticated user could
ask the chat a question and have it retrieve — and cite — another
company's proprietary video annotations and transcripts. This is how that
boundary is actually configured and operated. Full design rationale:
`docs/superpowers/specs/2026-06-23-rag-knowledge-silos-design.md`. This
file is the practical "how do I actually do it" companion to that spec.

Two repos are involved: `craftpilot_backend` (this repo, does the
enforcement) and `video_elicitation_annotation_tool` (where an expert
assigns their project to a cohort).

---

## The mental model: two boundaries, not one

| Content | Boundary | Controlled by | Enforced by |
|---|---|---|---|
| Video annotations & transcripts | **Cohort membership** (`mdl_cohort_members`) | Admin (setup) + delegated expert (day-to-day) | ChromaDB metadata filter on every query |
| Course pages/PDFs/resources | **Course enrollment** (`mdl_user_enrolments`) | Teacher + admin (already how Moodle works) | Only enrolled courses' collections are ever opened |

Cohort membership is used for annotations specifically because it is
**admin-controlled only** — there's no self-enrollment loophole, which
matters for content meant to stay behind a company boundary.

**Scope note:** this controls what the CraftPilot chat can *retrieve and
cite*. It is not a general file-permission system — if a video is
reachable through some other Moodle mechanism by direct URL, that's a
separate, unrelated access layer.

---

## Part 1 — One-time setup per company (Moodle site admin)

Do this once when a new company joins the platform. `craftpilot_backend`'s
`SiloService` reads whatever this setup produces — there is nothing to
configure in the backend itself.

1. **Site administration → Courses → Manage courses and categories** →
   create a course category named after the company (e.g. *"Company A"*).
2. **Site administration → Users → Cohorts** → create a cohort inside that
   category's context (e.g. *"Company A — Apprentices"*).
3. **Site administration → Users → Define roles** → create a custom role
   `cohort_delegate`, reusable across every company, with **only** the
   `moodle/cohort:assign` capability. Do **not** grant
   `moodle/cohort:manage` — a delegate can add/remove members of a cohort
   they're assigned to, but cannot create or delete cohorts, and cannot
   touch any cohort outside their assigned context.
4. Go to the company's category → **Assign roles** → assign the company's
   designated expert the `cohort_delegate` role **at that category context
   only**. (Same screen used earlier in this project for assigning
   Course creator/Manager roles at a category — see
   `docs/PLAYWRIGHT_DEBUGGING.md` §5 if you need a refresher on where that
   screen lives.)
5. Enable email self-registration with new-account approval delegated to
   the category manager, not the site admin — so new colleagues register
   themselves and the expert approves them, without you in the loop.

After this, the expert manages their cohort's membership independently
via **Site administration → Users → Cohorts → Assign** (their
`cohort_delegate` role scopes what they can see there to their own
company's cohort only). You are not involved again for day-to-day
membership changes.

---

## Part 2 — Assigning a project to a cohort (the expert, day-to-day)

This is a *separate* capability from `cohort_delegate` above.
`cohort_delegate` controls **cohort membership** in Moodle; assigning a
**project** to a cohort happens inside the Video Elicitation Tool, and
requires that tool's own `manage` capability — which, per
`backend/auth.py`'s role map, only `editingteacher` (Moodle's "Teacher"
role, not "Non-editing teacher") and `admin` have. A plain Teacher role in
the tool can view and annotate but cannot reassign a project's cohort.

1. Open the Video Elicitation Tool, open the project's settings (the
   pencil icon in the projects panel).
2. A **Visibility** dropdown appears:
   - *Open access — visible to all authenticated CraftPilot users*
     (default)
   - One option per cohort the expert manages (i.e. has a
     teacher/editingteacher/manager role in a course that cohort is
     enrolled into)
3. Pick one, save.

That's the whole action — there is no separate "apply" or "resync"
button. Saving triggers, automatically and transparently:

- The project's `allowed_cohort_id` is updated in the tool's own database.
- The tool calls `craftpilot_backend`'s
  `POST /api/resync-project-annotations` internally.
- The backend deletes that project's existing ChromaDB documents and
  re-ingests them with the new `cohort_id` / `open_access` metadata.

If the expert sees a contact-admin notice instead of the dropdown, it
means Part 1 hasn't happened for them yet (no `editingteacher`/`manager`
role on any cohort-enrolled course) — that's the signal to go do Part 1,
not a bug.

There's also a one-time dismissible banner shown to an expert who manages
a cohort but still has projects left on "open access" — a nudge, not an
enforcement mechanism; nothing changes automatically until they actually
open a project and pick a cohort.

---

## Part 3 — How enforcement actually works (for sanity-checking, not for configuring)

Every retrieval call is filtered:

```python
# services/rag_service.py — build_cohort_filter()
{
    "$or": [
        {"cohort_id": {"$in": user_cohort_ids}},  # cohorts this user belongs to
        {"open_access": True},                     # content marked open
    ]
}
```

Concretely:

- User belongs to zero cohorts → only `open_access` content is ever
  returned. There is no fallback to "everything."
- Moodle DB unreachable when resolving a user's cohorts/enrollments →
  the request fails **closed**: HTTP 503 in the request path (or,
  in the streaming chat path, an `{"event": "error", ...}` line
  followed by `[DONE]`) — nothing is retrieved, nothing leaks.
- `user_id` missing or `<= 0` → HTTP 403 immediately at
  `api/routes.py`, before any retrieval is attempted.

The guiding rule the whole design follows: **when in doubt, return
nothing rather than returning too much.**

---

## How to verify a silo is actually working

1. Pick a project you've assigned to a cohort.
2. As a user who **is** in that cohort: ask the chat something only that
   project's content could answer — it should retrieve and cite it.
3. As a user who is **not** in that cohort (and has no other cohort that
   would grant access): ask the same question — it should not surface
   that content, and should fall back to the generic "not found" message
   if nothing else matches.
4. Optionally, inspect the actual stored metadata directly (see
   `docs/PLAYWRIGHT_DEBUGGING.md` §6 for the `.env`-permission workaround
   needed to query ChromaDB from a shell): each annotation document for a
   restricted project should carry `cohort_id: <the cohort's id>` and
   `open_access: False`; an open-access one carries `cohort_id: -1` and
   `open_access: True`.

---

## Quick reference — who does what, where

| Action | Who | Where |
|---|---|---|
| Create a cohort for a new company | Site admin | Moodle: Site administration → Users → Cohorts |
| Delegate cohort-membership management | Site admin | Moodle: category → Assign roles → `cohort_delegate` |
| Add/remove people from a cohort | Delegated expert | Moodle: Site administration → Users → Cohorts → Assign |
| Assign a project's annotations to a cohort | Expert with `editingteacher`/`manage` | Video Elicitation Tool: project settings → Visibility |
| Everything else (filtering, resync, fail-closed behaviour) | Automatic | `craftpilot_backend` — nothing to configure |
