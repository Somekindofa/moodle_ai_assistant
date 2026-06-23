# RAG Knowledge Silos — Design Spec
**Date:** 2026-06-23
**Status:** Approved, ready for implementation planning

---

## Problem

CraftPilot's RAG is used across a shared Moodle platform where competing companies coexist. Currently any authenticated user can query any content in the vector store — video annotation know-how, course documents, transcripts — regardless of which company produced it. A user from Company B can retrieve Company A's proprietary elicitation segments and transcripts. This is a knowledge breach.

**Goal:** enforce cohort-level access boundaries so that retrieval is pre-filtered to content the requesting user is authorised to see. No restricted content is ever fetched, scored, or surfaced for an unauthorised user — the filter is applied before retrieval, not after.

---

## Chosen Approach

**Approach B — Validated User ID + Backend Cohort Lookup.**

- The PHP proxy (`chat_proxy.php`) validates the Moodle session and cross-checks that the `user_id` in the request body matches `$USER->id`. 307 redirect and streaming architecture are unchanged.
- The CraftPilot Python backend receives `user_id`, queries Moodle DB for cohort memberships, and scopes all ChromaDB queries accordingly.
- Video annotations are tagged with `cohort_id` at project level (set by the expert in the Video Elicitation Tool UI).
- Course content is already per-collection (`course_{id}`); it is scoped by active course enrollment.

---

## Moodle Operational Model

### Silo boundary definitions

| Content type | Silo boundary | Controlled by |
|---|---|---|
| Video annotation segments & transcripts | `mdl_cohort_members` — cohort membership | Site admin scaffolding + delegated expert |
| Course pages, PDFs, resources (CourseRAGService) | `mdl_user_enrolments` — active course enrollment | Teacher + admin |

Cohort membership is used for annotations because it is **admin-controlled only** — no self-enrollment loophole. Course enrollment is acceptable for course content because Moodle already enforces visibility at that boundary.

### Per-company onboarding (admin does once)

When a new company joins the platform:

1. Create a **course category** named after the company (e.g., *"Company A"*).
2. Inside that category context, create a **cohort** (e.g., *"Company A — Apprentices"*).
3. Create (once, reusable) a custom Moodle role `cohort_delegate` with capability `moodle/cohort:assign` only. Do **not** grant `moodle/cohort:manage` — delegates cannot create or delete cohorts.
4. Assign the company's designated expert the `cohort_delegate` role **at the Company A category context only**. They cannot touch any other category.
5. Enable email self-registration with approval delegated to the category manager (not the site admin). New colleagues register themselves; the expert approves.

After this five-minute setup, the expert manages their cohort independently. The site admin is never in the loop for day-to-day membership changes.

### What experts cannot do

- Create or delete cohorts (only `moodle/cohort:assign`, not `moodle/cohort:manage`).
- Manage cohorts outside their category.
- Self-enroll in cohorts they do not manage.

---

## Data Flow (with silos)

```
1. Moodle renders page → M.cfg.userid is available in browser JS (standard Moodle global).

2. Browser POSTs to chat_proxy.php:
   { message, conversation_thread_id, course_id, user_id: M.cfg.userid, sesskey, ... }

3. chat_proxy.php:
   a. require_login()                           — Moodle session valid
   b. confirm_sesskey($data['sesskey'])         — sesskey is cryptographically bound to this user
   c. assert (int)$data['user_id'] === $USER->id  — NEW: mismatch → HTTP 403, no redirect
   d. 307 redirect → /craftpilot-api/chat      — unchanged, streaming preserved

4. Browser re-POSTs same body to Apache.
   Apache injects X-Internal-Token header.     — proves request passed through Moodle server

5. CraftPilot backend receives { user_id, course_id, message, ... }:
   a. SiloService.get_allowed_cohorts(user_id)   — Moodle DB query, cached 60 s
   b. SiloService.get_enrolled_course_ids(user_id) — Moodle DB query, cached 60 s
   c. user_id missing, zero, or non-integer → HTTP 403 immediately, no retrieval
   d. allowed_cohorts empty AND enrolled_courses empty → return empty result

6. Retrieval:
   a. Video annotations → ChromaDB where filter (see Section 2)
   b. Course content    → only query collections in enrolled_course_ids

7. Response streams back normally.
```

**Trust chain:** `X-Internal-Token` (Apache) proves the request came through the Moodle server. The sesskey + `$USER->id` cross-check in PHP proves the `user_id` in the body is the currently authenticated user. The backend never trusts `user_id` on its own.

---

## Section 1 — Video Elicitation Tool: Project Cohort Tagging

### SQLite schema migration

Add to `migration.py` as a numbered, idempotent migration:

```python
def migration_NNN_add_project_cohort_id(cursor):
    columns = get_table_columns(cursor, "projects")
    if "allowed_cohort_id" not in columns:
        cursor.execute("ALTER TABLE projects ADD COLUMN allowed_cohort_id INTEGER DEFAULT NULL")
        logger.info("Added allowed_cohort_id to projects table")
    # NULL = open mode (any authenticated Moodle user)
    # Integer = restricted to that Moodle cohort ID
```

### Pydantic model updates (`models.py`)

`ProjectCreate`, `ProjectUpdate`, and `ProjectResponse` gain:

```python
allowed_cohort_id: Optional[int] = None
```

### New API endpoint (video elicitation FastAPI)

```
GET /api/cohorts/managed
Authorization: JWT (existing verification)

Response: [{ "cohort_id": int, "cohort_name": str }, ...]

Logic:
  1. Decode JWT → extract userid
  2. Query Moodle DB:
     SELECT DISTINCT c.id, c.name
     FROM mdl_cohort c
     JOIN mdl_enrol e ON e.customint1 = c.id AND e.enrol = 'cohort'
     JOIN mdl_context ctx ON ctx.instanceid = e.courseid AND ctx.contextlevel = 50
     JOIN mdl_role_assignments ra ON ra.contextid = ctx.id AND ra.userid = :userid
     JOIN mdl_role r ON r.id = ra.roleid AND r.shortname IN ('teacher','editingteacher','manager')
  3. Returns cohorts whose enrolled courses the user teaches.
  4. Empty list = user has no teacher role in any cohort-enrolled course.
```

### UI changes (`js/app.js`)

**Project create / edit modal** — after the project name field:

- If `/api/cohorts/managed` returns cohorts: show a "Visibility" single-select:
  - *Open access — visible to all authenticated CraftPilot users* (default)
  - *[Cohort Name] only* (one option per returned cohort)
- If the endpoint returns an empty list: show an info notice (no selector):

  > *You are not currently assigned as a teacher in any cohort-enrolled course. If your work should be protected from other organisations' search results in CraftPilot, please contact **[silo_contact_email]** to have the correct role assigned to your account. Until then, your annotations will be visible to all authenticated users.*
  >
  > *(Falls back to "your Moodle administrator" if `silo_contact_email` is not configured.)*

**Dismissible notification banner** (shown once per user on tool load):

Shown when the expert has at least one project with `allowed_cohort_id = NULL` and manages at least one cohort:

> *You have **N project(s)** whose annotations are currently visible to all authenticated CraftPilot users. If this content contains proprietary knowledge, open each project's settings to assign it to a cohort.*

Includes a **"Don't show me again"** button. Dismissed state stored in `localStorage` (key: `craftpilot_silo_banner_dismissed`). The button text is deliberate — it confirms the user has understood the feature, not just closed an annoyance.

### Retroactive cohort assignment & ChromaDB re-sync

When an expert updates `allowed_cohort_id` on an existing project:

1. Backend updates `projects.allowed_cohort_id` in SQLite.
2. Backend automatically triggers a re-sync for all annotations in that project:
   - Delete existing ChromaDB documents for those annotation IDs.
   - Re-ingest with updated metadata.
3. No separate "Apply" button — re-sync is automatic on save, transparent to the expert.

Existing projects at migration time → `allowed_cohort_id = NULL` → open mode. No content is hidden retroactively.

---

## Section 2 — ChromaDB Metadata & Retrieval Filter

### Metadata fields added at ingest time

`AnnotationIngestRequest` gains `allowed_cohort_id: Optional[int] = None`.

Each ChromaDB document for a video annotation receives two new metadata fields:

```python
"cohort_id": allowed_cohort_id if allowed_cohort_id is not None else -1,
"open_access": allowed_cohort_id is None   # True = any authenticated user may retrieve
```

`cohort_id = -1` is the sentinel for open-access documents. ChromaDB metadata values must be scalars; `-1` avoids a `NULL` which ChromaDB cannot filter on.

### Retrieval `where` clause

```python
# user_cohort_ids: list[int] from SiloService (may be empty)
where = {
    "$or": [
        {"cohort_id": {"$in": user_cohort_ids}},   # restricted docs this user can see
        {"open_access": True}                        # open docs visible to all
    ]
}
# If user_cohort_ids is empty, the $in clause matches nothing; only open_access docs surface.
```

This clause is injected into every ChromaDB similarity search and MMR search in `rag_service.py` when `user_id` is present.

### Course content scoping (CourseRAGService)

No metadata filter needed — the collection name IS the scope. The backend only queries `course_{id}` collections where `id` is in `SiloService.get_enrolled_course_ids(user_id)`. Collections outside that set are never opened.

---

## Section 3 — `SiloService` (new, `services/silo_service.py`)

```python
class SiloService:
    """Resolves per-user access scope from Moodle DB. Results cached 60 s."""

    def get_allowed_cohorts(self, user_id: int) -> list[int]:
        """
        Returns cohort IDs the user belongs to.
        Query: mdl_cohort_members WHERE userid = user_id
        Used to filter video annotation retrieval.
        """

    def get_enrolled_course_ids(self, user_id: int) -> list[str]:
        """
        Returns course IDs the user is actively enrolled in.
        Query: mdl_user_enrolments JOIN mdl_enrol
               WHERE userid = user_id AND ue.status = 0 AND e.status = 0
        Used to scope CourseRAGService collection queries.
        """
```

**Cache:** a simple `dict[int, (result, timestamp)]` per method. TTL = 60 seconds. No external dependency (no Redis). Invalidated on next request after TTL expires.

**Failure mode:** if the Moodle DB is unreachable, `SiloService` raises an exception. The pipeline catches it and returns HTTP 503 — no retrieval attempted, no content leaked.

### `ChatRequest` model update (`api/models.py`)

```python
user_id: Optional[int] = None
```

### Pipeline guard (`api/routes.py`)

```python
if not body.user_id or body.user_id <= 0:
    return HTTP 403  # No retrieval attempted
```

---

## Section 4 — Admin Settings

### `local_videoelicit/settings.php` addition

After the existing JWT / backend URL / token quota settings, add a new section:

```php
$settings->add(new admin_setting_heading(
    'local_videoelicit/silo_header',
    get_string('settings_silo_header', 'local_videoelicit'),
    get_string('settings_silo_header_desc', 'local_videoelicit')
));

$settings->add(new admin_setting_configtext(
    'local_videoelicit/silo_contact_email',
    get_string('settings_silo_contact_email', 'local_videoelicit'),
    get_string('settings_silo_contact_email_desc', 'local_videoelicit'),
    '',
    PARAM_EMAIL
));
```

**Lang strings to add** (`lang/en/local_videoelicit.php`):

```php
$string['settings_silo_header']           = 'Knowledge Silo';
$string['settings_silo_header_desc']      = 'Controls who can see elicitation content in the CraftPilot RAG.';
$string['settings_silo_contact_email']    = 'Silo contact email';
$string['settings_silo_contact_email_desc'] = 'Displayed to experts who have no cohort assigned. Leave blank to show a generic "contact your administrator" message.';
```

The PHP `index.php` reads this setting via `get_config('local_videoelicit', 'silo_contact_email')` at page load and embeds it in the JWT payload alongside `userid` and `roles`. The video elicitation frontend reads it from the decoded JWT — no extra round-trip needed.

---

## Section 5 — Failure Modes & Strict Guardrails

| Scenario | Behaviour |
|---|---|
| `user_id` absent or zero in request | HTTP 403 — no retrieval |
| `user_id` in body ≠ `$USER->id` (PHP check) | HTTP 403 at `chat_proxy.php` — never reaches backend |
| Moodle DB unreachable for cohort lookup | HTTP 503 — no retrieval, error logged |
| User has no cohort memberships | Only open-access annotations returned; no restricted content |
| User has no active course enrollments | No course content returned |
| Expert updates cohort on existing project | Automatic re-sync — old ChromaDB docs deleted, re-ingested with new metadata |
| Expert removes cohort (revert to open) | `cohort_id` set to `-1`, `open_access` set to `True` after re-sync |

**The guiding rule:** when in doubt, return nothing rather than returning too much.

---

## Out of Scope (v1)

- Multi-cohort per project (one cohort per project for now; `$in` filter makes v2 straightforward)
- Admin UI for bulk cohort re-tagging of existing projects
- Audit logging of silo access decisions
- Per-annotation (not per-project) cohort assignment
