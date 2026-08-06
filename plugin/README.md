# CraftPilot — Moodle Local Plugin (`local_craftpilot`)

AI teaching assistant for vocational-craft apprentices. Provides a site-wide chat
widget backed by a RAG pipeline over annotated expert video and course documents.

| | |
|---|---|
| Component | `local_craftpilot` |
| Version | `2026032600` (release `1.0.0`) |
| Requires | Moodle 4.3+ (`2023100900`) |
| Maturity | `MATURITY_STABLE` |

> **Supersedes `mod_craftpilot`.** The old activity-module implementation
> ([moodle-plugin-ai](https://github.com/Somekindofa/moodle-plugin-ai)) is archived and
> must not be used. See [Migration history](#migration-history).

---

## ⚠️ The install path is load-bearing

This plugin **must** live at `<moodleroot>/public/local/craftpilot`.

Moodle binds the component name to the directory path. `local/craftpilot` →
`local_craftpilot`, which in turn determines:

- DB table prefix — `mdl_local_craftpilot_conv`, `mdl_local_craftpilot_msg`
- Language string namespace — `get_string('x', 'local_craftpilot')`
- Capability names — `local/craftpilot:*`
- Web service method names — `local_craftpilot_*`

Renaming or relocating the directory breaks all four at once. This is precisely
why the `mod` → `local` change required a rewrite rather than a `git mv`.

---

## Setup

```bash
cd <moodleroot>/public/local/craftpilot
npm install            # restores node_modules (~41MB, not committed)
```

Then in Moodle: **Site administration → Plugins → Local plugins → CraftPilot**,
and set `internal_api_token`. It must match `INTERNAL_API_TOKEN` in the backend's
`.env`. Never hardcode it in source.

## Build

JS sources live in `amd/src/`; Moodle serves the compiled `amd/build/` directly.

```bash
npx grunt babel
php <moodleroot>/admin/cli/purge_caches.php
```

> **Do not delete `amd/build/dompurify.min.js`.** It is vendored from
> `node_modules/dompurify` and has no `amd/src` counterpart — `grunt babel` only
> compiles files that exist in `amd/src/`, so it will not be regenerated. It
> provides XSS sanitization for LLM output; losing it is a security regression.

## Layout

| Path | Purpose |
|---|---|
| `classes/backend_client.php` | HTTP client for the FastAPI backend |
| `chat_proxy.php` | Validates session + sesskey, then 307-redirects to `/craftpilot-api/chat` |
| `video_proxy.php` | Streams video through Moodle auth |
| `classes/observer.php` | Moodle event hooks that trigger course-content ingestion |
| `classes/course_content_extractor.php` | Pulls Page/Label/Resource content for indexing |
| `db/services.php` | Web service (AJAX) definitions |
| `db/install.xml` | Schema — DDL only, no data |
| `db/upgrade.php` | Version migrations |
| `amd/src/chat_interface.js` | Chat UI |
| `test_bench.php` + `classes/test_bench_questions.php` | RAG evaluation harness (synthetic questions) |
| `cli/migrate_from_mod.php` | One-shot migration from the retired `mod_craftpilot` |

## Backend dependency

A FastAPI service on `127.0.0.1:8000`, reverse-proxied at `/craftpilot-api/`.
Apache injects the `X-Internal-Token` header for that path. The backend lives in
a separate repository.

---

## Data & privacy

User conversations are stored **only** in the Moodle database
(`local_craftpilot_conv`, `local_craftpilot_msg`). They are never written to disk
in this tree and are never committed.

`db/install.xml` is schema-only — it defines table shape, not rows. `.gitignore`
blocks `*.sql`, `*.dump`, and `*.csv` as a guard rail. Keep it that way.

## Secret scanning

The repo ships a gitleaks pre-commit hook. Git does not install hooks
automatically on clone, so **run this once after cloning**:

```bash
git config core.hooksPath .githooks
```

Requires the `gitleaks` binary (`/usr/local/bin/gitleaks` on the current host).

---

## Migration history

CraftPilot began as `mod_craftpilot`, a Moodle **activity module** instantiated
per course. It was re-architected into a **local** plugin because the assistant
answers questions across the whole site rather than within a single course — a
per-course activity was the wrong container.

The old plugin was retired in March 2026 with zero course instances ever created.
Its repository is archived and read-only.
