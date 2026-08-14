# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What This Is

A **Moodle 5.1 LMS** installation (`/var/www/html/public`) with three custom components:

1. **`theme/almondb`** — Bootstrap 5 theme (parent: Boost)
2. **`local/craftpilot`** — AI assistant; site-wide chat widget over a RAG backend at `127.0.0.1:8000`
3. **`local/videoelicit`** — Video annotation plugin wrapping a FastAPI backend at `localhost:8005`

External FastAPI services: `/opt/craftpilot_backend/` (`craftpilot-backend`, port 8000) and `/opt/video_elicitation_annotation_tool/` (`videoelicit-backend`, port 8005).

**Version control.** `local/craftpilot` is **not** a git checkout — it is an unversioned `rsync` target. The source of truth is the `plugin/` subfolder of `AIMoveCAOR/moodle_ai_assistant`, checked out at `/opt/craftpilot_backend`. Edit there, then sync:

```bash
rsync -a --delete --exclude=node_modules --exclude=.claude \
  /opt/craftpilot_backend/plugin/ /var/www/html/public/local/craftpilot/
```

Never run `git` from `/var/www/html/public/local/craftpilot` — there is no repo there. Moodle core and the theme are not git-tracked at all.

See `DEV_HANDOFF.md` for the everyday workflow.

## Essential Commands

### Moodle cache

```bash
php /var/www/html/admin/cli/purge_caches.php
```

Run after any change to PHP, language strings, AMD JavaScript, capabilities, DB schema, or templates. Most changes are invisible without it.

⚠️ **CLI scripts are outside the webroot.** `$CFG->dirroot` is `/var/www/html/public`, but `admin/cli/` lives at `/var/www/html/admin/cli/`. `public/admin/` exists but has **no `cli/` subdirectory**, so a relative `admin/cli/...` path from the webroot fails. Always use the absolute path.

### Craftpilot JavaScript build

Run from `/var/www/html/public/local/craftpilot`:

```bash
npx grunt babel                  # one-time build
npx grunt dev                    # file watcher (auto-recompile on save)
php /var/www/html/admin/cli/purge_caches.php
```

`node`, `npm`, and `npx` are in PATH at `/usr/bin`.

**Critical**: `amd/build/chat_interface.min.js` must be a compiled AMD `define(...)` file, not raw ES6. Raw `import` statements in the build file break all Moodle JavaScript site-wide (RequireJS fails to load `core/first`, leaving navigation hidden).

**Do not delete `amd/build/dompurify.min.js`.** It is vendored from `node_modules/dompurify` and has no `amd/src` counterpart, so `grunt babel` will not regenerate it. It sanitizes LLM output before `innerHTML`; losing it is an XSS regression. This is why `amd/build/` is committed to git — Moodle serves it directly with no deploy-time build step.

### Video elicitation backend

```bash
systemctl status videoelicit-backend
systemctl restart videoelicit-backend
journalctl -u videoelicit-backend -f      # live logs

# Dev mode with auto-reload
cd /opt/video_elicitation_annotation_tool
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8005
```

### RequireJS cache (when JS changes aren't loading)

```bash
rm -f /var/www/moodledata/localcache/requirejs/*
```

### Craftpilot checks (run after any plugin change)

```bash
# PHP syntax across the plugin
find /var/www/html/public/local/craftpilot -name '*.php' -not -path '*/node_modules/*' \
  -exec php -l {} \; | grep -v 'No syntax errors'

# Backend health
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/api/health   # expect 200
```

The RAG evaluation harness is `test_bench.php` (admin UI), with fixed questions in `classes/test_bench_questions.php`.

## Architecture

### Theme SCSS pipeline

Moodle compiles SCSS on demand — there is **no manual build step** for CSS.

- Edit SCSS in `theme/almondb/scss/almondb/` (components) or `scss/preset/default.scss`
- `lib.php::theme_almondb_get_precompiled_css()` serves the pre-compiled `style/moodle.css`
- `lib.php::theme_almondb_get_pre_scss()` injects color variables from theme admin settings (`$primary`, `$hynavbar`, `$hythemecolor`)
- To recompile: purge caches — Moodle recompiles on next page load

`primary-navigation` uses `.moremenu` with `opacity: 0` by default; JS adds `.observed` to reveal it. If AMD JS fails, navigation stays invisible.

### Craftpilot AMD JavaScript

- Source: `amd/src/chat_interface.js` (ES6 with `import`)
- Build: `amd/build/chat_interface.min.js` (AMD `define()` — what Moodle loads)

Grunt runs Babel with `@babel/preset-env` targeting `modules: 'amd'`. Verify the build output is `define(...)`, not `import`.

### Video elicitation data flow

```
Browser → local/videoelicit/index.php (PHP mints a JWT, embeds it in the iframe URL)
       → iframe /videoelicit-ui/?token=<JWT>  (Apache ProxyPass → FastAPI:8005)
       → FastAPI validates JWT (shared secret: Moodle config ↔ .env MOODLE_JWT_SECRET)
```

⚠️ **`api_proxy.php` is NOT in this path.** Browser traffic reaches the backend through Apache's `ProxyPass` rules in `/etc/httpd/conf.d/moodle-ssl.conf`, not through PHP. Treat it as dormant: do not cite it when explaining how the plugin works, and do not add calls to it without first confirming it is wired up and its `backend_url` setting is correct.

Video streaming bypasses the backend entirely: `Browser → stream.php → Moodle File API → 206 Partial Content`. Do not break the HTTP Range logic in `stream.php` — seeking depends on it. `stream.php` and `stream_ticket.php` both require `jwt_helper::verify_token()`; if it goes missing, streaming for non-session clients dies with a fatal error.

### Moodle AMD module loading

Moodle bundles all AMD modules into a single cached file via `requirejs.php`, cached in `/var/www/moodledata/localcache/requirejs/`. If any module in the bundle contains raw ES6 `import` syntax, RequireJS throws `SyntaxError` and all JS on the site breaks.

## Craftpilot plugin (`local/craftpilot`)

Single LLM provider: **Infomaniak AI Tools** (OpenAI-compatible), called by the Python backend — not by Moodle. `classes/external/get_user_credentials.php` is a no-op session check.

- `chat_proxy.php` — validates the Moodle session and `sesskey`, then issues a **307 redirect** to `/craftpilot-api/chat`. The browser re-POSTs directly to Apache's `ProxyPass`, bypassing PHP-FPM (which buffered the stream and caused 504s). Apache injects `X-Internal-Token` for that path.
- `classes/backend_client.php` — server-side HTTP client; reads the token via `get_config('local_craftpilot', 'internal_api_token')`
- `classes/observer.php` — event hooks that trigger course-content ingestion into the RAG index
- `amd/src/chat_interface.js` — streaming chat UI; markdown via `marked.js`, sanitized with DOMPurify before `innerHTML`
- RAG backend: `http://127.0.0.1:8000/api/chat`
- Tables: `mdl_local_craftpilot_conv`, `mdl_local_craftpilot_msg`, `mdl_local_craftpilot_cm_index`

## Videoelicit plugin (`local/videoelicit`)

**Four capabilities**: `view`, `annotate`, `manage`, `viewall` — always check before file operations.

**File storage backends** (`source_type` in `local_videoelicit_videos`):

- `uploaded` — the only value present in production
- `local` — Moodle File API (`get_file_storage()->get_file()`)
- `webdav` — OwnCloud/Nextcloud via `external_url`; credentials in plugin settings

The WebDAV path is configured but unused; if you re-enable it, provision a service account rather than a personal login. Verify what is actually in use:

```bash
php -r 'define("CLI_SCRIPT",1);require("/var/www/html/config.php");global $DB;
  foreach($DB->get_records_sql("SELECT source_type, COUNT(*) c
    FROM {local_videoelicit_videos} GROUP BY source_type") as $r)
    echo "  $r->source_type = $r->c\n";'
```

**Database changes**: edit `db/install.xml` (XMLDB, never raw SQL), increment `version.php`, add an upgrade step in `db/upgrade.php` guarded by `if ($oldversion < YYYYMMDDXX)`, then visit Site Admin → Notifications. **Removing a scheduled task also needs a `version.php` bump** — otherwise Moodle never re-reads `db/tasks.php` and the old row lingers in `mdl_task_scheduled`.

**API calls from JS** reach the backend via Apache's `ProxyPass` rules, not via PHP — see the `api_proxy.php` note above.

## Moodle PHP conventions

- `defined('MOODLE_INTERNAL') || die();` in all included files
- Use the `$CFG`, `$DB`, `$USER`, `$PAGE`, `$OUTPUT` globals
- Input: `required_param()` / `optional_param()` with `PARAM_*` — never `$_GET`/`$_POST`
- Database: `$DB->get_record()`, `$DB->insert_record()`, etc. — never raw SQL
- Always `require_login()` then `require_capability()` in entry points
- AMD: `define(['core/ajax', ...], function(Ajax, ...) { return { init: ... }; })`

### `page_init` does not exist for local plugins

`theme_NAME_page_init(moodle_page $page)` is a **theme-only** callback. Moodle never invokes `local_NAME_page_init()`. `$PAGE->requires->js_call_amd()` placed there silently does nothing.

**Rule**: in a `local_` plugin, queue AMD calls from `local_NAME_before_footer()`:

```php
function local_myplugin_before_footer(): string {
    global $OUTPUT, $PAGE;
    $PAGE->requires->js_call_amd('local_myplugin/my_module', 'init', [...]);
    return $OUTPUT->render_from_template('local_myplugin/widget', [...]);
}
```

`before_footer` is supported in Moodle 5.x via the `core\hook\output\before_footer_html_generation` shim, and fires before `$OUTPUT->footer()` writes the final `<script>` block, so requirements added there are picked up.

## Key file locations

| What | Where |
|------|-------|
| Moodle config (DB, wwwroot) | `/var/www/html/config.php` |
| Moodle CLI scripts | `/var/www/html/admin/cli/` — outside the webroot |
| Theme SCSS components | `theme/almondb/scss/almondb/` |
| Theme pre-compiled CSS | `theme/almondb/style/moodle.css` |
| Craftpilot plugin (live, unversioned) | `local/craftpilot/` |
| Craftpilot JS source | `local/craftpilot/amd/src/chat_interface.js` |
| Craftpilot JS build | `local/craftpilot/amd/build/chat_interface.min.js` |
| Craftpilot plugin README | `local/craftpilot/README.md` |
| Craftpilot backend + plugin source | `/opt/craftpilot_backend/` — repo `AIMoveCAOR/moodle_ai_assistant` |
| Videoelicit plugin PHP | `local/videoelicit/` |
| Videoelicit backend | `/opt/video_elicitation_annotation_tool/` — repo `AIMoveCAOR/video_elicitation_annotation_tool` |
| FastAPI service entry | `/opt/video_elicitation_annotation_tool/backend/main.py` |
| FastAPI config/env | `/opt/video_elicitation_annotation_tool/.env` |
| Apache vhost | `/etc/httpd/conf.d/moodle-ssl.conf` |
| RequireJS bundle cache | `/var/www/moodledata/localcache/requirejs/` |

## Database access (phpMyAdmin)

Reachable at `https://localhost/phpmyadmin`, **localhost only** — use a browser running on the VM.

Server `localhost`, database `moodle`, user `moodleuser`. The password is in `/var/www/html/config.php` (`$CFG->dbpass`); it is not recorded in any documentation file.
