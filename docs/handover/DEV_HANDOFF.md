# Dev handoff — aimove (Moodle + AI tools)

Read this top to bottom once. It's meant to get you productive without
anyone walking you through it live. Everything runs on one production
server; there is no staging step and no PR review gate — just make the
change carefully, verify it in place, and push it to the git repo it
came from.

---

## 1. What this is, in one paragraph

A Moodle course platform (`aimove.minesparis.psl.eu`) with two
custom-built add-ons: an AI chat/RAG assistant ("craftpilot") that
answers student questions from course content, and a video annotation
tool ("videoelicit") used for craft-procedure elicitation interviews.
Both add-ons have two halves: a Moodle plugin (PHP, lives inside
Moodle) and a separate Python backend service (FastAPI) that the plugin
talks to.

## 2. Starting a work session

SSH in as `root`, then drop into the sandboxed `claude-runner` user
**before** starting Claude Code:

```bash
sudo -u claude-runner -i
claude
```

This is not optional. The `claude-runner` account has **no sudo** and
physically cannot read `.env` files (kernel-level ACL denial), so
`INFOMANIAK_API_KEY`, `MOODLE_DB_PASSWORD`, `INTERNAL_API_TOKEN`, and
every other secret on the box are unreadable to the agent. Running
`claude` directly as root defeats the cage — a misbehaving model would
then be able to leak any secret into a transcript, a commit, or a tool
call.

For manual work without an LLM, stay as root — you'll need the
privileges for `systemctl restart`, `journalctl`, MariaDB access, and
cache purges. Just be mindful about what you paste anywhere.

## 3. What's actually running

Four things, all managed by `systemctl`:

| Service | What it is | Port | Runs as | Code lives at |
|---|---|---|---|---|
| `httpd` | Apache — front door, TLS, reverse proxy | 443 (public), 80 | apache | config in `/etc/httpd/conf.d/` |
| `mariadb` | The database (single DB, `moodle`) | local only | mysql | — |
| `craftpilot-backend` | AI assistant backend (FastAPI/uvicorn, conda env) | 8000 | root | `/opt/craftpilot_backend` |
| `videoelicit-backend` | Video annotation backend (FastAPI, its own venv) | 8005 | apache | `/opt/video_elicitation_annotation_tool/backend` |

Moodle itself isn't a systemd service — just PHP files that `httpd`
serves via `mod_php`/`php-fpm`. Moodle's code is at `/var/www/html` (the
webroot is `/var/www/html/public/`, but `admin/cli/*.php` and
`config.php` live one level up at `/var/www/html/` directly — normal for
this Moodle version).

**How a request flows:** browser → Apache (443) → either Moodle's PHP
directly, or Apache proxies specific URL paths to one of the two
backends (`/craftpilot-api/` → :8000, `/videoelicit-ui/` and `/api/` →
:8005). The exact proxy rules are in
`/etc/httpd/conf.d/moodle-ssl.conf`.

**Health & logs:**
```bash
systemctl status httpd mariadb craftpilot-backend videoelicit-backend
tail -100 /tmp/craftpilot_backend.log
journalctl -u videoelicit-backend -n 100 --no-pager
journalctl -u httpd -n 100 --no-pager
```

## 4. Where the code lives — one repo per app, plugin as a subfolder

Both apps now follow the **same layout**: one repo holds the Python
backend at its root, and the Moodle plugin lives inside it as a
subfolder that gets `rsync`'d out to the live Moodle path (never
symlinked, never a git checkout of its own).

| App | Repo | Backend path | Plugin subfolder | Live plugin path |
|---|---|---|---|---|
| AI assistant | `github.com/Somekindofa/moodle_ai_assistant` | `/opt/craftpilot_backend` | `plugin/` | `/var/www/html/public/local/craftpilot` |
| Video tool | `github.com/Somekindofa/video_elicitation_annotation_tool` | `/opt/video_elicitation_annotation_tool` | `local_videoelicit/` | `/var/www/html/public/local/videoelicit` |

Practically, for either app: edit inside the repo's plugin subfolder,
sync into place, purge caches, commit.

```bash
# craftpilot
rsync -a --delete --exclude=node_modules --exclude=.claude \
  /opt/craftpilot_backend/plugin/ /var/www/html/public/local/craftpilot/

# videoelicit
rsync -a --delete \
  /opt/video_elicitation_annotation_tool/local_videoelicit/ \
  /var/www/html/public/local/videoelicit/

php /var/www/html/admin/cli/purge_caches.php
```

The backend half of either app just needs a service restart to pick up
code changes — no sync step, since `/opt/*_backend` (or
`/opt/video_elicitation_annotation_tool`) *is* the checkout Python runs
from directly.

Moodle core (`/var/www/html`, minus the `local/` plugin folders) is
**not** git-tracked — vendored codebase, updated separately.

⚠️ **This wasn't always one repo for craftpilot.** The plugin was split
into its own repo (`moodle-local-craftpilot`) on 2026-08-03, then merged
back on 2026-08-06 — that repo only ever had 3 commits and was never
independently released, so the split was pure overhead. It's now
archived on GitHub, not deleted; its commits (including a real bugfix)
are still fully reachable — `git log plugin-merge-2026-08-06 --oneline`
in `moodle_ai_assistant` (that tag marks its pre-merge tip; plain
`git log plugin/` won't show them, see `/opt/craftpilot_backend/CLAUDE.md`
for why). If you find an old doc, PR, or note referencing
`moodle-local-craftpilot` as a separate clone target, it's stale.

## 5. Everyday feature work — what to touch and what to run after

Direct edits on the live server. There is no staging; be deliberate,
and always verify after each change.

### Craftpilot Moodle plugin — PHP, templates, lang strings

Edit source at `/opt/craftpilot_backend/plugin/`, **not** the live path
directly — the live path is an unversioned sync target (see §4). After
any `.php`, `.mustache`, `db/*.xml`, `lang/*.php`, or capability/DB
change:

```bash
rsync -a --delete --exclude=node_modules --exclude=.claude \
  /opt/craftpilot_backend/plugin/ /var/www/html/public/local/craftpilot/
php /var/www/html/admin/cli/purge_caches.php
```

For DB schema changes: edit `plugin/db/install.xml` (XMLDB format,
never raw SQL), increment `plugin/version.php`, add an upgrade step in
`plugin/db/upgrade.php` with an `if ($oldversion < YYYYMMDDXX)` guard,
sync, then visit Site Admin → Notifications to run it.

### Craftpilot Moodle plugin — JavaScript

Source: `plugin/amd/src/chat_interface.js` (ES6 with `import`).
Build:  `plugin/amd/build/chat_interface.min.js` (AMD `define()`, what
Moodle loads).

```bash
cd /opt/craftpilot_backend/plugin
npx grunt babel                            # one-shot build
# OR
npx grunt dev                              # file watcher (auto-recompile)
rsync -a --delete --exclude=node_modules --exclude=.claude \
  /opt/craftpilot_backend/plugin/ /var/www/html/public/local/craftpilot/
php /var/www/html/admin/cli/purge_caches.php
rm -f /var/www/moodledata/localcache/requirejs/*     # if the change won't load
```

**Never edit `amd/build/chat_interface.min.js` directly.** It must be a
compiled AMD `define(...)` file. Any raw ES6 `import` slipping into the
build file breaks RequireJS site-wide — `core/first` fails to load, and
the primary navigation goes invisible on every page. If nav ever
disappears everywhere, that's the symptom; fix the build.

**`amd/build/dompurify.min.js` is the exception — do NOT delete it.**
It's vendored from `node_modules/dompurify` and has no `amd/src`
counterpart, so `grunt babel` will not regenerate it. It sanitizes LLM
output before `innerHTML`; losing it is an XSS regression. This is also
why `plugin/amd/build/` is committed to git rather than gitignored.

### Craftpilot Python backend

Path: `/opt/craftpilot_backend/`

```bash
systemctl restart craftpilot-backend
tail -f /tmp/craftpilot_backend.log
```

Conda env: `moodle_backend` at `/root/miniconda3/envs/moodle_backend`.
If you add a dependency, install with that env's pip and update
`environment.yml` — but see §7's rough-edge note about langchain pins
before ever rebuilding the env from scratch.

Architecture, RAG pipeline internals, and per-service design notes:
`/opt/craftpilot_backend/CLAUDE.md`. Known failure modes:
`/opt/craftpilot_backend/TROUBLESHOOT.md`.

### Debugging a RAG call — LangSmith traces

Every `/api/chat` request is traced end-to-end (`retrieve_initial` →
`refine_query_prf` → `retrieve_final_dual` → `rerank` → `generate`) via
LangSmith, **EU West region** (`https://eu.api.smith.langchain.com`),
project `Craftpilot`. This is the tool for questions like *"why didn't
the model answer?"* or *"why didn't the reranker get the docs the
earlier retrieval stage found?"* — each trace breaks out every pipeline
step with its actual inputs and outputs, not just the final response.

The account credentials are **dev-only, intentionally not written
anywhere in docs or git** — they live solely in
`LANGSMITH_API_KEY` / `LANGSMITH_ENDPOINT` / `LANGSMITH_PROJECT` in
`/opt/craftpilot_backend/.env` (root-owned, mode 600, ACL-blocked from
`claude-runner` — see §6). Ask whoever holds root for the value; never
paste it into a transcript, commit, or doc.

**Migrated 2026-08-14.** Tracing was **re-pointed to the `aimove.caor`
LangSmith account, not disabled** — `LANGSMITH_TRACING` stays `true`.
Verified end-to-end: runs `stream_response` and `ChatOpenAI` landed in
project `Craftpilot`.

⚠️ **The key must be a *workspace*-scoped Service Key, not an
org-scoped one.** An org-scoped key authenticates fine but is rejected
by the trace endpoints, and the failure is close to invisible:

- `/sessions` and `/runs` return a bare `403 {"detail":"Forbidden"}` —
  **identical to the response for a completely invalid key.**
- The real reason is only visible on `/auth`, which returns
  `{"error":"org_scoped_key_requires_workspace"}`.
- LangSmith sends traces on a **background thread and swallows the
  result**, so chat keeps working perfectly with tracing 100% dead.
  Nothing appears in the app or in `journalctl`.

So **"chat still works" is not evidence that tracing works.** The only
valid check is that a run actually appears. This probe prints project
names but never the key:

```bash
/root/miniconda3/envs/moodle_backend/bin/python -c "
import dotenv; dotenv.load_dotenv('/opt/craftpilot_backend/.env')
from langsmith import Client
print([p.name for p in Client().list_projects(limit=10)])"
```

(The alternative fix — keeping an org-scoped key and adding
`LANGSMITH_WORKSPACE_ID`, which the SDK sends as `X-Tenant-Id` — works
too, but was rejected: it splits the scope across two variables, and a
future reader can delete the second one without knowing why it exists.)

### Videoelicit Moodle plugin

Path (source, git-tracked):
`/opt/video_elicitation_annotation_tool/local_videoelicit/`

Path (live, served by Apache):
`/var/www/html/public/local/videoelicit/`

Edit source, sync into place, purge caches, commit:

```bash
rsync -a --delete \
  /opt/video_elicitation_annotation_tool/local_videoelicit/ \
  /var/www/html/public/local/videoelicit/
php /var/www/html/admin/cli/purge_caches.php
```

**How browser calls actually reach the backend** — not through PHP.
`index.php` mints a JWT and embeds it in the iframe URL; the SPA at
`/videoelicit-ui/` is then proxied straight to `127.0.0.1:8005` by Apache's
`ProxyPass` rules. Video byte-range requests go to `stream.php`, which
authenticates via Moodle session, then an opaque `?ticket=`, then a legacy
JWT — in that order.

⚠️ **`api_proxy.php` is dormant, despite what older docs say.** No `.js`,
`.php`, or `.mustache` file calls it — only documentation mentions it, and
its `backend_url` pointed at a port nothing listens on until 2026-08-07.
Don't cite it when explaining the architecture, and don't route new code
through it without first confirming it works end to end.

### Videoelicit Python backend

Path: `/opt/video_elicitation_annotation_tool/`

```bash
systemctl restart videoelicit-backend
journalctl -u videoelicit-backend -f
```

Iterative dev (auto-reload):
```bash
cd /opt/video_elicitation_annotation_tool
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8005
```

Architecture and design notes:
`/opt/video_elicitation_annotation_tool/CLAUDE.md`.

### Theme (almondb)

Path: `/var/www/html/public/theme/almondb/`

SCSS lives in `scss/almondb/` (components) and `scss/preset/default.scss`.
**No manual build step** — Moodle recompiles SCSS on next page load
after a cache purge:

```bash
php /var/www/html/admin/cli/purge_caches.php
```

If your SCSS change doesn't appear, the pre-compiled
`style/moodle.css` cache (served by
`lib.php::theme_almondb_get_precompiled_css()` for performance) is
stale — a cache purge regenerates it.

### Apache config

Paths:
- `/etc/httpd/conf.d/moodle-ssl.conf` — vhost, proxy rules, headers
- `/etc/httpd/conf/httpd.conf` — server scope, dotfile deny block

```bash
httpd -t                          # syntax check first, always
systemctl reload httpd
```

Two things Apache carries that you MUST NOT break:
- The `X-Internal-Token` header injected on `<Location /craftpilot-api/>`.
  It's how the backend authenticates browser-followed 307 redirects out
  of `chat_proxy.php`. If it goes missing, chat returns 401.
- The dotfile deny block (server scope). See §6.

## 6. Secrets — what has to match where

Three secrets must be internally consistent across multiple files:

1. **JWT secret** — videoelicit's Moodle plugin signs tokens, the
   Python backend verifies them. Lives in: Moodle admin setting
   `local_videoelicit/jwt_secret`, and the backend's `.env` as
   `MOODLE_JWT_SECRET`. If the Moodle setting is ever unset,
   `jwt_helper.php` silently falls back to an insecure default —
   verify the setting is populated, don't assume.
2. **Internal API token** — gates calls between Apache and craftpilot's
   backend. Lives in: `/etc/httpd/conf.d/moodle-ssl.conf` (header
   `X-Internal-Token`), craftpilot's `.env` as `INTERNAL_API_TOKEN`,
   and the Moodle DB config `local_craftpilot/internal_api_token`.
3. **Database password** — the `moodleuser` MariaDB account. Lives in:
   `config.php` (`$CFG->dbpass`) and both apps' `.env` as
   `MOODLE_DB_PASSWORD`.

`.env` files are root-owned, mode 600, and ACL-blocked from
`claude-runner`. If you're inside a Claude session and need a value,
exit back to root (leave the `sudo -u claude-runner -i` shell) rather
than loosening the perms.

### 6b. Full credential inventory — who owns what

The three above must *match across files*. This table is the wider set: every
credential the stack uses, who owns it after the August 2026 handoff, and how
to rotate it. **Names and locations only — never write a value here.**

| Credential | Owned by | Lives in | Rotating it |
|---|---|---|---|
| `INFOMANIAK_API_KEY` + `INFOMANIAK_PRODUCT_ID` | Infomaniak **organisation** account | **both** `.env` files | See 6c — not a simple key swap |
| `LANGSMITH_API_KEY` | ✅ **`aimove.caor` LangSmith account** (migrated 2026-08-14) | craftpilot `.env` | Tracing only; service unaffected if it breaks. Must be a **workspace-scoped** Service Key — an org-scoped key fails silently, see §5 *Debugging a RAG call*. |
| GitHub push auth | per-repo **deploy keys** on the server | `~/.ssh/` + remotes | Generate a new key, add it as a repo Deploy key with write access, swap the remote, verify a push, *then* remove the old one |
| `MOODLE_JWT_SECRET` | server-local | Moodle setting `local_videoelicit/jwt_secret` **and** videoelicit `.env` | Generate 64 random chars, set in both, restart `videoelicit-backend`. If the Moodle setting is ever blank, `jwt_helper.php` silently falls back to an insecure default — check it is populated. |
| `INTERNAL_API_TOKEN` | server-local | Apache `moodle-ssl.conf`, craftpilot `.env`, Moodle DB `local_craftpilot/internal_api_token` | All **three** together, then `httpd -t && systemctl reload httpd` and restart the backend. Miss one and chat 401s. |
| `MOODLE_DB_PASSWORD` | server-local | `config.php` (`$CFG->dbpass`) + **both** `.env` files | Change in MariaDB first, then all three files, then restart both backends |
| WebDAV service account | *(cleared — see below)* | Moodle settings `local_videoelicit/webdav_*` | Unused: all production videos are `source_type = uploaded`. If you re-enable it, provision a real **service account**, never a person's login. |

**If a value is lost:** everything except Infomaniak and LangSmith is
server-local and can simply be regenerated using the rows above — there is no
external party to ask. For those two, the account holder is the org
administrator; there is no recovery path through this server.

### 6c. Migrating the Infomaniak account

This is **not** just an API-key rotation, for two reasons:

- `INFOMANIAK_PRODUCT_ID` is interpolated into the API **URL path** —
  `https://api.infomaniak.com/2/ai/{product_id}/openai/v1`
  (`services/rag_service.py`, `services/summary_service.py`) and
  `.../cohere/v2/rerank` (`services/reranker_service.py`). A different account
  means a different product, so the ID changes too.
- The key is duplicated in **two** `.env` files. Miss the videoelicit one and
  transcription plus AI tagging break while chat keeps working — a confusing
  partial failure that looks like a video bug.

The real risk is **model availability**, not credentials. Verify before
cutting over, using the new key:

```bash
curl -s -H "Authorization: Bearer $NEW_KEY" \
  "https://api.infomaniak.com/2/ai/$NEW_PRODUCT_ID/openai/v1/models"
```
All three must be present: `swiss-ai/Apertus-70B-Instruct-2509` (chat, tagging,
advisory), `whisper` (transcription), and the Qwen reranker (retrieval).

Then update both `.env` files, `systemctl restart craftpilot-backend
videoelicit-backend`, and run the checks in `DEPENDENCIES.md` §0 and §6.

**Keep the old key live until every check passes.** Then revoke it and
re-run one chat request expecting it to *fail closed* — that is the only proof
the new credentials are actually in use rather than a cached client.

**Dotfiles are blocked at the web server.** `/etc/httpd/conf/httpd.conf`
carries a server-scope deny block for `.git`, `.env`, `.claude`,
`.svn`, `.hg`, `.npmrc`, `.pypirc`. Do not remove it, and do not move
it inside a `<VirtualHost>`: three vhosts serve `/var/www/html/public`
(ports 80 and 443), so a vhost-scoped rule would leave holes. It must
also stay a `<DirectoryMatch>`, not just a `<FilesMatch>` — `.git/index`
plus loose objects under `.git/objects/` are enough to reconstruct a
full source tree, so blocking `.git/config` alone achieves nothing.

Verify after any Apache change — both must return 403:
```bash
for s in http https; do curl -s -o /dev/null -w "$s -> %{http_code}\n" -k \
  "$s://aimove.minesparis.psl.eu/local/craftpilot/.git/config"; done
```

## 7. History worth knowing (so weird things don't look mysterious)

- **`mod_craftpilot` is dead.** An earlier version of the assistant was
  a Moodle *activity module* at `/var/www/html/public/mod/craftpilot`
  (repo `moodle-plugin-ai`, now archived). Retired March 2026 with zero
  course instances ever created. The live plugin is `local/craftpilot`.
  Confirm which is which: open the browser Network tab, find any AJAX
  request, and read `methodname`. Live calls `local_craftpilot_*`; dead
  used `mod_craftpilot_*`. Editing the wrong tree has already cost one
  full debugging session.
- **`mdl_craftpilot*` tables (no `local_` prefix) belong to the dead
  plugin and are all empty.** Don't read or write them.
- **`.git` directories used to be downloadable over the public web.**
  Until 2026-08-03,
  `https://aimove.minesparis.psl.eu/mod/craftpilot/.git/config` returned
  HTTP 200 to anyone. Apache's stock `<Files ".ht*">` rule does not
  cover `.git`, `.env`, or `.claude`. The server-scope deny block in
  `/etc/httpd/conf/httpd.conf` is what closed it — see §6.
- **Both Moodle plugins had drifted from git before 2026-07-30**:
  `local_craftpilot` didn't exist in *any* git repo (only ever lived on
  this live server), and `local_videoelicit` had several files edited
  directly on the server and never committed. Both were reconciled into
  their repos. If you ever find live server code that doesn't match git
  again, that means someone hand-edited past the workflow — copy the
  live version into the matching repo, commit it, then continue from
  there. Don't silently overwrite the live version with an older git
  version.
- **Craftpilot's plugin has now lived in three places, in order:** a
  bare live-server-only copy with no git repo at all (until 2026-07-30)
  → a `local_craftpilot/` subfolder of the backend repo, added by PR #19
  (2026-07-30) → its own standalone repo, `moodle-local-craftpilot`
  (2026-08-03) → merged back into the backend repo as `plugin/`
  (2026-08-06), because the standalone-repo split turned out to add a
  two-PR-per-feature tax with no real benefit (only 3 commits ever made
  there). The `local_craftpilot/` subfolder from the second step was
  dead weight for the whole time in between and was deleted in the same
  session as the `plugin/` merge. If you find code, docs, or a PR
  referencing any earlier of these, it's stale — `plugin/` in
  `moodle_ai_assistant` is now the one and only source of truth.

## 8. Known rough edges (not urgent, just worth knowing)

- `craftpilot_backend/environment.yml` has no version pins on the
  langchain/langgraph package family. If you ever need to rebuild that
  conda env from scratch, don't trust a fresh `conda env create` to
  give you a working set of versions — snapshot what's actually
  installed first (`conda list -n moodle_backend | grep -i langchain`)
  and pin to those, since newer langchain releases have breaking API
  changes the current code depends on.
- `craftpilot_backend`'s systemd unit expects its conda env at
  `/root/miniconda3/envs/moodle_backend`. Don't be alarmed if
  `environment.yml`'s embedded `prefix:` metadata line points somewhere
  else — that's stale export metadata, not a real path anyone relies
  on.
- `videoelicit-backend`'s systemd unit runs
  `/usr/local/bin/videoelicit-start.sh` — a script that lives outside
  both app repos (not covered by any file sync). If it's ever missing
  on a fresh box, copy it verbatim from a known-working server rather
  than reconstructing it from scratch.
- The `primary-navigation` in the almondb theme uses `.moremenu` with
  `opacity: 0` by default; JS adds `.observed` to reveal it. If any AMD
  module in the site-wide bundle fails to load (typical cause: raw ES6
  slipped into `amd/build/`), primary nav goes invisible everywhere on
  the site. That's the symptom, not the bug — fix the JS build.

## 9. Where the deeper docs live

- `/opt/craftpilot_backend/CLAUDE.md` — AI backend architecture (RAG
  pipeline, PRF strategy, chunking, per-course collections).
- `/opt/craftpilot_backend/TROUBLESHOOT.md` — known backend failure
  modes and fixes.
- `/opt/video_elicitation_annotation_tool/CLAUDE.md` — videoelicit
  architecture notes.
- `/var/www/html/public/CLAUDE.md` — Moodle-side conventions, essential
  commands, key file locations.
- `/var/www/html/public/local/craftpilot/README.md` — craftpilot Moodle
  plugin setup, build, and why `amd/build/` is committed.
