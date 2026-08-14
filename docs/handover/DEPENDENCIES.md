# Dependencies — what these plugins need that isn't in the repo

Both Moodle plugins are only *half* of their application. The other half is a
Python service, an Apache proxy rule, a database setting, and a third-party
API. Cloning the repo gets you none of that.

This file answers one question: **what must exist outside the repos for the
plugins to work, and is it actually there right now?**

Every row has a command that proves it. Run the whole file top to bottom after
a reboot, after a migration, or when something is broken and you don't yet know
what. Anything that doesn't print the expected result is your bug.

Last verified: **2026-08-07**.

---

## 0. One-shot health sweep

Run this first. If everything passes, skip to whatever you were actually doing.

```bash
systemctl is-active httpd mariadb craftpilot-backend videoelicit-backend
curl -s -o /dev/null -w 'craftpilot  :8000 -> %{http_code}\n' http://127.0.0.1:8000/api/health
curl -s -o /dev/null -w 'videoelicit :8005 -> %{http_code}\n' http://127.0.0.1:8005/api/health
```
Expect four `active` lines and two `200`s.

---

## 1. Runtime services

All four are `systemd`-managed. Moodle itself is not a service — it is PHP files
Apache serves.

| Service | Port | Runs as | Code |
|---|---|---|---|
| `httpd` | 443 / 80 | apache | `/etc/httpd/conf.d/` |
| `mariadb` | local socket | mysql | — |
| `craftpilot-backend` | 8000 | **root** | `/opt/craftpilot_backend` |
| `videoelicit-backend` | 8005 | apache | `/opt/video_elicitation_annotation_tool/backend` |

```bash
systemctl status httpd mariadb craftpilot-backend videoelicit-backend --no-pager
ss -tlnp | grep -E ':(8000|8005)'      # expect exactly these two; nothing on 8006
```

> **The port is 8005, not 8006.** Docs, the plugin default, and the stored
> Moodle setting all said `8006` until 2026-08-07 — a value nothing has ever
> listened on. If you see 8006 anywhere, it is a bug, not a second service.

---

## 2. Interpreters and virtual environments

Neither backend runs on the system Python. Both paths are baked into their
systemd units, so a moved or rebuilt env breaks startup with no other warning.

| What | Path | Used by |
|---|---|---|
| conda env `moodle_backend` | `/root/miniconda3/envs/moodle_backend` | craftpilot-backend |
| venv | `/opt/video_elicitation_annotation_tool/.venv` | videoelicit-backend |
| node / npm / npx | `/usr/bin` (v22) | craftpilot AMD build only |

```bash
ls -d /root/miniconda3/envs/moodle_backend /opt/video_elicitation_annotation_tool/.venv
command -v node npx
```

⚠️ **Do not rebuild the conda env from `environment.yml`.** The
langchain/langgraph packages are deliberately unpinned there, and newer
releases have breaking API changes this code depends on. Snapshot first:
```bash
conda list -n moodle_backend | grep -iE 'langchain|langgraph'
```
and pin to those versions. See `DEV_HANDOFF.md` §8.

---

## 3. Files that live in no repo

The single most fragile category — covered by no git repo and no sync command.
If you rebuild this box, these do not come back on their own.

| File | Why it matters |
|---|---|
| `/usr/local/bin/videoelicit-start.sh` | `videoelicit-backend.service` execs it. Not in either repo. Copy it verbatim from a working server; do not reconstruct it. |
| `/opt/craftpilot_backend/.env` | Gitignored, root-only, mode 600. |
| `/opt/video_elicitation_annotation_tool/.env` | Same. |
| `/etc/httpd/conf.d/moodle-ssl.conf` | Vhost, all proxy rules, the `X-Internal-Token` header. |
| `/etc/httpd/conf/httpd.conf` | The server-scope dotfile deny block. |
| `/var/www/html/config.php` | Moodle DB credentials and `wwwroot`. |

```bash
ls -l /usr/local/bin/videoelicit-start.sh /var/www/html/config.php
ls -l /opt/craftpilot_backend/.env /opt/video_elicitation_annotation_tool/.env
```

---

## 3b. Outbound network — port 22 is blocked

**This VM cannot open outbound SSH connections on port 22.** `ssh git@github.com`
times out. Nothing in either repo documents this, and it is invisible until the
first time someone tries SSH git access.

GitHub's alternate SSH endpoint on **port 443** works, so `/root/.ssh/config`
routes both remotes through it:

```
Host github-craftpilot
  HostName ssh.github.com
  Port 443
  ...
```

```bash
# Should print REACHABLE. If it ever fails, git push over SSH is dead.
timeout 10 bash -c 'cat < /dev/null > /dev/tcp/ssh.github.com/443' \
  && echo REACHABLE || echo BLOCKED
```

Discovered 2026-08-13 while setting up deploy keys. Assume any other outbound
port is blocked until proven otherwise.

---

## 4. Apache configuration the plugins depend on

Browser traffic reaches both backends through `ProxyPass`, **not** through PHP.

```bash
grep -nE 'ProxyPass|RequestHeader' /etc/httpd/conf.d/moodle-ssl.conf
httpd -t                       # syntax check before any reload
```

Three things that must not break:

1. **`X-Internal-Token` on `<Location /craftpilot-api/>`** — how the backend
   authenticates the browser-followed 307 redirect out of `chat_proxy.php`.
   Missing header ⇒ chat returns 401.
2. **`/videoelicit-ui/` → `127.0.0.1:8005`** — the annotation SPA. The
   `/videoelicit-ui/api/videos/` rule must stay *above* the general
   `/videoelicit-ui/` rule; Apache matches in order.
3. **The dotfile deny block** (server scope, in `httpd.conf`, not the vhost).
   Until 2026-08-03 `.git` directories were publicly downloadable.

```bash
# Both must return 403.
for s in http https; do curl -s -o /dev/null -w "$s -> %{http_code}\n" -k \
  "$s://aimove.minesparis.psl.eu/local/craftpilot/.git/config"; done
```

---

## 5. Moodle database settings

Plugin settings live in the DB, not in any config file, so they survive a repo
reset and are invisible to `grep`. These are the ones with a hard dependency.

```bash
php -r 'define("CLI_SCRIPT",1);require("/var/www/html/config.php");
foreach (["local_craftpilot","local_videoelicit"] as $p) {
  echo "$p:\n";
  foreach ((array) get_config($p) as $k => $v) {
    if (preg_match("/key|secret|token|pass/i",$k)) { $v = $v === "" ? "(EMPTY!)" : "(set, ".strlen($v)." chars)"; }
    echo "  $k = $v\n";
  }
}'
```

| Setting | Must be | Breaks if wrong |
|---|---|---|
| `local_videoelicit/backend_url` | `http://localhost:8005` | `api_proxy.php` calls fail |
| `local_videoelicit/jwt_secret` | non-empty, matches `MOODLE_JWT_SECRET` | **silently falls back to an insecure default** |
| `local_craftpilot/internal_api_token` | matches Apache + `.env` | chat 401s |

---

## 6. Third-party APIs

| Service | Credential | Used for | Failure mode |
|---|---|---|---|
| Infomaniak AI Tools | `INFOMANIAK_API_KEY` + `INFOMANIAK_PRODUCT_ID`, in **both** `.env` files | chat generation, reranking, Whisper STT, AI tagging | chat returns nothing; transcription fails |
| LangSmith | `LANGSMITH_API_KEY` | tracing only | debugging gets harder; service unaffected |

The product ID is **part of the URL path**
(`api.infomaniak.com/2/ai/{product_id}/openai/v1`), so changing accounts means
changing both values, in both files.

```bash
# Reads the key from the env file without printing it.
set -a; . /opt/craftpilot_backend/.env; set +a
curl -s -H "Authorization: Bearer $INFOMANIAK_API_KEY" \
  "https://api.infomaniak.com/2/ai/$INFOMANIAK_PRODUCT_ID/openai/v1/models" \
  | python3 -c 'import json,sys; print(*[m["id"] for m in json.load(sys.stdin)["data"]], sep="\n")'
```
Three model families must appear, or features fail silently:

| Model | Consumed by |
|---|---|
| `swiss-ai/Apertus-70B-Instruct-2509` | craftpilot generation; videoelicit tagging + advisory |
| `whisper` (async batch STT) | videoelicit transcription (`fr`) |
| Qwen reranker (Cohere-style `/cohere/v2/rerank`) | craftpilot retrieval |

---

## 7. Browser-side dependencies

| What | Source | Risk |
|---|---|---|
| Three.js | **`cdn.jsdelivr.net`, fetched at runtime** | The BVH skeleton viewer breaks if jsDelivr is blocked or down. This is the only unvendored runtime dependency. Consider vendoring it. |
| `marked` | vendored in `amd/build/` | — |
| DOMPurify | vendored in `amd/build/dompurify.min.js` | **Has no `amd/src` counterpart — `grunt babel` will NOT regenerate it.** Deleting it is an XSS regression. This is why `amd/build/` is committed. |

```bash
curl -s -o /dev/null -w 'jsdelivr -> %{http_code}\n' https://cdn.jsdelivr.net/npm/three/build/three.module.js
ls -l /var/www/html/public/local/craftpilot/amd/build/dompurify.min.js
```

**Validating the AMD build.** Older docs said "check the file starts with
`define([`". That check is wrong — Babel emits its helper functions first, so a
perfectly good build starts with `function _typeof(o)`. Check for the real
failure instead, raw ES6 module syntax:

```bash
F=/var/www/html/public/local/craftpilot/amd/build/chat_interface.min.js
grep -qE '^\s*(import|export)\s' "$F" && echo "FAIL: raw ES6 in build" || echo "PASS"
```
A failure here breaks **all** JavaScript site-wide: RequireJS cannot load
`core/first`, and the primary navigation goes invisible on every page. That
symptom means the JS build, not the theme.

---

## 8. Deployment coupling — the trap

Neither live plugin directory is a git repo. Both are `rsync` targets, so the
repo and the live site can silently disagree in **either** direction.

**Each repo contains both halves of its application.** One repo, two things on
disk. `/opt/craftpilot_backend` is misleadingly named — it is the *whole* repo
(Python backend at the root, Moodle plugin in `plugin/`), not just the backend.

| Repo (the git checkout) | Backend half | Plugin half | Deployed to |
|---|---|---|---|
| `/opt/craftpilot_backend` | `api/ core/ services/ server.py` | `plugin/` | `/var/www/html/public/local/craftpilot` |
| `/opt/video_elicitation_annotation_tool` | `backend/` | `local_videoelicit/` | `/var/www/html/public/local/videoelicit` |

The deployed directories have **no `.git`** — `git status` there tells you
nothing. Edit in `/opt`, then `rsync`. Editing the webroot copy makes changes
no commit records, and the next deploy destroys them silently.

| Live path | Source of truth |
|---|---|
| `/var/www/html/public/local/craftpilot` | `/opt/craftpilot_backend/plugin/` |
| `/var/www/html/public/local/videoelicit` | `/opt/video_elicitation_annotation_tool/local_videoelicit/` |

```bash
diff -rq /opt/craftpilot_backend/plugin/ /var/www/html/public/local/craftpilot/ \
  --exclude=node_modules --exclude=.claude
diff -rq /opt/video_elicitation_annotation_tool/local_videoelicit/ \
         /var/www/html/public/local/videoelicit/
```

If these differ, **establish the direction before you fix it** — the two causes
need opposite responses:

```bash
stat -c '%y %n' <repo>/<file> <live>/<file>   # which side is newer?
git log --oneline -3 -- <path>                # do commits explain the diff?
```

- **Repo newer + commits match the diff** → an undeployed commit. Deploy
  repo → live. *(This was the real case on 2026-08-07: live was missing
  `jwt_helper::verify_token()`, which `stream.php` and `stream_ticket.php`
  both call, so unauthenticated streaming returned `Call to undefined
  method`. The fix had been committed two days earlier and never synced.)*
- **Live newer with no matching commit** → someone hand-edited past the
  workflow. Copy live → repo, commit, then continue.

Never blindly copy one way because a doc told you to.

---

## 9. What is *not* a dependency

Time-savers — these look load-bearing and are not.

- **`mod/craftpilot`** — dead since March 2026, zero course instances. The live
  plugin is `local/craftpilot`. Confirm via any AJAX request's `methodname`:
  live calls `local_craftpilot_*`.
  ⚠️ It is nonetheless **still installed and still visible** in Moodle
  (`mdl_modules.visible = 1`), so it remains in the activity chooser. Unused,
  but not uninstalled — see the open decision in `OFFBOARDING.md`.
- **`mdl_craftpilot*` tables** (no `local_` prefix) — belong to that dead
  plugin, all empty.
- **`api_proxy.php`** — referenced only in documentation. No `.js`, `.php`, or
  `.mustache` file calls it. Real traffic goes through Apache's `ProxyPass`.
- **`app.py` / `app_legacy.py`** in the craftpilot backend — an old Gradio/
  tkinter desktop prototype. The live entrypoint is `server:app`, which imports
  only `api.routes` and `config.settings`. Nothing imports either file.
- **The WebDAV storage backend** — configured but unused; all production rows
  are `source_type = uploaded`.
