# Offboarding checklist — moving off personal accounts

Every external account this system depends on is currently held in one
person's name. This is the ordered list of actions to change that.

**Order matters more than anything else here.** Every credential below fails
*silently*: a dead Infomaniak key looks like an empty chat reply, a revoked
GitHub token looks like nothing at all until the next deploy. Each phase ends
with a verification step. Do not start a phase until the previous one verifies.

Nothing in this file can be automated from the server — every step needs
someone logged into a third-party account.

Status as of **2026-08-13**:

- **Phase 1 — done**, except transferring the two archived repos.
- **Phase 2 — done and verified.** The server pushes with its own deploy keys.
- **Phase 3 (Infomaniak) — blocked**, see the note in that section.
- **Phases 4–5 — not started.**

---

## Phase 1 — GitHub: personal account → organisation

Both repos now live in the org **`AIMoveCAOR`**, transferred 2026-08-13:

- `github.com/AIMoveCAOR/moodle_ai_assistant` — craftpilot backend + `plugin/`
- `github.com/AIMoveCAOR/video_elicitation_annotation_tool` — video backend + `local_videoelicit/`

Both are **public — a deliberate decision**, confirmed 2026-08-13. No secrets
were ever committed (`.env` gitignored in both, history scans clean).

- [x] **Create the organisation.** GitHub Free is enough — unlimited private
      repos, no cost. Suggested name `aimove-minesparis`. Use a role-based
      billing email, not a personal one.
- [x] **Decide public vs private, deliberately.** Decided 2026-08-13: **both
      stay public** under the org. (`CLAUDE.md` previously described one as
      private — that was wrong and has been corrected.)
- [x] **Transfer both repos** — Settings → General → Danger Zone → Transfer.
      Issues, PRs and stars come along, and GitHub redirects the old URLs.
- [x] **Add the successor as `Owner`**, not `Member` — they need to manage
      settings and billing without you.
- [ ] **Delete the orphaned `CLAUDE_CODE_OAUTH_TOKEN` secret** on
      `moodle_ai_assistant`, and revoke it in the Claude account. Its workflow
      no longer exists in `HEAD`; only the secret and a stale workflow record
      remain. Actions secrets are not reliably carried across a transfer —
      check with `gh secret list` afterwards and re-create only what is needed.
- [x] **Repoint the server's remotes** (done 2026-08-13):
      ```bash
      cd /opt/craftpilot_backend
      git remote set-url origin https://github.com/AIMoveCAOR/moodle_ai_assistant.git
      cd /opt/video_elicitation_annotation_tool
      git remote set-url origin https://github.com/AIMoveCAOR/video_elicitation_annotation_tool.git
      ```
      Still HTTPS, so pushes still authenticate as the **personal** `gh` login.
      Phase 2 replaces that credential — the address is fixed, the identity is not.
- [ ] **Archived repos — STILL ON THE PERSONAL ACCOUNT.** Audited 2026-08-13
      by cloning both and counting commits. They are **not** equivalent:

      | Repo | Commits | Unique history? | Action |
      |---|---|---|---|
      | `moodle-plugin-ai` (public) | **382**, 2025-07-15 → 2026-08-03 | ✅ exists nowhere else | ✅ **transferred to `AIMoveCAOR` 2026-08-13** |
      | `Somekindofa/moodle-local-craftpilot` (private) | **3**, all 2026-08-03/06 | ❌ all three are in `moodle_ai_assistant` | ⏳ **still personal** — fate undecided; see below |

      `moodle-plugin-ai` transferred cleanly and **remained archived** through
      the move — unarchiving first was not required.

      `moodle-plugin-ai` is the entire development history of the retired
      `mod/craftpilot` activity module. Losing it loses that record permanently.

      `moodle-local-craftpilot` was a short-lived staging repo: created as a
      snapshot import on 2026-08-03, given two fixes, merged into
      `moodle_ai_assistant/plugin/` on 2026-08-06 (commit `a7a2f12`). Verified
      redundant — a bare clone contains exactly the 3 commits already merged.

      > Note: `git log -- plugin/` shows only the merge commit. That is git's
      > history simplification hiding a subtree merge, **not** lost history.
      > Use `git log 2cd6709 ^6ef7a1e` to see what came across.

      Archived repos are read-only, which blocks the transfer. Sequence:
      **unarchive → transfer to `AIMoveCAOR` → re-archive.** Or by CLI:
      ```bash
      gh api -X POST repos/Somekindofa/moodle-plugin-ai/transfer        -f new_owner=AIMoveCAOR
      gh api -X POST repos/Somekindofa/moodle-local-craftpilot/transfer -f new_owner=AIMoveCAOR
      ```
      **How to tell a transfer really happened:** the old path must redirect.
      `gh api repos/Somekindofa/<name> --jq .full_name` printing
      `AIMoveCAOR/<name>` is proof; printing `Somekindofa/<name>` means it did
      not happen. A transfer *moves* a repo — it never leaves a copy behind, so
      there is no risk of deleting the org's copy along with yours.

- [ ] **Other personal repos that may be institutional work** — flagged
      2026-08-13, ownership undecided: `elicitation_extraction_vlam`,
      `whisper_transcription_webserver`, `videos-transcripts_dataset` (private),
      `moodle-plugin-template-figma` (private). None is a runtime dependency of
      this server. Decide with the project lead; record the outcome here.

**Verify before moving on:**
```bash
cd /opt/craftpilot_backend && git ls-remote origin HEAD
cd /opt/video_elicitation_annotation_tool && git ls-remote origin HEAD
```

> **Branch protection:** not recommended here. Work goes straight to `main` on
> a live server with no staging (see `DEV_HANDOFF.md` §1); a PR gate would just
> be friction for a one-person team. Revisit when there is more than one dev.

---

## Phase 2 — Server push access that outlives you

**Done and verified 2026-08-13.** Both `root` and `claude-runner` now push with
their own per-repo deploy keys over `ssh.github.com:443`. `gh` is still
authenticated as the personal account, but **nothing depends on it any more** —
`gh auth logout` is now safe (it is a last-day item, see `BEFORE_YOU_LEAVE.md`).

*Historical note:* earlier revisions of this file claimed git's credential
helper was `!gh auth git-credential`. That was never true on this box — no
`credential.helper` is set in any scope, global or per-repo. The remotes were
plain HTTPS and `gh` supplied auth another way. Moot now that both are SSH,
but do not go looking for a helper that does not exist.

Use **per-repo deploy keys**, not a bot account — no seat cost, no shared
password, per-repo scope, independently revocable.

- [x] Generate one key per repo (done 2026-08-13):
      ```bash
      ssh-keygen -t ed25519 -f /root/.ssh/id_aimove_craftpilot  -N "" -C "aimove-server craftpilot"
      ssh-keygen -t ed25519 -f /root/.ssh/id_aimove_videoelicit -N "" -C "aimove-server videoelicit"
      ```
- [x] Add each **public** key as a repo **Deploy key with write access**
      (Settings → Deploy keys). One key per repo — do not reuse.
- [x] Add host aliases so each remote uses its own key, in `/root/.ssh/config`:
      ```
      Host github-craftpilot
        HostName ssh.github.com
        Port 443
        User git
        IdentityFile /root/.ssh/id_aimove_craftpilot
        IdentitiesOnly yes

      Host github-videoelicit
        HostName ssh.github.com
        Port 443
        User git
        IdentityFile /root/.ssh/id_aimove_videoelicit
        IdentitiesOnly yes
      ```
      ⚠️ **Port 443, not 22.** This VM's outbound port 22 is firewalled —
      `ssh git@github.com` times out. GitHub's `ssh.github.com:443` endpoint
      works. See `DEPENDENCIES.md` §3b.
- [x] Switch remotes to SSH (done 2026-08-13):
      ```bash
      cd /opt/craftpilot_backend
      git remote set-url origin git@github-craftpilot:AIMoveCAOR/moodle_ai_assistant.git
      cd /opt/video_elicitation_annotation_tool
      git remote set-url origin git@github-videoelicit:AIMoveCAOR/video_elicitation_annotation_tool.git
      ```
- [x] Set a neutral commit identity — **for both users** (done 2026-08-13):
      ```bash
      git config --global user.name  "aimove server"
      git config --global user.email "aimove.caor@minesparis.psl.eu"
      sudo -u claude-runner git config --global user.name  "aimove server (claude-runner)"
      sudo -u claude-runner git config --global user.email "aimove.caor@minesparis.psl.eu"
      ```
      ⚠️ `claude-runner`'s identity was set to the departing developer's
      **personal Gmail**, so agent-made commits were stamped with it. Easy to
      miss because `git config --global` is per-user — checking `root` alone
      tells you nothing about `claude-runner`. Verify both:
      ```bash
      git config --global user.email; sudo -u claude-runner git config --global user.email
      ```

- [x] **Decided 2026-08-13: `claude-runner` KEEPS push access**, via its own
      key pair — not by sharing root's. Rationale and trade-off:
      - It gets `/home/claude-runner/.ssh/id_aimove_{craftpilot,videoelicit}`,
        registered as a *second* deploy key on each repo. Root's keys are
        untouched, so `DEV_HANDOFF.md` §2's secret cage still holds —
        `claude-runner` never reads anything under `/root`.
      - Revoking it later is two deletions on GitHub's Deploy keys pages, with
        no effect on root.
      - ⚠️ **Trade-off accepted knowingly:** this gives an AI agent unreviewed
        push access to both production repos, and there is deliberately no
        branch protection. Acceptable while a human supervises each session.
        **Successor: revisit when more than one person works on this.**
      - `claude-runner` could already *commit* (ACLs grant it write on both
        `.git` directories); it only ever lacked a way to *push*.

**Verify — write access must be proven before you log out of anything.**

⚠️ `git push --dry-run` is **not** a sufficient check. With nothing to push it
prints `Everything up-to-date` without ever asking GitHub about permissions —
a green result that tested nothing. Probe `git-receive-pack` directly instead;
it is non-mutating but forces the authorization decision:

```bash
ssh git@github-craftpilot  "git-receive-pack 'AIMoveCAOR/moodle_ai_assistant.git'" \
  < /dev/null 2>&1 | head -c 120
ssh git@github-videoelicit "git-receive-pack 'AIMoveCAOR/video_elicitation_annotation_tool.git'" \
  < /dev/null 2>&1 | head -c 120
```

A ref advertisement (`00xx<sha> refs/heads/...`) means **write granted**.
`ERROR: The key you are using does not have write access` means the deploy key
was added without "Allow write access" ticked.

Also confirm each key is bound to the repo you think it is:
```bash
ssh -T git@github-craftpilot; ssh -T git@github-videoelicit
```
A **deploy key** greets you with the *repo* name (`Hi AIMoveCAOR/…!`); a
*personal* key greets you with a username. If the names are swapped, swap the
deploy keys on GitHub — do not edit the ssh config.

- [ ] **Only now**: `gh auth logout` as `root`, and as `claude-runner`.

---

## Phase 3 — Infomaniak: personal wallet → prepaid org account

> 🚧 **BLOCKED as of 2026-08-13 — pending an institutional decision on how AI
> services are paid for.** This is the one phase that cannot be finished from
> the server, and the one with a hard deadline.
>
> **What is at stake.** The Infomaniak key in both `.env` files is a *personal*
> key on a *personal* wallet. It pays for:
> - craftpilot chat generation and retrieval reranking,
> - videoelicit transcription (Whisper), AI tagging, and advisory.
>
> When that wallet stops being funded — or the departing developer's account is
> closed — **all of the above stop working**, and they stop *silently*: chat
> returns an empty reply, transcription just fails. Nothing logs "your billing
> lapsed."
>
> **The deadline is 2026-09-11 — the departure date — not a preference.**
> Whoever owns the budget decision needs to know that, in writing, now. An
> institutional finance decision takes weeks; there are about four. If it is
> not settled by then, the realistic fallback is not "it keeps working" but
> **a planned, announced degradation**: decide in advance whether AI features
> get disabled cleanly with a user-visible notice, or are left to fail
> silently mid-session for whoever is using them. Record the outcome here.

Read `DEV_HANDOFF.md` §6c first. Summary: this is **not** just an API-key
rotation. The product ID is part of the API URL, and the key lives in **two**
`.env` files.

- [ ] **Create the Infomaniak organisation account**; add the successor as an
      administrator immediately.
- [ ] **Confirm prepaid billing terms for AI Tools directly with Infomaniak**
      before relying on a ~100 €/month prepaid model — verify rather than
      assume, then top up.
- [ ] **Create the AI Tools product** under the org and record the new product
      ID.
- [ ] **Generate an API key** scoped to that product.
- [ ] **Verify the models exist — before touching any config:**
      ```bash
      curl -s -H "Authorization: Bearer $NEW_KEY" \
        "https://api.infomaniak.com/2/ai/$NEW_PRODUCT_ID/openai/v1/models"
      ```
      All three must be present, or features break silently:
      `swiss-ai/Apertus-70B-Instruct-2509`, `whisper`, and the Qwen reranker.
- [ ] **Update both `.env` files** — as `root`, not `claude-runner`:
      `/opt/craftpilot_backend/.env` and
      `/opt/video_elicitation_annotation_tool/.env`.
- [ ] `systemctl restart craftpilot-backend videoelicit-backend`
- [ ] **Run the full verification below with the old key still live.**
- [ ] **Then revoke the old key** and re-run the chat check, expecting it to
      *fail*. That is the only proof the new credentials are genuinely in use
      and not a cached client.

---

## Phase 4 — Remaining personal ties

- [ ] **LangSmith** (`LANGSMITH_API_KEY`, craftpilot `.env`) — personal
      account, EU region, project `Craftpilot`. Tracing only; losing it
      degrades debugging, not service. Either move it to an org account or set
      `LANGSMITH_TRACING=false` and record that tracing is unavailable. Do not
      leave a departing person's key in place.
- [ ] **WebDAV settings** (`local_videoelicit/webdav_*`) — held a named
      person's institutional OwnCloud login rather than a service account.
      Unused in production (all videos are `source_type = uploaded`), so
      clearing it is safe. If WebDAV is ever re-enabled, provision a real
      service account.
- [ ] **Participant-facing privacy notice** (`js/app.js`, the ReSOuRCE data
      notice shown to interview participants) names two contacts. This is a
      GDPR contact point in a research project, so it must name a **real,
      reachable person** — it cannot simply be deleted. Decide with the
      project lead whether the successor replaces the departing contact or the
      remaining named contact stands alone. **This is a data-protection
      decision, not a code cleanup.**
- [x] **Git commit identity** — was the departing developer's personal Gmail
      under `claude-runner`. Fixed 2026-08-13 (see Phase 2). Re-check with:
      ```bash
      git config --global user.email; sudo -u claude-runner git config --global user.email
      ```
      Both must print `aimove.caor@minesparis.psl.eu`.
- [ ] **Moodle admin accounts.** Six site admins exist (2026-08-13), but only
      two have logged in this year:

      | Admin | Last access |
      |---|---|
      | `theo.akbas` | 2026-08-06 — departing |
      | `dimitris.makrygiannis` | 2026-04-08 |
      | `brenda.olivas` | 2024-04-11 |
      | `raphael.menegaldo` | 2023-09-27 |
      | `hekatonkheiros` (`aimove@mines-paristech.fr`) | 2023-09-26 — role account, dormant |
      | `gavriela.senteri` | 2023-09-12 |

      `dimitris.makrygiannis` is also in `/root/.ssh/authorized_keys`, so is
      almost certainly the successor — **but 4 months stale is not proof the
      login still works.** Ask them to log in and confirm *before* the
      departing admin is demoted. List them again with:
      ```bash
      cd /var/www/html/public && php -r 'define("CLI_SCRIPT",1);
        require("/var/www/html/config.php"); global $DB;
        foreach (explode(",", (string)get_config(null,"siteadmins")) as $id) {
          if ($id === "") continue;
          $u = $DB->get_record("user", ["id"=>(int)$id], "username,email,lastaccess");
          printf("  %-22s %s\n", $u->username,
            $u->lastaccess ? date("Y-m-d", $u->lastaccess) : "NEVER"); }'
      ```
      Note a role account `hekatonkheiros` / `aimove@mines-paristech.fr`
      already exists, dormant since 2023 — distinct from the new
      `aimove.caor@minesparis.psl.eu`. Decide which is canonical rather than
      creating a third.

      Demote or disable the departing admin only *after* the successor
      confirms theirs works.
- [x] **Server access** — confirmed 2026-08-13: the colleague has `root`.
      ⚠️ **But it is *shared* root, not independent access.** Established
      2026-08-13:
      - **No human user accounts exist.** The only non-system account is
        `claude-runner` (uid 1000), a service account.
      - `wheel` is empty and `/etc/sudoers.d/` is empty — nobody has a sudo
        path; everyone logs in *as* `root`.
      - `sshd` has both `PermitRootLogin yes` and `PasswordAuthentication yes`.
      - Consequence: **there is no attribution.** `last` shows every session as
        `root`. Nothing on this box records who did what.

      **This changes what "remove the departing developer's access" means.**
      There is no account to delete. See the revised step in
      `BEFORE_YOU_LEAVE.md` §5.

- [ ] **Recommended for the successor: give each human their own account.**
      Not urgent, not required for the handoff, and deliberately left undone —
      but a system where every action is an unattributable `root` login is a
      problem that grows with the team. `useradd` + `wheel` + key-only auth,
      then `PermitRootLogin prohibit-password`.

---

## Phase 5 — Things nobody on this list owns

Flagged, deliberately not actioned. These need an institutional owner, and
**no credential rotation can fix them**:

- **TLS certificate.** `/etc/httpd/certs/wildcard.crt` is a GÉANT/Sectigo
  wildcard for `*.minesparis.psl.eu` that **expired 2024-03-13**. The site
  still works, so TLS is evidently terminated by an institutional proxy in
  front of this box — but nobody has written down who renews it, or what
  happens to this server's own cert. Find the owner.
- **DNS** for `aimove.minesparis.psl.eu`.
- **The VM itself** — who provisions, patches, backs up, and pays for it.

  🔎 **Lead, found 2026-08-13.** `/root/.ssh/authorized_keys` contains a key
  commented **`root@ansible.interne.mines-paristech.fr`**, and
  `/etc/ssh/sshd_config.d/00-ansible_system_role.conf` is an Ansible-managed
  drop-in. **This VM is under central configuration management by Mines Paris
  IT.** Whoever operates that Ansible control node either owns the VM or knows
  who does — and is very likely also the answer to the TLS certificate
  question above. **Start here.** Ask while you still have an institutional
  account; this is much harder to chase from outside.

  Ten other people hold root keys on this box (`Vincent`, `Sabrine`, `Brice`,
  `Johan`, `Riaz`, `Clef_SSH_Valery`, `dimitris.makrygiannis`,
  `cwazana@pc-charlie`, `id_rsa_satory`, plus Ansible). Some may be former
  staff. **Auditing that list is not part of this handoff**, but the successor
  should know it exists and that nobody appears to be curating it.
- **`cloud.minesparis.psl.eu`** — the institutional OwnCloud, if WebDAV is ever
  re-enabled.

Write the answers here as you find them. An unowned dependency is the failure
that outlasts every handoff.

---

## Open decisions left for the successor

Deliberately not decided during the handoff. Each is safe as-is; none is
urgent. Record the outcome here when you settle it.

- **What to do with `local/videoelicit/api_proxy.php`.** It is a complete,
  working PHP proxy that injects a JWT — but **no `.js`, `.php`, or
  `.mustache` file calls it**. Browser traffic reaches the backend through
  Apache's `ProxyPass` rules instead. Its `backend_url` also pointed at a dead
  port (8006) until 2026-08-07, which suggests it has not run in a long time.

  It was left in place, with the port corrected and the docs rewritten to
  describe it as dormant. The open question is whether it was abandoned or
  intended as a future server-mediated API path. **Delete it** if you are
  confident it is dead — it is one file plus four doc references. **Keep it**
  if you might want a PHP-mediated path later; it costs nothing where it sits.
  Do not route new code through it without first testing it end to end.

- **Repo visibility.** Both repos are public. Confirmed deliberate 2026-08-13
  — see Phase 1.

- **Whether to uninstall `mod_craftpilot`.** Found 2026-08-13: the dead
  activity module is not merely dead code on disk — it is **still registered
  and still visible** in Moodle:

  ```
  mod_craftpilot version = 2026022700
  mdl_modules row: EXISTS (visible = 1)   ← teachers can still add it to courses
  course instances: 0
  all 8 mdl_craftpilot* tables: 0 rows
  ```

  So any teacher can add it from the activity chooser, and it would half-work
  against a backend contract that no longer exists. Nobody ever has (zero
  instances, zero rows), so this is latent rather than active breakage.

  **Left deliberately untouched** — uninstalling a Moodle module is a real
  operation, and `CLAUDE.md` forbids editing that tree. The clean fix is
  *Site admin → Plugins → Plugins overview → uninstall `mod_craftpilot`*,
  which drops the tables and the `mdl_modules` row. Its history is preserved
  in `moodle-plugin-ai` (see Phase 1), so uninstalling loses nothing.

  A lighter interim option: hide it from the activity chooser without
  uninstalling (*Site admin → Plugins → Activity modules → Manage activities*
  → eye icon).

## Full verification

Run after each phase. Everything must pass **before** revoking any old
credential.

```bash
# Services and health
systemctl is-active httpd mariadb craftpilot-backend videoelicit-backend
curl -s -o /dev/null -w 'craftpilot  :8000 -> %{http_code}\n' http://127.0.0.1:8000/api/health
curl -s -o /dev/null -w 'videoelicit :8005 -> %{http_code}\n' http://127.0.0.1:8005/api/health

# Backend URL setting agrees with reality (expect 8005)
php -r 'define("CLI_SCRIPT",1);require("/var/www/html/config.php");
        echo get_config("local_videoelicit","backend_url"), "\n";'

# Dotfile deny block still closed — both must be 403
for s in http https; do curl -s -o /dev/null -w "$s -> %{http_code}\n" -k \
  "$s://aimove.minesparis.psl.eu/local/craftpilot/.git/config"; done

# Git auth works as the machine account
cd /opt/craftpilot_backend && git ls-remote origin HEAD
cd /opt/video_elicitation_annotation_tool && git ls-remote origin HEAD
```

**In a browser — automated checks cannot cover these:**

1. Send a craftpilot chat message in a course; confirm it **streams** a reply.
   Exercises Infomaniak generation, the reranker, and `X-Internal-Token`.
2. Open a videoelicit activity, play a video, and **seek mid-stream**;
   confirms HTTP 206 Range handling still works.
3. Trigger a **transcription**. This is the check most likely to fail after an
   Infomaniak migration and the one everybody forgets — it is the only thing
   that exercises the second `.env` file.
4. Confirm the **primary navigation is visible** on any page. Invisible nav
   site-wide is the canary for a broken AMD bundle.
