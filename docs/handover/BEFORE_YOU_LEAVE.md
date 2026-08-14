# Before you leave — last-day checklist

**This file is different from `OFFBOARDING.md`.**

- `OFFBOARDING.md` = the migration work, done **in advance**, while you're still
  here to fix what breaks.
- **This file** = the irreversible revocations, done on your **last day**, after
  everything in `OFFBOARDING.md` verifies.

Doing anything here early will break production. Doing it late leaves your
personal credentials in a system you no longer work on. Both are bad — hence
two files.

**Rule: nothing in this file happens until the matching `OFFBOARDING.md` phase
is done and verified.** Every item below is "remove the old thing", and each
one assumes the new thing already works.

**Departure date: 2026-09-11.** That is the day this file gets executed —
*not* the day to discover what is still broken. Anything needing someone
else's action (Infomaniak billing, Mines Paris IT, the handover walkthrough)
must be started weeks earlier, because you will not be here to chase it.

Status: **not started** (this file). See `OFFBOARDING.md` for advance work,
which is largely done.

---

## 0. Gate — do not start until all of these are true

- [ ] `OFFBOARDING.md` Phases 1–4 complete and verified
- [ ] Successor has `root` access to this VM — ⚠️ **confirmed 2026-08-13, but
      it is a *shared* root credential, not an independent account.** Read §5
      before you revoke anything: if the credential is the root *password*,
      rotating it locks your colleague out too, so it must be done with them
      present.
- [ ] Successor has a Moodle site-admin account and has logged into it
- [ ] Successor is an Owner of the GitHub org
- [ ] Successor is an administrator on the Infomaniak organisation account
- [ ] Deploy keys are working — `git push --dry-run` succeeds on both repos
- [ ] Someone other than you can reach the org's email inbox and its 2FA

If any of these is false, stop. Fix it first.

---

## 1. Claude Code / Anthropic

Your personal Claude subscription is authenticated on this VM under **two**
separate user accounts. Both must go, or your subscription keeps being usable
by whoever logs into this box.

- [ ] Log out as `root`:
      ```bash
      claude   # then: /logout
      ```
- [ ] Log out as `claude-runner`:
      ```bash
      sudo -u claude-runner -i
      claude   # then: /logout
      ```
- [ ] Confirm both credential files are gone:
      ```bash
      ls -l /root/.claude/.credentials.json \
            /home/claude-runner/.claude/.credentials.json
      ```
      Both should report "No such file". If either remains, delete it.
- [ ] **Revoke the GitHub MCP connection.** `root`'s credentials also hold an
      MCP OAuth grant for the GitHub plugin (`plugin:github:github|…`) tied to
      your GitHub account. Logging out should clear it — verify, and also
      revoke the authorisation from your GitHub account's
      *Settings → Applications → Authorized OAuth Apps*.
- [ ] **Delete your session transcripts.** These contain your working history
      and possibly pasted secrets:
      ```bash
      rm -f /root/.claude/history.jsonl /home/claude-runner/.claude/history.jsonl
      rm -rf /root/.claude/projects /home/claude-runner/.claude/projects
      ```
- [ ] Delete the `CLAUDE_CODE_OAUTH_TOKEN` secret on the `moodle_ai_assistant`
      repo, and revoke that token in your Claude account:
      ```bash
      gh secret delete CLAUDE_CODE_OAUTH_TOKEN -R AIMoveCAOR/moodle_ai_assistant
      ```
      *(Safe to do any time — its workflow no longer exists in `HEAD`. It is
      unrelated to your Claude login above.)*
- [ ] Tell the successor that Claude Code on this VM now needs **their own**
      login, and point them at `DEV_HANDOFF.md` §2 for the `claude-runner`
      sandbox rule.

---

## 2. GitHub

- [ ] Confirm deploy keys work — **this must pass before the next step**:
      ```bash
      cd /opt/craftpilot_backend && git push --dry-run origin HEAD
      cd /opt/video_elicitation_annotation_tool && git push --dry-run origin HEAD
      ```
- [ ] Log out the CLI as both users:
      ```bash
      gh auth logout
      sudo -u claude-runner -i gh auth logout
      ```
- [ ] Confirm nothing is still authenticated as you:
      ```bash
      gh auth status; sudo -u claude-runner gh auth status
      ```
- [ ] Reset the global git identity so future commits aren't attributed to you:
      ```bash
      git config --global user.name  "aimove server"
      git config --global user.email "<role-address>"
      ```
- [ ] Remove yourself as an Owner of the org — **last**, after confirming the
      successor can administer it alone
- [ ] **`Somekindofa/moodle-local-craftpilot` — settle its fate. Last chance.**
      `moodle-plugin-ai` was transferred to the org on 2026-08-13; this one was
      deliberately kept on your personal account pending a decision.

      **The decision expires on your last day.** It is **private**, so nobody
      at the org can see it exists, and once your account lapses no one can
      transfer it. "Decide later" becomes "gone" by default.

      Low stakes either way — 3 commits, all already present in
      `AIMoveCAOR/moodle_ai_assistant` (verified 2026-08-13). So:
      ```bash
      # keep it under the org:
      gh api -X POST repos/Somekindofa/moodle-local-craftpilot/transfer -f new_owner=AIMoveCAOR
      # verify — must print AIMoveCAOR/moodle-local-craftpilot:
      gh api repos/Somekindofa/moodle-local-craftpilot --jq .full_name
      ```
      …or delete it, or leave it and **write here that leaving it was
      deliberate** so nobody later wonders what was lost.

---

## 3. Infomaniak

- [ ] Confirm the new org key is live and working (chat **and** transcription)
- [ ] Revoke your personal API key
- [ ] Re-run one chat request and confirm it **fails** — proof the new
      credentials are genuinely in use, not a cached client
- [ ] Confirm no billing for this project remains on your personal wallet
- [ ] Remove yourself from the Infomaniak organisation account

---

## 4. LangSmith

- [x] ~~Either move tracing to an org account, or set `LANGSMITH_TRACING=false`~~
      **Done 2026-08-14 — tracing was re-pointed, not disabled.** Now on the
      `aimove.caor` account, EU region, project `Craftpilot`, using a
      **workspace-scoped** Service Key. Verified: runs landed.
- [x] ~~Remove `LANGSMITH_API_KEY` if tracing is disabled~~ — n/a, tracing is on
- [x] Record in `DEV_HANDOFF.md` which of the two you chose — see §5
      *Debugging a RAG call*, including the silent-failure trap
- [x] **Revoke your personal LangSmith key** — done 2026-08-14, key rotated.
      Re-verify at any time (expect `['Craftpilot']`; a `403` means the wrong
      key is in `.env`):
      ```bash
      /root/miniconda3/envs/moodle_backend/bin/python -c "
      import dotenv; dotenv.load_dotenv('/opt/craftpilot_backend/.env')
      from langsmith import Client
      print([p.name for p in Client().list_projects(limit=10)])"
      ```
      Expect `['Craftpilot']`. A `403` here means the wrong key is in `.env`.
- [ ] **Create a real LangSmith organisation and add Dimitris to it.**
      *Decided 2026-08-14, not yet done.* The workspace currently sits in an
      auto-created org named `Personal` (`is_personal: true`, free tier)
      belonging to `aimove.caor`, so access depends on holding that one login
      rather than on org membership.

      ⚠️ Moving the workspace changes **both** the API key and the workspace
      id, so `.env` must be updated and the backend restarted. Re-verify with
      the probe above — remember that a broken key is **silent**: chat keeps
      working and nothing is logged. Only a run appearing proves it.

---

## 5. Institutional accounts

- [ ] Confirm nothing still references your institutional login. The WebDAV
      settings were cleared on 2026-08-07 — verify they're still empty:
      ```bash
      php -r 'define("CLI_SCRIPT",1);require("/var/www/html/config.php");
        foreach (["webdav_username","webdav_password","webdav_storage_path"] as $k)
          echo "  $k = ".var_export(get_config("local_videoelicit",$k), true)."\n";'
      ```
      All three should be `false` (unset).
- [ ] **Privacy notice** (`js/app.js`, ReSOuRCE data notice shown to interview
      participants) still names you as a GDPR contact. This needs a real,
      reachable person — settle with the project lead who replaces you.
      **A data-protection decision, not a code cleanup.**
- [ ] Demote or disable your Moodle site-admin account — after the successor
      confirms theirs works
- [ ] **Revoke your server access — read this, it is not a simple checkbox.**
      There is **no user account of yours to delete**: this VM has no human
      accounts at all, and everyone logs in directly as `root` (confirmed
      2026-08-13). So "removing your access" means removing the *shared* root
      credential you both use. Two cases:

      **✅ Resolved 2026-08-13 — this is the easy case.** Root access is by
      **SSH key**, not password. `/root/.ssh/authorized_keys` held 14 keys, of
      which **four are yours**: three commented `theo.akbas` and one
      `somekindofathing`. Removing them leaves the other ten — including your
      colleague's — completely untouched.

      ```bash
      # 1. back up, timestamped
      cp /root/.ssh/authorized_keys "/root/.ssh/authorized_keys.bak.$(date +%F)"

      # 2. show what will be removed — CHECK THIS before step 3
      grep -nE 'theo\.akbas|somekindofathing' /root/.ssh/authorized_keys

      # 3. remove only those lines
      grep -vE 'theo\.akbas|somekindofathing' \
        "/root/.ssh/authorized_keys.bak.$(date +%F)" > /root/.ssh/authorized_keys

      # 4. verify: expect 10 keys, none of them yours
      awk '{print NR": "$3}' /root/.ssh/authorized_keys
      ```

      **Keep an active session open** while you verify — if something goes
      wrong you want a shell that is already authenticated. Then confirm from
      your own machine that you are genuinely locked out:
      ```bash
      ssh root@aimove.minesparis.psl.eu    # must FAIL
      ```
      If it succeeds, an agent still holds your key, or a key of yours carries
      a different comment. Do not stop until it fails.

      Then confirm you are actually out — from *your own* machine, after
      rotating:
      ```bash
      ssh root@aimove.minesparis.psl.eu    # must fail
      ```

---

## 6. Handover of things only you know

- [ ] Walk the successor through one real change end to end: edit → `rsync` →
      purge caches → verify. The deploy step is the part docs never convey well
- [ ] Confirm they've read `DEV_HANDOFF.md`, `DEPENDENCIES.md`, `CLAUDE.md`
- [ ] Point them at `DEPENDENCIES.md` §8 — the repo/live drift trap that bit us
      on 2026-08-07
- [ ] Hand over any credentials **not** in `DEV_HANDOFF.md` §6b
- [ ] Tell them what's half-finished, and what you'd have done next

---

## 7. Add your own items here

Things that occur to you between now and then. Better written down badly than
remembered late.

- [ ] 
- [ ] 
- [ ] 

---

## 8. Final sweep — run on your actual last day

```bash
# Nothing authenticated as you
gh auth status 2>&1; sudo -u claude-runner gh auth status 2>&1
ls -l /root/.claude/.credentials.json /home/claude-runner/.claude/.credentials.json 2>&1

# Your name and personal accounts gone from config
git config --global user.email
grep -rniE 'akbas|somekindofa|protonmail' /opt/craftpilot_backend /opt/video_elicitation_annotation_tool \
  --include='*.py' --include='*.php' --include='*.js' --include='*.yml' 2>/dev/null | grep -v node_modules

# Everything still actually works without you
systemctl is-active httpd mariadb craftpilot-backend videoelicit-backend
curl -s -o /dev/null -w 'craftpilot  :8000 -> %{http_code}\n' http://127.0.0.1:8000/api/health
curl -s -o /dev/null -w 'videoelicit :8005 -> %{http_code}\n' http://127.0.0.1:8005/api/health
```

Then, in a browser, with the successor watching: send a chat message, play and
seek a video, and run a transcription. If all three work and nothing above is
authenticated as you, you're done.
