# Handover docs — snapshot, not the original

These five files are a **backup copy**. The live, authoritative versions are
on the production VM at `/var/www/html/public/`:

| File | What it is |
|------|------------|
| `CLAUDE.md` | Conventions and gotchas; read by Claude Code automatically |
| `DEV_HANDOFF.md` | Everyday workflow — what to edit, what to run after |
| `DEPENDENCIES.md` | Out-of-repo dependencies (services, ports, credentials) |
| `OFFBOARDING.md` | The 2026-08/09 migration off personal accounts |
| `BEFORE_YOU_LEAVE.md` | Last-day revocation checklist |

## Why this copy exists

The Moodle webroot is **not** version-controlled — `/var/www/html/public` is
not a git repository, and Moodle core and the theme are not tracked anywhere.
Until 2026-08-14 these documents existed in exactly one place, with no history
and no backup: a bad `rsync`, a disk failure, or a VM rebuild would have taken
the entire handover with it.

## Sync direction — this matters

**Webroot → repo.** Always. Edit the files in `/var/www/html/public/`, then
refresh this copy:

```bash
for f in CLAUDE.md DEV_HANDOFF.md DEPENDENCIES.md OFFBOARDING.md BEFORE_YOU_LEAVE.md; do
  cp "/var/www/html/public/$f" "/opt/craftpilot_backend/docs/handover/$f"
done
cd /opt/craftpilot_backend && git add docs/handover && git commit && git push
```

Never edit these copies and sync them back. `CLAUDE.md` in particular is read
by Claude Code from the **webroot** path — a change made only here has no
effect on anything.

If the two ever disagree, check modification times before assuming which is
newer. This project has been bitten by guessing drift direction the wrong way
round (see `DEPENDENCIES.md` §8).

**Snapshot taken:** 2026-08-14.
