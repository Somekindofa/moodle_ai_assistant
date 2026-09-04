# Runbook — re-ingesting the course corpus

Follow these in order. Steps 1–4 are safe and reversible. Step 5 is the long one.

**Why this is needed:** the heading-hierarchy and breadcrumb-translation fixes
(see `changes/02-ingestion-chunking.md`) repair nothing already indexed. All
11,611 existing course chunks keep their bad breadcrumbs until re-ingested.

**Do not use the "Re-ingest All" button in the admin panel.** It sets
`set_time_limit(300)` on a job that takes hours, and clears
`local_craftpilot_cm_index` before it starts — so it dies partway through and
also loses the record of what it had done. The CLI script below exists because
of that.

---

## 1. Back up the vector database (~1.3 GB, /var has 193 GB free)

```bash
sudo systemctl stop craftpilot-backend
sudo cp -a /opt/craftpilot_backend/chroma_langchain_db \
           /var/backups/craftpilot/chroma_pre_reingest_$(date +%Y%m%d)
sudo systemctl start craftpilot-backend
```

Stop the service first: the local Chroma client is **not process-safe**, so
copying while it writes can capture a torn file.

To roll back later: stop the service, move the backup over
`chroma_langchain_db`, start the service.

## 2. Deploy the CLI script

The plugin has a separate, non-symlinked copy under the web root, so the repo
copy is not what runs:

```bash
sudo cp /opt/craftpilot_backend/plugin/cli/reingest_all.php \
        /var/www/html/public/local/craftpilot/cli/reingest_all.php
sudo chown apache:apache /var/www/html/public/local/craftpilot/cli/reingest_all.php
```

## 3. Dry run — costs nothing, makes no LLM calls

```bash
sudo -u apache php /var/www/html/public/local/craftpilot/cli/reingest_all.php --all --dry-run
```

Expect ~571 modules listed. If the count is wildly different, stop and
investigate before going further.

## 4. Pilot on one course, then CHECK IT

Course 109 is the Greek test course, which is where the breadcrumb bug was
found — so it is the right pilot.

```bash
sudo -u apache php /var/www/html/public/local/craftpilot/cli/reingest_all.php --course=109
```

Then verify two things before committing to a full run:

1. **Time it.** Note how long it took and how many modules it did. That is your
   real rate — multiply out to 571 modules for the true full-run duration.
2. **Check a breadcrumb actually improved.** Ask the assistant a question inside
   course 109 and confirm the answer still cites the right material. The
   breadcrumb fix is proven structurally but its *translation quality* was never
   measured against a live LLM (see `changes/02-ingestion-chunking.md`), so this
   is the first real evidence either way.

If the pilot looks wrong, restore the backup from step 1 and stop.

## 5. Full run

It takes hours, so run it detached — `screen` is installed:

```bash
screen -S reingest
sudo -u apache php /var/www/html/public/local/craftpilot/cli/reingest_all.php --all 2>&1 | tee /var/log/craftpilot_reingest.log
# detach with Ctrl-A then D; reattach with: screen -r reingest
```

If it is interrupted, continue without redoing finished work:

```bash
sudo -u apache php /var/www/html/public/local/craftpilot/cli/reingest_all.php --all --resume
```

`--resume` also retries only the modules that errored.

## 6. Afterwards

```bash
sudo systemctl restart craftpilot-backend
```

---

## What it will cost

Measured, not estimated: **8,970 of 11,611 chunks carry a `source_language`
tag** and therefore need a translation call. Roughly 400 tokens in and 400 out
per chunk puts the run in the region of **7 million tokens**. Price that against
your Infomaniak plan before step 5 — step 4's pilot gives you a real per-module
figure to multiply.

Note the breadcrumb fix *reduces* calls relative to the old behaviour: headings
are now translated once per page instead of once per chunk.

## One thing worth investigating first

The language distribution across those 8,970 chunks contains entries that are
almost certainly **misdetections** by `py3langid`: Latin (77), Esperanto (5),
Walloon (3), Breton (5), Occitan (4). Short or heavily technical French text
is the usual cause. Those chunks are being translated from a language they are
not in, which wastes calls and may be corrupting the text.

Spot-check a handful before the full run — if they are French being
"translated" from Latin, fixing detection first would both cut the bill and
improve quality. This is a separate defect and is not fixed.
