"""
Backfill: translate existing course-content chunks that predate the
ingestion-translation feature (services/course_rag_service.py) to French,
in place.

Why this is needed: the 79% non-French figure from
eval/11_corpus_language_audit.py is content that was indexed BEFORE
ingestion-time translation existed — new content is covered automatically
going forward (translation runs on every module save), but this already-
indexed content will only get translated if a teacher happens to re-save
it. This script does it as a one-off batch job instead.

Safe to interrupt and re-run: chunks already translated (tagged
`source_language` in metadata) are skipped, so a partial run just picks up
where it left off. Can take a long time for the full corpus (roughly 9,000
real translation calls across ~11.5k chunks per the audit script's numbers)
— that's why this is a standalone script, not an HTTP endpoint; no request
would stay open that long.

Run from /opt/craftpilot_backend:
    PYTHONNOUSERSITE=1 /root/miniconda3/envs/moodle_backend/bin/python \
        scripts/backfill_course_translations.py
"""

import os
import sys
import time
import logging

os.chdir('/opt/craftpilot_backend')
sys.path.insert(0, '/opt/craftpilot_backend')

logging.basicConfig(level=logging.WARNING)

from config.settings import ConfigurationManager
from services.rag_service import RAGService
from services.course_rag_service import CourseRAGService


def main():
    print("=" * 70)
    print("Course Content Translation Backfill")
    print("=" * 70)

    config_manager = ConfigurationManager()
    rag = RAGService(config_manager)
    course_rag = CourseRAGService(
        embeddings=rag.embeddings, persist_directory=rag.config.persist_directory,
        config_manager=config_manager,
    )
    rag_config = config_manager.get_config().rag

    if not rag_config.enable_ingestion_translation:
        print("ENABLE_INGESTION_TRANSLATION is false — nothing to do (this "
              "script uses the same kill-switch as live ingestion).")
        return

    if course_rag._translation_llm is None:
        print("Translation LLM failed to initialize — check INFOMANIAK_API_KEY "
              "/ INFOMANIAK_PRODUCT_ID. Aborting.")
        return

    course_ids = course_rag._enumerate_populated_courses()
    print(f"Found {len(course_ids)} populated course collections\n")

    totals = {"total": 0, "already_tagged": 0, "translated": 0, "unchanged_french": 0, "failed": 0}
    PROGRESS_EVERY = 10

    def make_progress_printer(course_label):
        def _on_progress(idx, total, stats):
            if idx % PROGRESS_EVERY == 0 or idx == total:
                print(f"    course_{course_label}: {idx}/{total} examined — "
                      f"translated={stats['translated']} failed={stats['failed']} "
                      f"already_done={stats['already_tagged']} french={stats['unchanged_french']}",
                      flush=True)
        return _on_progress

    for i, cid in enumerate(course_ids):
        print(f"[{i+1}/{len(course_ids)}] course_{cid}: starting...", flush=True)
        start = time.time()
        stats = course_rag.backfill_translations(
            cid, rag_config, throttle_seconds=0.3, on_progress=make_progress_printer(cid),
        )
        elapsed = time.time() - start

        for k in totals:
            totals[k] += stats[k]

        if stats["translated"] or stats["failed"]:
            print(f"[{i+1}/{len(course_ids)}] course_{cid}: "
                  f"{stats['translated']} translated, {stats['failed']} failed, "
                  f"{stats['already_tagged']} already done, "
                  f"{stats['unchanged_french']} already French "
                  f"({elapsed:.1f}s)")
        else:
            print(f"[{i+1}/{len(course_ids)}] course_{cid}: nothing to do ({elapsed:.1f}s)")

    print("\n" + "=" * 70)
    print(f"TOTAL: {totals['translated']} translated, {totals['failed']} failed, "
          f"{totals['already_tagged']} already done, "
          f"{totals['unchanged_french']} already French, "
          f"{totals['total']} chunks examined across {len(course_ids)} collections")
    if totals["failed"]:
        print(f"\n{totals['failed']} chunks failed to translate (API errors) — "
              "safe to re-run this script, they were left untagged and will be retried.")
    print("=" * 70)


if __name__ == '__main__':
    main()
