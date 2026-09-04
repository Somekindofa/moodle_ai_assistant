#!/usr/bin/env python
"""Apply the craft->cohort safety net to already-indexed annotation documents.

Why this exists
---------------
The 319 annotation documents in ``moodle_assistant_collection`` were bulk
loaded before the silo write-path worked. They all carry
``project_name='unknown'``, so they cannot be matched back to a project and
re-synced the normal way. Their only usable discriminator is ``craft``.

This script brings those legacy documents in line with CRAFT_COHORT_MAP, the
same map the live ingestion path uses (services/annotation_service.py), so that
a document's access does not depend on when it happened to be indexed.

It is idempotent: re-running it after a re-sync re-applies the same labels.

Usage
-----
    python scripts/apply_craft_silo.py --dry-run     # report only, no writes
    python scripts/apply_craft_silo.py --apply       # perform the update

Verification is built in: with --dry-run it prints the exact before/after
counts per craft, and after --apply it re-reads the collection through
build_cohort_filter() to prove the documents are unreachable to a user who is
not in the mapped cohort.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402

from services.annotation_service import CRAFT_COHORT_MAP  # noqa: E402
from services.rag_service import build_cohort_filter  # noqa: E402

CHROMA_PATH = "/opt/craftpilot_backend/chroma_langchain_db"
COLLECTION = "moodle_assistant_collection"
BATCH = 100


def _target_state(meta):
    """Return (cohort_id, open_access) this document should have, or None if
    the craft is unmapped and it should be left alone."""
    cohortid = CRAFT_COHORT_MAP.get(meta.get("craft") or "")
    if cohortid is None:
        return None
    return cohortid, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="report only")
    group.add_argument("--apply", action="store_true", help="write changes")
    args = parser.parse_args()

    if not CRAFT_COHORT_MAP:
        print("CRAFT_COHORT_MAP is empty — nothing to enforce. Is CRAFT_COHORT_MAP "
              "set in .env? Refusing to run rather than silently doing nothing.")
        return 1

    print(f"CRAFT_COHORT_MAP: {CRAFT_COHORT_MAP}\n")

    col = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
    data = col.get(include=["metadatas"])

    before = Counter((m.get("craft"), m.get("cohort_id"), m.get("open_access"))
                     for m in data["metadatas"])
    print("Current state (craft, cohort_id, open_access) -> count:")
    for key, n in sorted(before.items(), key=lambda kv: str(kv[0])):
        print(f"  {key} -> {n}")

    ids, metas = [], []
    for doc_id, meta in zip(data["ids"], data["metadatas"]):
        target = _target_state(meta)
        if target is None:
            continue
        cohortid, open_access = target
        if meta.get("cohort_id") == cohortid and meta.get("open_access") is open_access:
            continue  # already correct
        new_meta = dict(meta)
        new_meta["cohort_id"] = cohortid
        new_meta["open_access"] = open_access
        ids.append(doc_id)
        metas.append(new_meta)

    print(f"\nDocuments needing update: {len(ids)}")
    if not ids:
        print("Nothing to do — already consistent.")
    elif args.dry_run:
        sample = Counter(m.get("craft") for m in metas)
        print("Would set (cohort_id, open_access=False) for:")
        for craft, n in sample.items():
            print(f"  {craft} -> cohort {CRAFT_COHORT_MAP[craft]}  ({n} docs)")
        print("\nDry run — no changes written.")
        return 0
    else:
        for start in range(0, len(ids), BATCH):
            col.update(ids=ids[start:start + BATCH], metadatas=metas[start:start + BATCH])
        print(f"Updated {len(ids)} documents.")

    # Prove the result through the real filter rather than trusting the write.
    print("\nVerification via build_cohort_filter():")
    checks = [("user with no cohorts", [])]
    checks += [(f"user in cohort {cid} only", [cid]) for cid in sorted(set(CRAFT_COHORT_MAP.values()))]
    checks.append(("user in every mapped cohort", sorted(set(CRAFT_COHORT_MAP.values()))))

    for label, cohorts in checks:
        got = col.get(where=build_cohort_filter(cohorts), include=["metadatas"])
        per_craft = Counter(m.get("craft") for m in got["metadatas"]
                            if m.get("craft") in CRAFT_COHORT_MAP)
        visible = dict(per_craft) or "none"
        print(f"  {label:34} -> mapped-craft docs visible: {visible}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
