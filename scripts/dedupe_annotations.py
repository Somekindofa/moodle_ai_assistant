#!/usr/bin/env python
"""Collapse duplicated annotation documents down to one per annotation.

Why this exists
---------------
``add_documents`` used to let langchain_chroma mint a random UUID per call, so
the startup annotation sync appended a fresh copy of the whole corpus on every
restart. The store reached ~24 copies of each of the 16 real annotations.

Duplicates never broke the silo — every copy carries identical access labels —
but they crowd retrieval: a top-5 annotation search could return five copies of
the same clip, pushing out genuinely different material.

``stable_document_id`` now makes ingestion idempotent, so this is a one-off
repair of the backlog. It is safe to re-run.

Which copy is kept
------------------
The one whose id already matches the stable id, if present; otherwise the last
one written (Chroma preserves insertion order, and the newest copy reflects the
most recent labels and translation). Every other copy of that annotation is
deleted.

Usage
-----
    python scripts/dedupe_annotations.py --dry-run
    python scripts/dedupe_annotations.py --apply
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402

from services.rag_service import stable_document_id  # noqa: E402

CHROMA_PATH = "/opt/craftpilot_backend/chroma_langchain_db"
COLLECTION = "moodle_assistant_collection"
BATCH = 200


class _Doc:
    """Minimal stand-in so stable_document_id can score a stored record."""

    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    col = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
    data = col.get(include=["metadatas", "documents"])

    total = len(data["ids"])
    groups = defaultdict(list)
    for doc_id, meta, text in zip(data["ids"], data["metadatas"], data["documents"]):
        groups[stable_document_id(_Doc(text, meta))].append((doc_id, meta))

    keep, drop = [], []
    for stable_id, members in groups.items():
        preferred = next((m for m in members if m[0] == stable_id), members[-1])
        keep.append(preferred)
        drop.extend(m[0] for m in members if m[0] != preferred[0])

    print(f"Documents stored      : {total}")
    print(f"Distinct annotations  : {len(groups)}")
    print(f"Duplicates to delete  : {len(drop)}")
    print()
    per_craft = Counter(meta.get("craft") for _, meta in keep)
    print("Surviving documents by craft:")
    for craft, n in sorted(per_craft.items(), key=lambda kv: str(kv[0])):
        print(f"  {craft:28} {n}")

    # Never let a craft disappear entirely — that would mean losing content,
    # not just duplicates.
    before_crafts = {m.get("craft") for m in data["metadatas"]}
    after_crafts = set(per_craft)
    if before_crafts != after_crafts:
        print(f"\nABORT: crafts would be lost: {before_crafts - after_crafts}")
        return 1

    if not drop:
        print("\nNothing to do — no duplicates.")
        return 0

    if args.dry_run:
        print(f"\nDry run — would delete {len(drop)} documents, keeping {len(keep)}.")
        return 0

    for start in range(0, len(drop), BATCH):
        col.delete(ids=drop[start:start + BATCH])

    after = col.get(include=["metadatas"])
    print(f"\nDeleted {len(drop)} duplicates. Documents now: {len(after['ids'])}")

    remaining = Counter(m.get("craft") for m in after["metadatas"])
    print("Final state by craft:")
    for craft, n in sorted(remaining.items(), key=lambda kv: str(kv[0])):
        print(f"  {craft:28} {n}")

    lv_open = sum(1 for m in after["metadatas"]
                  if m.get("craft") == "lv_rivetage_maletterie" and m.get("open_access"))
    print(f"\nSilo check — LV docs still marked open_access: {lv_open} (must be 0)")
    return 0 if lv_open == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
