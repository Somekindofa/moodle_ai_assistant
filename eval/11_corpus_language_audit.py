"""
11_corpus_language_audit.py
Read-only diagnostic: samples page_content from every existing course_*
ChromaDB collection (and the annotation collection) through py3langid,
tallying the non-French fraction per collection.

Prerequisite for the course-content backfill decision (see design spec) —
if the existing corpus is already ~100% French, bulk backfill tooling is
speculative work not worth building. Run this BEFORE deciding.

Does not modify anything — no writes, no re-ingestion, no translation.
"""

import os
import sys
import logging

os.chdir('/opt/craftpilot_backend')
sys.path.insert(0, '/opt/craftpilot_backend')

logging.basicConfig(level=logging.WARNING)

from config.settings import ConfigurationManager
from services.rag_service import RAGService
from services.course_rag_service import CourseRAGService
from services.translation_service import load_langid

SAMPLE_SIZE = 20  # docs sampled per collection — cheap, not exhaustive


def audit_collection(collection, langid_identifier, name):
    data = collection.get(limit=SAMPLE_SIZE)
    texts = data.get('documents') or []
    if not texts:
        return {'collection': name, 'sampled': 0, 'non_french': 0, 'fraction': 0.0}

    non_french = 0
    for text in texts:
        if not text:
            continue
        lang, confidence = langid_identifier.classify(text)
        if lang != 'fr' and confidence >= 0.5:
            non_french += 1

    return {
        'collection': name,
        'sampled': len(texts),
        'non_french': non_french,
        'fraction': round(non_french / len(texts), 3) if texts else 0.0,
    }


def main():
    print("=" * 70)
    print("STEP 11: Corpus Language Audit (read-only)")
    print("=" * 70)

    config_manager = ConfigurationManager()
    rag = RAGService(config_manager)
    course_rag = CourseRAGService(embeddings=rag.embeddings, persist_directory=rag.config.persist_directory)

    langid_identifier = load_langid()
    if langid_identifier is None:
        print("py3langid unavailable — cannot audit. Aborting.")
        return

    results = []

    print("\n--- Annotation collection ---")
    ann_result = audit_collection(rag.vector_store, langid_identifier, 'annotations')
    results.append(ann_result)
    print(f"  {ann_result['non_french']}/{ann_result['sampled']} sampled docs non-French "
          f"({ann_result['fraction']:.0%})")

    print("\n--- Course collections ---")
    course_ids = course_rag._enumerate_populated_courses()
    print(f"  found {len(course_ids)} populated course collections")
    for cid in course_ids:
        collection = course_rag._get_collection(cid)
        result = audit_collection(collection, langid_identifier, f'course_{cid}')
        results.append(result)
        if result['non_french'] > 0:
            print(f"  course_{cid}: {result['non_french']}/{result['sampled']} sampled docs "
                  f"non-French ({result['fraction']:.0%})")

    total_sampled = sum(r['sampled'] for r in results)
    total_non_french = sum(r['non_french'] for r in results)
    overall_fraction = round(total_non_french / total_sampled, 3) if total_sampled else 0.0

    print("\n" + "=" * 70)
    print(f"OVERALL: {total_non_french}/{total_sampled} sampled docs non-French "
          f"({overall_fraction:.0%}) across {len(results)} collections")
    if overall_fraction < 0.02:
        print("Corpus appears essentially all-French — course-content backfill "
              "tooling is likely not worth building; new content is covered "
              "going forward by the ingestion-time translation feature.")
    else:
        print("Meaningful non-French fraction found — worth building a course-"
              "content backfill mechanism (see design spec, deferred pending "
              "this audit).")
    print("=" * 70)


if __name__ == '__main__':
    main()
