"""
10_cross_lingual_corpus_eval.py
Cross-lingual CORPUS eval: validates ingestion-time translation (see
services/translation_service.py, annotation_service.py, course_rag_service.py)
end to end, in a scratch ChromaDB collection — never touches production data.

Two things this checks together, since neither alone proves the feature works:

1. New capability: content verbalized/authored in a non-French language
   (English, Greek — see eval/fixtures/xling_annotations_seed.json /
   xling_course_chunks_seed.json), once ingested through the new translation
   path, retrieves correctly for a FRENCH query (Config A, raw similarity
   search).
2. Composition: the existing query-side translation feature
   (detect_and_translate_query, eval/09_cross_lingual_eval.py's Config D)
   still retrieves correctly once the corpus itself is mixed-language, not
   just each feature working in isolation.

Ground truth: eval/fixtures/ground_truth_corpus_xling.json.

Not a retrofit of 09_cross_lingual_eval.py — that script tests query-side
translation only, against the existing (assumed French-only) production
corpus. This script tests the corpus side, against fixtures seeded fresh
into an isolated collection.
"""

import os
import sys
import json
import logging

os.chdir('/opt/craftpilot_backend')
sys.path.insert(0, '/opt/craftpilot_backend')

logging.basicConfig(level=logging.WARNING)

from config.settings import ConfigurationManager, AppConfig, RAGConfig
from services.rag_service import RAGService
from services.annotation_service import AnnotationService
from services.course_rag_service import CourseRAGService
from langchain_core.messages import HumanMessage

RESULTS_DIR = '/opt/craftpilot_backend/eval/results'
FIXTURES_DIR = '/opt/craftpilot_backend/eval/fixtures'
SCRATCH_PERSIST_DIR = os.path.join(RESULTS_DIR, '_scratch_chroma_xling')
os.makedirs(RESULTS_DIR, exist_ok=True)

K_VALUES = [1, 3, 5]


# ── Metrics (identical to 04_evaluate_retrieval.py / 09_cross_lingual_eval.py,
# copy-pasted per this repo's existing per-script eval convention) ─────────

def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    if k == 0:
        return 0.0
    return len(set(top_k) & set(relevant_ids)) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    if not relevant_ids:
        return 0.0
    return len(set(top_k) & set(relevant_ids)) / len(relevant_ids)


def average_precision(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    hits, score = 0, 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in set(relevant_ids):
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant_ids)


def compute_metrics(query_results):
    if not query_results:
        return {}
    metrics = {}
    for k in K_VALUES:
        p_vals = [r[f'p_at_{k}'] for r in query_results]
        r_vals = [r[f'r_at_{k}'] for r in query_results]
        metrics[f'p_at_{k}'] = round(sum(p_vals) / len(p_vals), 4)
        metrics[f'r_at_{k}'] = round(sum(r_vals) / len(r_vals), 4)
    map_vals = [r['map'] for r in query_results]
    metrics['map'] = round(sum(map_vals) / len(map_vals), 4)
    metrics['n_queries'] = len(query_results)
    return metrics


def make_state(query):
    return {
        'messages': [HumanMessage(content=query)],
        'context': [],
        'video_metadata': None,
        'refined_query': None,
        'hypothetical_document': None,
        'enhanced_query': None,
        'query_variants': [],
        'route': None,
        'selected_domain': None,
        'course_id': None,
    }


def extract_annotation_ids(docs):
    return [int(d.metadata['annotation_id']) for d in docs if d.metadata.get('annotation_id') is not None]


def extract_course_sources(docs):
    return [d.metadata.get('source') for d in docs if d.metadata.get('type') == 'course_content']


# ── Seeding ──────────────────────────────────────────────────────────────

def seed_scratch_corpus(rag, annotation_service, course_rag_service):
    print("--- Seeding scratch corpus ---")

    with open(os.path.join(FIXTURES_DIR, 'xling_annotations_seed.json'), encoding='utf-8') as f:
        annotations = json.load(f)
    for ann in annotations:
        docs = annotation_service.annotation_to_documents(ann, use_extended=False)
        rag.add_documents(docs)
        translated = docs[0].page_content != ann['transcription']
        print(f"  seeded annotation {ann['annotation_id']} ({ann['language']}) — "
              f"{'translated' if translated else 'unchanged'}")

    with open(os.path.join(FIXTURES_DIR, 'xling_course_chunks_seed.json'), encoding='utf-8') as f:
        chunks = json.load(f)
    for c in chunks:
        n = course_rag_service.ingest_module(
            course_id=c['course_id'], module_id=c['module_id'], module_type=c['module_type'],
            module_name=c['module_name'], section_name=c['section_name'], content_html=c['content_html'],
        )
        print(f"  seeded course {c['course_id']}/module {c['module_id']} — {n} chunk(s)")


def spot_check_translations(annotation_service):
    """Manual sanity check: print translated text next to originals. No
    automated garbage-translation guard exists (see design spec) — this is
    the only catch for it before rollout."""
    print("\n--- Spot check: translated vs original ---")
    with open(os.path.join(FIXTURES_DIR, 'xling_annotations_seed.json'), encoding='utf-8') as f:
        annotations = json.load(f)
    for ann in annotations:
        docs = annotation_service.annotation_to_documents(ann, use_extended=False)
        if docs and docs[0].page_content != ann['transcription']:
            print(f"  [{ann['annotation_id']}] ({ann['language']}) original:  {ann['transcription']}")
            print(f"  [{ann['annotation_id']}]        translated: {docs[0].page_content}\n")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 10: Cross-Lingual CORPUS Eval (scratch collection)")
    print("=" * 70)

    scratch_config = AppConfig(rag=RAGConfig(persist_directory=SCRATCH_PERSIST_DIR))
    config_manager = ConfigurationManager(config=scratch_config)

    rag = RAGService(config_manager)
    annotation_service = AnnotationService(config_manager)
    course_rag_service = CourseRAGService(
        embeddings=rag.embeddings, persist_directory=SCRATCH_PERSIST_DIR, config_manager=config_manager,
    )

    seed_scratch_corpus(rag, annotation_service, course_rag_service)
    spot_check_translations(annotation_service)

    with open(os.path.join(FIXTURES_DIR, 'ground_truth_corpus_xling.json'), encoding='utf-8') as f:
        ground_truth = json.load(f)

    print("\n--- Running queries against scratch corpus ---")
    query_results = []
    for item in ground_truth:
        qid, query, source = item['qid'], item['query'], item['source']

        if item['query_language'] == 'fr':
            if source == 'annotation':
                docs = rag.similarity_search(query, k=5)
                retrieved = extract_annotation_ids(docs)
                relevant = item.get('relevant_annotation_ids', [])
            else:
                course_id = item['relevant_course_sources'][0].split('_module_')[0].replace('course_', '')
                docs = course_rag_service.similarity_search(query, course_id=course_id, k=5)
                retrieved = [d.metadata.get('source') for d in docs]
                relevant = item.get('relevant_course_sources', [])
        else:
            # Config D: existing query-side translation path, against the
            # now-mixed-language scratch corpus.
            state = make_state(query)
            s0 = rag.detect_and_translate_query(state)
            state0 = {**state, **s0}
            s1 = rag.retrieve_initial(state0)
            docs = s1.get('context', [])
            retrieved = extract_annotation_ids(docs) if source == 'annotation' else extract_course_sources(docs)
            relevant = item.get('relevant_annotation_ids', []) if source == 'annotation' else item.get('relevant_course_sources', [])

        ap = average_precision(retrieved, relevant)
        result = {
            'qid': qid, 'query': query, 'source': source, 'query_language': item['query_language'],
            'retrieved': retrieved, 'relevant': relevant,
            'p_at_1': round(precision_at_k(retrieved, relevant, 1), 4),
            'p_at_3': round(precision_at_k(retrieved, relevant, 3), 4),
            'p_at_5': round(precision_at_k(retrieved, relevant, 5), 4),
            'r_at_1': round(recall_at_k(retrieved, relevant, 1), 4),
            'r_at_3': round(recall_at_k(retrieved, relevant, 3), 4),
            'r_at_5': round(recall_at_k(retrieved, relevant, 5), 4),
            'map': round(ap, 4),
        }
        query_results.append(result)
        print(f"  [{qid}] MAP={ap:.2f}  retrieved={retrieved}  relevant={relevant}")

    aggregate = compute_metrics(query_results)
    out = {'per_query': query_results, 'aggregate': aggregate}
    out_path = os.path.join(RESULTS_DIR, '10_cross_lingual_corpus_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"MAP={aggregate.get('map')}  n={aggregate.get('n_queries')}")
    print(f"Saved {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
