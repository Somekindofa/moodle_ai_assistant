"""
Step 4: Retrieval ablation evaluation.
Config A: Baseline similarity_search
Config B: Full PRF pipeline
Config C: HyDE

Saves results/config_{a,b,c}_results.json and prints summary table.
"""

import os
import sys
import json
import time
import logging

os.chdir('/opt/craftpilot_backend')
sys.path.insert(0, '/opt/craftpilot_backend')

logging.basicConfig(level=logging.WARNING)

from config.settings import ConfigurationManager
from services.rag_service import RAGService
from langchain_core.messages import HumanMessage

RESULTS_DIR = '/opt/craftpilot_backend/eval/results'
FIXTURES_DIR = '/opt/craftpilot_backend/eval/fixtures'
os.makedirs(RESULTS_DIR, exist_ok=True)

K_VALUES = [1, 3, 5]


# ── Metrics ───────────────────────────────────────────────────────────────────

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


# ── Retrieval helpers ─────────────────────────────────────────────────────────

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


def run_config_a(rag, query):
    """Config A: Direct similarity_search on annotation collection."""
    docs = rag.similarity_search(query, k=5)
    return docs, None  # (docs, refined_query)


def run_config_b(rag, query):
    """Config B: PRF — retrieve_initial → refine_query_prf → retrieve_final_dual."""
    state = make_state(query)

    s1 = rag.retrieve_initial(state)
    initial_context = s1.get('context', [])

    state2 = {**state, **s1}
    s2 = rag.refine_query_prf(state2)
    refined_query = s2.get('refined_query', query)

    state3 = {**state2, **s2}
    s3 = rag.retrieve_final_dual(state3)
    final_docs = s3.get('context', [])

    return final_docs, refined_query, initial_context


def run_config_c(rag, query):
    """Config C: HyDE — generate_hypothetical_document → retrieve_with_hyde."""
    state = make_state(query)

    s1 = rag.generate_hypothetical_document(state)
    state2 = {**state, **s1}
    s2 = rag.retrieve_with_hyde(state2)
    docs = s2.get('context', [])

    return docs, None


def extract_ids_from_docs(docs):
    """
    Extract matching identifiers from retrieved documents.
    Returns list of annotation_ids (int) for annotation docs,
    or source strings for course docs.
    """
    ids = []
    for doc in docs:
        meta = doc.metadata
        doc_type = meta.get('type', '')
        if doc_type == 'video_annotation':
            ann_id = meta.get('annotation_id')
            if ann_id is not None:
                ids.append(int(ann_id))
        elif doc_type == 'course_content':
            source = meta.get('source', '')
            ids.append(source)
        else:
            # Fallback: try annotation_id, then source
            ann_id = meta.get('annotation_id')
            if ann_id is not None:
                ids.append(int(ann_id))
            else:
                source = meta.get('source', '')
                if source:
                    ids.append(source)
    return ids


def get_relevant_ids(item):
    """
    Get the relevant IDs for a ground truth item.
    For annotations: list of annotation_ids (int).
    For course content: list of source strings.
    For adversarial: empty.
    """
    if item['source'] == 'annotation':
        return [int(x) for x in item.get('relevant_annotation_ids', [])]
    elif item['source'] == 'course_content':
        return item.get('relevant_course_sources', [])
    else:
        return []


def compute_metrics(query_results):
    """Compute aggregate metrics from per-query results (excluding adversarial)."""
    non_adv = [r for r in query_results if r['register'] != 'adversarial']

    if not non_adv:
        return {}

    metrics = {}
    for k in K_VALUES:
        p_vals = [r[f'p_at_{k}'] for r in non_adv]
        r_vals = [r[f'r_at_{k}'] for r in non_adv]
        metrics[f'p_at_{k}'] = round(sum(p_vals) / len(p_vals), 4)
        metrics[f'r_at_{k}'] = round(sum(r_vals) / len(r_vals), 4)

    map_vals = [r['map'] for r in non_adv]
    metrics['map'] = round(sum(map_vals) / len(map_vals), 4)
    metrics['n_queries'] = len(non_adv)

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def run_config(config_name, rag, ground_truth, retrieval_fn):
    """Run one retrieval config over all queries."""
    print(f"\n--- Running Config {config_name} ---")
    query_results = []

    for i, item in enumerate(ground_truth):
        qid = item['qid']
        query = item['query']
        register = item['register']
        relevant_ids = get_relevant_ids(item)

        print(f"  [{i+1}/{len(ground_truth)}] {qid}: {query[:60]}...", end='', flush=True)

        try:
            extra = retrieval_fn(rag, query)
            if config_name == 'B':
                docs, refined_query, initial_context = extra
                initial_context_sources = [d.metadata.get('source') for d in initial_context]
            else:
                docs, refined_query = extra
                initial_context_sources = []
        except Exception as e:
            print(f" ERROR: {e}")
            query_results.append({
                'qid': qid,
                'query': query,
                'register': register,
                'error': str(e),
                'retrieved_ids': [],
                'relevant_ids': relevant_ids,
                'p_at_1': 0.0, 'p_at_3': 0.0, 'p_at_5': 0.0,
                'r_at_1': 0.0, 'r_at_3': 0.0, 'r_at_5': 0.0,
                'map': 0.0,
                'refined_query': None,
                'initial_context_sources': [],
            })
            continue

        retrieved_ids = extract_ids_from_docs(docs)
        retrieved_sources = [d.metadata.get('source') for d in docs]

        if register == 'adversarial':
            result = {
                'qid': qid,
                'query': query,
                'register': register,
                'retrieved_ids': retrieved_ids,
                'retrieved_sources': retrieved_sources,
                'relevant_ids': [],
                'p_at_1': 0.0, 'p_at_3': 0.0, 'p_at_5': 0.0,
                'r_at_1': 0.0, 'r_at_3': 0.0, 'r_at_5': 0.0,
                'map': 0.0,
                'refined_query': refined_query,
                'initial_context_sources': initial_context_sources,
            }
            print(f" (adv, {len(docs)} docs)")
        else:
            p1 = precision_at_k(retrieved_ids, relevant_ids, 1)
            p3 = precision_at_k(retrieved_ids, relevant_ids, 3)
            p5 = precision_at_k(retrieved_ids, relevant_ids, 5)
            r1 = recall_at_k(retrieved_ids, relevant_ids, 1)
            r3 = recall_at_k(retrieved_ids, relevant_ids, 3)
            r5 = recall_at_k(retrieved_ids, relevant_ids, 5)
            ap = average_precision(retrieved_ids, relevant_ids)

            result = {
                'qid': qid,
                'query': query,
                'register': register,
                'retrieved_ids': retrieved_ids,
                'retrieved_sources': retrieved_sources,
                'relevant_ids': relevant_ids,
                'p_at_1': round(p1, 4),
                'p_at_3': round(p3, 4),
                'p_at_5': round(p5, 4),
                'r_at_1': round(r1, 4),
                'r_at_3': round(r3, 4),
                'r_at_5': round(r5, 4),
                'map': round(ap, 4),
                'refined_query': refined_query,
                'initial_context_sources': initial_context_sources,
            }
            print(f" P@1={p1:.2f} P@3={p3:.2f} MAP={ap:.2f}")

        query_results.append(result)
        time.sleep(0.5)  # small pause between LLM calls

    aggregate = compute_metrics(query_results)
    return {'per_query': query_results, 'aggregate': aggregate}


def main():
    print("=" * 70)
    print("STEP 4: Retrieval Ablation Evaluation")
    print("=" * 70)

    # Load ground truth
    gt_path = os.path.join(FIXTURES_DIR, 'ground_truth.json')
    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    print(f"Loaded {len(ground_truth)} queries from {gt_path}")

    # Initialize RAG service
    config_manager = ConfigurationManager()
    rag = RAGService(config_manager)

    # Config A: Baseline
    def config_a_fn(rag, query):
        docs, rq = run_config_a(rag, query)
        return docs, rq

    results_a = run_config('A', rag, ground_truth, config_a_fn)
    with open(os.path.join(RESULTS_DIR, 'config_a_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_a, f, ensure_ascii=False, indent=2)
    print("Saved config_a_results.json")

    # Config B: PRF
    def config_b_fn(rag, query):
        return run_config_b(rag, query)

    results_b = run_config('B', rag, ground_truth, config_b_fn)
    with open(os.path.join(RESULTS_DIR, 'config_b_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_b, f, ensure_ascii=False, indent=2)
    print("Saved config_b_results.json")

    # Config C: HyDE
    def config_c_fn(rag, query):
        docs, rq = run_config_c(rag, query)
        return docs, rq

    results_c = run_config('C', rag, ground_truth, config_c_fn)
    with open(os.path.join(RESULTS_DIR, 'config_c_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_c, f, ensure_ascii=False, indent=2)
    print("Saved config_c_results.json")

    # Print summary table
    def fmt(v):
        return f"{v:.4f}"

    agg_a = results_a['aggregate']
    agg_b = results_b['aggregate']
    agg_c = results_c['aggregate']

    print("\n")
    print("=" * 80)
    print("RETRIEVAL ABLATION SUMMARY")
    print("=" * 80)
    header = f"{'Config':<12} {'P@1':<8} {'P@3':<8} {'P@5':<8} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'MAP':<8}"
    print(header)
    print("-" * 80)

    def row(name, agg):
        return (f"{name:<12} "
                f"{fmt(agg.get('p_at_1',0)):<8} "
                f"{fmt(agg.get('p_at_3',0)):<8} "
                f"{fmt(agg.get('p_at_5',0)):<8} "
                f"{fmt(agg.get('r_at_1',0)):<8} "
                f"{fmt(agg.get('r_at_3',0)):<8} "
                f"{fmt(agg.get('r_at_5',0)):<8} "
                f"{fmt(agg.get('map',0)):<8}")

    print(row("A (raw)", agg_a))
    print(row("B (PRF)", agg_b))
    print(row("C (HyDE)", agg_c))
    print("=" * 80)
    print(f"(n={agg_a.get('n_queries',0)} non-adversarial queries)")

    # PRF compliance log: show refined queries
    print("\n--- PRF Refined Queries (Config B) ---")
    for r in results_b['per_query']:
        if r.get('refined_query') and r['register'] != 'adversarial':
            print(f"  [{r['qid']}]")
            print(f"    Original:  {r['query']}")
            print(f"    Refined:   {r['refined_query']}")

    print("\n" + "=" * 70)
    print("COMPLETE: Retrieval evaluation done.")
    print("=" * 70)


if __name__ == '__main__':
    main()
