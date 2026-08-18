"""
09_cross_lingual_eval.py
Cross-lingual retrieval eval: same 33 non-adversarial queries as
fixtures/ground_truth.json, hand-translated to English
(fixtures/ground_truth_en.json), same ground-truth relevant_ids.

Runs Config A (raw similarity_search) and Config B (PRF: retrieve_initial ->
refine_query_prf -> retrieve_final_dual) against the English queries, using
the exact same RAGService methods as 04_evaluate_retrieval.py. Compares
against the existing French results already on disk
(eval/results/config_a_results.json / config_b_results.json) to isolate the
effect of query language from the effect of the PRF step itself.

Also logs the language of `refined_query` for each English input, since the
PRF prompt template is hardcoded in French (see rag_service.py
refine_query_prf) and may silently translate the query.
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


# ── Metrics (identical to 04_evaluate_retrieval.py) ────────────────────────

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


# ── Retrieval helpers ─────────────────────────────────────────────────────

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
    docs = rag.similarity_search(query, k=5)
    return docs, None


def run_config_b(rag, query):
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


def extract_ids_from_docs(docs):
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
            ann_id = meta.get('annotation_id')
            if ann_id is not None:
                ids.append(int(ann_id))
            else:
                source = meta.get('source', '')
                if source:
                    ids.append(source)
    return ids


def get_relevant_ids(item):
    if item['source'] == 'annotation':
        return [int(x) for x in item.get('relevant_annotation_ids', [])]
    elif item['source'] == 'course_content':
        return item.get('relevant_course_sources', [])
    else:
        return []


def compute_metrics(query_results):
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


# crude language guess: count French-only accented chars / stopwords vs
# plain-ASCII English function words. Not a real language ID, just enough to
# flag "did PRF actually translate this" in a summary column.
FR_MARKERS = set("àâäéèêëîïôöùûüç") | {
    ' le ', ' la ', ' les ', ' du ', ' des ', ' une ', ' un ', ' et ',
    ' est ', ' pour ', ' avec ', ' dans ', ' que ', ' qui ',
}
EN_MARKERS = {
    ' the ', ' is ', ' are ', ' for ', ' with ', ' in ', ' that ',
    ' which ', ' how ', ' what ', ' why ',
}


def guess_language(text):
    if text is None:
        return 'unknown'
    lower = f" {text.lower()} "
    fr_score = sum(1 for ch in lower if ch in FR_MARKERS)
    fr_score += sum(1 for m in FR_MARKERS if isinstance(m, str) and len(m) > 1 and m in lower)
    en_score = sum(1 for m in EN_MARKERS if m in lower)
    if fr_score > en_score:
        return 'fr'
    elif en_score > fr_score:
        return 'en'
    return 'ambiguous'


# ── Main run loop ───────────────────────────────────────────────────────────

def run_config(config_name, rag, ground_truth, retrieval_fn):
    print(f"\n--- Running Config {config_name} (English queries) ---")
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
                'qid': qid, 'query': query, 'register': register,
                'error': str(e), 'retrieved_ids': [], 'relevant_ids': relevant_ids,
                'p_at_1': 0.0, 'p_at_3': 0.0, 'p_at_5': 0.0,
                'r_at_1': 0.0, 'r_at_3': 0.0, 'r_at_5': 0.0, 'map': 0.0,
                'refined_query': None, 'refined_query_lang': None,
                'initial_context_sources': [],
            })
            continue

        retrieved_ids = extract_ids_from_docs(docs)
        retrieved_sources = [d.metadata.get('source') for d in docs]
        refined_lang = guess_language(refined_query) if config_name == 'B' else None

        if register == 'adversarial':
            result = {
                'qid': qid, 'query': query, 'register': register,
                'retrieved_ids': retrieved_ids, 'retrieved_sources': retrieved_sources,
                'relevant_ids': [],
                'p_at_1': 0.0, 'p_at_3': 0.0, 'p_at_5': 0.0,
                'r_at_1': 0.0, 'r_at_3': 0.0, 'r_at_5': 0.0, 'map': 0.0,
                'refined_query': refined_query, 'refined_query_lang': refined_lang,
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
                'qid': qid, 'query': query, 'register': register,
                'retrieved_ids': retrieved_ids, 'retrieved_sources': retrieved_sources,
                'relevant_ids': relevant_ids,
                'p_at_1': round(p1, 4), 'p_at_3': round(p3, 4), 'p_at_5': round(p5, 4),
                'r_at_1': round(r1, 4), 'r_at_3': round(r3, 4), 'r_at_5': round(r5, 4),
                'map': round(ap, 4),
                'refined_query': refined_query, 'refined_query_lang': refined_lang,
                'initial_context_sources': initial_context_sources,
            }
            lang_tag = f" [refined->{refined_lang}]" if refined_lang else ""
            print(f" P@1={p1:.2f} P@3={p3:.2f} MAP={ap:.2f}{lang_tag}")

        query_results.append(result)
        time.sleep(0.5)

    aggregate = compute_metrics(query_results)
    return {'per_query': query_results, 'aggregate': aggregate}


def load_fr_baseline(config_letter):
    path = os.path.join(RESULTS_DIR, f'config_{config_letter}_results.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)['aggregate']


def main():
    print("=" * 70)
    print("STEP 9: Cross-Lingual Retrieval Eval (EN queries vs FR corpus)")
    print("=" * 70)

    gt_path = os.path.join(FIXTURES_DIR, 'ground_truth_en.json')
    with open(gt_path, encoding='utf-8') as f:
        ground_truth = json.load(f)
    print(f"Loaded {len(ground_truth)} English queries from {gt_path}")

    config_manager = ConfigurationManager()
    rag = RAGService(config_manager)

    results_a = run_config('A', rag, ground_truth, lambda rag, q: run_config_a(rag, q))
    with open(os.path.join(RESULTS_DIR, 'config_a_en_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_a, f, ensure_ascii=False, indent=2)
    print("Saved config_a_en_results.json")

    results_b = run_config('B', rag, ground_truth, lambda rag, q: run_config_b(rag, q))
    with open(os.path.join(RESULTS_DIR, 'config_b_en_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_b, f, ensure_ascii=False, indent=2)
    print("Saved config_b_en_results.json")

    fr_a = load_fr_baseline('a')
    fr_b = load_fr_baseline('b')
    en_a = results_a['aggregate']
    en_b = results_b['aggregate']

    def fmt(v):
        return f"{v:.4f}"

    def row(name, agg):
        if not agg:
            return f"{name:<16} (no French baseline found — run 04_evaluate_retrieval.py first)"
        return (f"{name:<16} "
                f"{fmt(agg.get('p_at_1',0)):<8} "
                f"{fmt(agg.get('p_at_3',0)):<8} "
                f"{fmt(agg.get('p_at_5',0)):<8} "
                f"{fmt(agg.get('r_at_1',0)):<8} "
                f"{fmt(agg.get('r_at_3',0)):<8} "
                f"{fmt(agg.get('r_at_5',0)):<8} "
                f"{fmt(agg.get('map',0)):<8}")

    print("\n")
    print("=" * 88)
    print("CROSS-LINGUAL SUMMARY  (FR = baseline queries, EN = translated queries, same relevant_ids)")
    print("=" * 88)
    header = f"{'Config':<16} {'P@1':<8} {'P@3':<8} {'P@5':<8} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'MAP':<8}"
    print(header)
    print("-" * 88)
    print(row("FR - A (raw)", fr_a))
    print(row("EN - A (raw)", en_a))
    print("-" * 88)
    print(row("FR - B (PRF)", fr_b))
    print(row("EN - B (PRF)", en_b))
    print("=" * 88)
    print(f"(n={en_a.get('n_queries',0)} non-adversarial queries per language)")

    # Delta: does PRF help or hurt EN queries specifically, and how much
    # worse is EN than FR for each config (the actual cross-lingual gap)?
    if fr_a and fr_b:
        print("\n--- Deltas ---")
        print(f"MAP  FR: raw={fr_a['map']:.4f}  PRF={fr_b['map']:.4f}  (PRF effect: {fr_b['map']-fr_a['map']:+.4f})")
        print(f"MAP  EN: raw={en_a['map']:.4f}  PRF={en_b['map']:.4f}  (PRF effect: {en_b['map']-en_a['map']:+.4f})")
        print(f"Cross-lingual gap (raw):  FR-EN = {fr_a['map']-en_a['map']:+.4f}")
        print(f"Cross-lingual gap (PRF):  FR-EN = {fr_b['map']-en_b['map']:+.4f}")

    # Language of refined queries — did PRF translate EN input into FR?
    print("\n--- PRF refined_query language (Config B, English input) ---")
    lang_counts = {}
    for r in results_b['per_query']:
        if r['register'] == 'adversarial':
            continue
        lang_counts[r['refined_query_lang']] = lang_counts.get(r['refined_query_lang'], 0) + 1
    print(f"  {lang_counts}")
    print("  Sample:")
    for r in results_b['per_query'][:6]:
        if r['register'] == 'adversarial':
            continue
        print(f"    [{r['qid']}] EN: {r['query']}")
        print(f"      -> refined ({r['refined_query_lang']}): {r['refined_query']}")

    print("\n" + "=" * 70)
    print("COMPLETE: Cross-lingual eval done.")
    print("=" * 70)


if __name__ == '__main__':
    main()
