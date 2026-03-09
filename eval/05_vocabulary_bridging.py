"""
05_vocabulary_bridging.py
Cosine similarity analysis: raw vs PRF-refined query vs relevant document embedding.
Quantifies the vocabulary bridging effect of PRF.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, '/opt/craftpilot_backend')
os.chdir('/opt/craftpilot_backend')

from openai import OpenAI

API_KEY = "REDACTED_OLD_INFOMANIAK_KEY"
PRODUCT_ID = "106980"
BASE_URL = f"https://api.infomaniak.com/2/ai/{PRODUCT_ID}/openai/v1"
EMB_MODEL = "bge_multilingual_gemma2"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def embed(texts):
    """Embed a list of texts, returning list of vectors."""
    response = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [r.embedding for r in response.data]


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    # Load Config B results
    with open('eval/results/config_b_results.json') as f:
        config_b = json.load(f)

    # Load ground truth
    with open('eval/fixtures/ground_truth.json') as f:
        gt = json.load(f)
    gt_map = {q['qid']: q for q in gt}

    # Load relevant doc texts from ChromaDB
    import chromadb
    chroma_client = chromadb.PersistentClient(path='./chroma_langchain_db')
    ann_col = chroma_client.get_collection('moodle_assistant_collection')

    # Get all annotation docs
    all_docs = ann_col.get(limit=100)
    doc_by_ann_id = {}
    for doc_text, meta in zip(all_docs['documents'], all_docs['metadatas']):
        aid = meta.get('annotation_id')
        if aid is not None:
            doc_by_ann_id[aid] = doc_text

    print(f"Loaded {len(doc_by_ann_id)} annotation docs from ChromaDB")

    results = []
    for entry in config_b['per_query']:
        qid = entry['qid']
        register = entry['register']
        if register == 'adversarial':
            continue
        raw_query = entry['query']
        refined_query = entry.get('refined_query') or raw_query
        relevant_ids = entry.get('relevant_ids', [])

        if not relevant_ids:
            continue

        # Get relevant doc texts
        rel_doc_texts = [doc_by_ann_id[aid] for aid in relevant_ids if aid in doc_by_ann_id]
        if not rel_doc_texts:
            print(f"  WARNING: no doc texts found for {qid} relevant_ids={relevant_ids}")
            continue

        print(f"  Embedding [{qid}] (register={register})...")

        # Embed all at once: raw_query, refined_query, relevant_docs
        texts_to_embed = [raw_query, refined_query] + rel_doc_texts
        try:
            vectors = embed(texts_to_embed)
        except Exception as e:
            print(f"    Embedding failed: {e}")
            continue

        raw_vec = vectors[0]
        refined_vec = vectors[1]
        doc_vecs = vectors[2:]

        # Compute cosine sims
        raw_sims = [cosine_sim(raw_vec, dv) for dv in doc_vecs]
        refined_sims = [cosine_sim(refined_vec, dv) for dv in doc_vecs]

        mean_raw = float(np.mean(raw_sims))
        mean_refined = float(np.mean(refined_sims))
        delta = mean_refined - mean_raw

        results.append({
            'qid': qid,
            'register': register,
            'raw_query': raw_query,
            'refined_query': refined_query,
            'mean_cos_raw': mean_raw,
            'mean_cos_refined': mean_refined,
            'delta_cos': delta,
            'relevant_ids': relevant_ids,
        })
        print(f"    cos_raw={mean_raw:.4f}, cos_refined={mean_refined:.4f}, Δ={delta:+.4f}")

    # Compute aggregate stats by register
    from collections import defaultdict
    by_register = defaultdict(list)
    for r in results:
        by_register[r['register']].append(r)
    by_register['all'] = results

    aggregate = {}
    for reg, items in by_register.items():
        if not items:
            continue
        raw_sims = [i['mean_cos_raw'] for i in items]
        ref_sims = [i['mean_cos_refined'] for i in items]
        deltas = [i['delta_cos'] for i in items]

        stat = {
            'n': len(items),
            'mean_cos_raw': float(np.mean(raw_sims)),
            'std_cos_raw': float(np.std(raw_sims)),
            'mean_cos_refined': float(np.mean(ref_sims)),
            'std_cos_refined': float(np.std(ref_sims)),
            'mean_delta': float(np.mean(deltas)),
            'std_delta': float(np.std(deltas)),
        }

        # Statistical test
        if len(items) >= 5:
            try:
                from scipy import stats
                # Paired Wilcoxon signed-rank test
                stat_result = stats.wilcoxon(ref_sims, raw_sims, alternative='greater')
                stat['test'] = 'wilcoxon'
                stat['p_value'] = float(stat_result.pvalue)
            except Exception as e:
                try:
                    from scipy import stats
                    stat_result = stats.ttest_rel(ref_sims, raw_sims, alternative='greater')
                    stat['test'] = 'ttest_rel'
                    stat['p_value'] = float(stat_result.pvalue)
                except Exception as e2:
                    stat['test'] = 'none'
                    stat['p_value'] = None
        else:
            stat['test'] = 'n_too_small'
            stat['p_value'] = None

        aggregate[reg] = stat

    output = {
        'per_query': results,
        'aggregate': aggregate,
    }

    Path('eval/results').mkdir(exist_ok=True)
    with open('eval/results/vocabulary_bridging.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print table
    print("\n=== VOCABULARY BRIDGING RESULTS ===")
    print(f"{'Query Type':<18} | {'cos(raw, doc)':<16} | {'cos(refined, doc)':<19} | {'Δcos':>8} | {'p-value':>10}")
    print("-" * 80)
    for reg in ['novice', 'expert', 'all']:
        if reg in aggregate:
            a = aggregate[reg]
            p = f"{a['p_value']:.4f}" if a['p_value'] is not None else "N/A"
            print(f"{reg:<18} | {a['mean_cos_raw']:.4f} ± {a['std_cos_raw']:.4f} | {a['mean_cos_refined']:.4f} ± {a['std_cos_refined']:.4f} | {a['mean_delta']:>+.4f} | {p:>10}")

    print(f"\nSaved to eval/results/vocabulary_bridging.json")


if __name__ == '__main__':
    main()
