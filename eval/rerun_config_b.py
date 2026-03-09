"""Re-run Config B with the corrected RAG service (post guardrail fix)."""
import os, sys, json, time, logging
os.chdir('/opt/craftpilot_backend')
sys.path.insert(0, '/opt/craftpilot_backend')
logging.disable(logging.CRITICAL)

from config.settings import ConfigurationManager
from services.rag_service import RAGService
from services.annotation_service import AnnotationService
from services.course_rag_service import CourseRAGService
from langchain_core.messages import HumanMessage


def make_state(query):
    return {
        'messages': [HumanMessage(content=query)],
        'context': [], 'video_metadata': None, 'refined_query': None,
        'hypothetical_document': None, 'enhanced_query': None,
        'query_variants': [], 'route': None, 'selected_domain': None, 'course_id': None,
    }


def precision_at_k(r, rel, k):
    return len(set(r[:k]) & set(rel)) / k if k else 0.0


def recall_at_k(r, rel, k):
    return len(set(r[:k]) & set(rel)) / len(rel) if rel else 0.0


def average_precision(r, rel):
    if not rel:
        return 0.0
    rel_set = set(rel)
    hits, score = 0, 0.0
    for i, rid in enumerate(r):
        if rid in rel_set:
            hits += 1
            score += hits / (i + 1)
    return score / len(rel)


def extract_ids(docs):
    ids = []
    for doc in docs:
        m = doc.metadata
        if m.get('type') == 'video_annotation' and m.get('annotation_id') is not None:
            ids.append(int(m['annotation_id']))
        elif m.get('type') == 'course_content':
            ids.append(m.get('source', ''))
        else:
            ann_id = m.get('annotation_id')
            if ann_id is not None:
                ids.append(int(ann_id))
    return ids


def main():
    config_manager = ConfigurationManager()
    annotation_service = AnnotationService(config_manager)
    rag = RAGService(config_manager, annotation_service=annotation_service)
    course_rag = CourseRAGService(
        embeddings=rag.embeddings,
        persist_directory=rag.config.persist_directory,
    )
    rag.course_rag_service = course_rag

    with open('eval/fixtures/ground_truth.json') as f:
        ground_truth = json.load(f)

    print(f"Config B re-run: {len(ground_truth)} queries")
    results = []

    for i, item in enumerate(ground_truth):
        qid = item['qid']
        query = item['query']
        register = item['register']

        if item['source'] == 'annotation':
            relevant_ids = [int(x) for x in item.get('relevant_annotation_ids', [])]
        elif item['source'] == 'course_content':
            relevant_ids = item.get('relevant_course_sources', [])
        else:
            relevant_ids = []

        print(f"[{i+1}/{len(ground_truth)}] {qid}", end=' ', flush=True)

        try:
            state = make_state(query)
            s1 = rag.retrieve_initial(state)
            initial_sources = [d.metadata.get('source') for d in s1.get('context', [])]
            state.update(s1)
            s2 = rag.refine_query_prf(state)
            refined_query = s2.get('refined_query', query)
            state.update(s2)
            s3 = rag.retrieve_final_dual(state)
            final_docs = s3.get('context', [])
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'qid': qid, 'query': query, 'register': register,
                'error': str(e), 'retrieved_ids': [], 'relevant_ids': relevant_ids,
                'p_at_1': 0.0, 'p_at_3': 0.0, 'p_at_5': 0.0,
                'r_at_1': 0.0, 'r_at_3': 0.0, 'r_at_5': 0.0, 'map': 0.0,
                'refined_query': None, 'initial_context_sources': [],
            })
            continue

        retrieved_ids = extract_ids(final_docs)
        retrieved_sources = [d.metadata.get('source') for d in final_docs]

        if register == 'adversarial':
            r = {
                'qid': qid, 'query': query, 'register': register,
                'retrieved_ids': retrieved_ids, 'retrieved_sources': retrieved_sources,
                'relevant_ids': [], 'p_at_1': 0.0, 'p_at_3': 0.0, 'p_at_5': 0.0,
                'r_at_1': 0.0, 'r_at_3': 0.0, 'r_at_5': 0.0, 'map': 0.0,
                'refined_query': refined_query, 'initial_context_sources': initial_sources,
            }
            print(f"adv {len(final_docs)}docs")
        else:
            p1 = precision_at_k(retrieved_ids, relevant_ids, 1)
            p3 = precision_at_k(retrieved_ids, relevant_ids, 3)
            p5 = precision_at_k(retrieved_ids, relevant_ids, 5)
            r1 = recall_at_k(retrieved_ids, relevant_ids, 1)
            r3 = recall_at_k(retrieved_ids, relevant_ids, 3)
            r5 = recall_at_k(retrieved_ids, relevant_ids, 5)
            ap = average_precision(retrieved_ids, relevant_ids)
            r = {
                'qid': qid, 'query': query, 'register': register,
                'retrieved_ids': retrieved_ids, 'retrieved_sources': retrieved_sources,
                'relevant_ids': relevant_ids,
                'p_at_1': round(p1, 4), 'p_at_3': round(p3, 4), 'p_at_5': round(p5, 4),
                'r_at_1': round(r1, 4), 'r_at_3': round(r3, 4), 'r_at_5': round(r5, 4),
                'map': round(ap, 4), 'refined_query': refined_query,
                'initial_context_sources': initial_sources,
            }
            print(f"P@3={p3:.2f} MAP={ap:.2f} n_ids={len(retrieved_ids)}")

        results.append(r)
        time.sleep(0.3)

    non_adv = [r for r in results if r['register'] != 'adversarial']
    agg = {}
    for k in [1, 3, 5]:
        agg[f'p_at_{k}'] = round(sum(r[f'p_at_{k}'] for r in non_adv) / len(non_adv), 4)
        agg[f'r_at_{k}'] = round(sum(r[f'r_at_{k}'] for r in non_adv) / len(non_adv), 4)
    agg['map'] = round(sum(r['map'] for r in non_adv) / len(non_adv), 4)
    agg['n_queries'] = len(non_adv)

    output = {'per_query': results, 'aggregate': agg}
    with open('eval/results/config_b_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n=== Config B Aggregate ===")
    for k in [1, 3, 5]:
        print(f"  P@{k}={agg[f'p_at_{k}']:.4f}  R@{k}={agg[f'r_at_{k}']:.4f}")
    print(f"  MAP={agg['map']:.4f}  n={agg['n_queries']}")
    print("Saved eval/results/config_b_results.json")


if __name__ == '__main__':
    main()
