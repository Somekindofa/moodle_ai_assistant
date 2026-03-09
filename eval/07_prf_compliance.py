"""
07_prf_compliance.py
PRF Term Grounding Rate: checks that the LLM-refined query uses vocabulary
attested in the top-3 first-pass documents, not invented terms.
"""

import sys
import os
import json
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/opt/craftpilot_backend')
os.chdir('/opt/craftpilot_backend')

# French stopwords (simple list)
FR_STOPWORDS = {
    'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'en', 'au', 'aux',
    'pour', 'par', 'sur', 'dans', 'avec', 'est', 'sont', 'que', 'qui', 'quoi',
    'comment', 'pourquoi', 'quand', 'où', 'quel', 'quelle', 'quels', 'quelles',
    'il', 'elle', 'ils', 'elles', 'je', 'tu', 'nous', 'vous', 'on', 'me', 'se',
    'lui', 'leur', 'leurs', 'ce', 'cet', 'cette', 'ces', 'mon', 'ma', 'mes',
    'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'votre', 'leur',
    'pas', 'ne', 'plus', 'très', 'bien', 'aussi', 'même', 'tout', 'tous',
    'après', 'avant', 'pendant', 'lorsque', 'quand', 'si', 'car', 'mais', 'ou',
    'donc', 'or', 'ni', 'à', 'the', 'lors', 'puis', 'via', 'lors',
    'afin', 'ainsi', 'cela', 'celui', 'celle', 'ceux', 'celles',
    'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'vouloir', 'pouvoir',
    'faut', 'doit', 'peut', 'peut', 'avoir',
}


def extract_technical_terms(text):
    """Extract candidate technical terms (words > 4 chars, not stopwords)."""
    # Tokenize: keep only alphabetic tokens (including accented French)
    tokens = re.findall(r"[a-zA-ZÀ-ÿ\-]{5,}", text.lower())
    terms = [t for t in tokens if t not in FR_STOPWORDS and len(t) >= 5]
    return list(set(terms))


def term_in_docs(term, doc_texts):
    """Check if term appears in any of the doc texts (case-insensitive)."""
    term_lower = term.lower()
    for doc in doc_texts:
        if term_lower in doc.lower():
            return True
    return False


def main():
    print("=== PRF COMPLIANCE CHECK ===")

    # Load Config B results
    with open('eval/results/config_b_results.json') as f:
        config_b = json.load(f)

    # Load annotation docs from ChromaDB to get doc texts for initial context
    import chromadb
    chroma_client = chromadb.PersistentClient(path='./chroma_langchain_db')
    ann_col = chroma_client.get_collection('moodle_assistant_collection')
    all_docs = ann_col.get(limit=100)
    doc_by_source = {}
    for doc_text, meta in zip(all_docs['documents'], all_docs['metadatas']):
        src = meta.get('source', '')
        doc_by_source[src] = doc_text

    compliance_results = []
    all_grounding_rates = []

    for entry in config_b['per_query']:
        if entry['register'] == 'adversarial':
            continue
        raw_query = entry['query']
        refined_query = entry.get('refined_query') or raw_query
        initial_sources = entry.get('initial_context_sources', [])

        # Get doc texts for top-3 initial context docs
        top3_docs = [doc_by_source[s] for s in initial_sources[:3] if s in doc_by_source]

        if not top3_docs:
            print(f"  [{entry['qid']}] No initial context docs found, skipping")
            continue

        # Extract technical terms from refined query
        technical_terms = extract_technical_terms(refined_query)

        if not technical_terms:
            print(f"  [{entry['qid']}] No technical terms extracted")
            continue

        # Check grounding
        grounded = [t for t in technical_terms if term_in_docs(t, top3_docs)]
        not_grounded = [t for t in technical_terms if not term_in_docs(t, top3_docs)]
        grounding_rate = len(grounded) / len(technical_terms) if technical_terms else 1.0
        all_grounding_rates.append(grounding_rate)

        compliance_results.append({
            'qid': entry['qid'],
            'register': entry['register'],
            'raw_query': raw_query,
            'refined_query': refined_query,
            'technical_terms': technical_terms,
            'grounded_terms': grounded,
            'ungrounded_terms': not_grounded,
            'grounding_rate': grounding_rate,
            'n_terms': len(technical_terms),
            'n_grounded': len(grounded),
        })

        flag = "⚠️" if grounding_rate < 0.90 else "✓"
        print(f"  {flag} [{entry['qid']}] rate={grounding_rate:.2%} ({len(grounded)}/{len(technical_terms)} terms grounded)")
        if not_grounded:
            print(f"      Ungrounded: {not_grounded[:5]}")

    # Aggregate
    import numpy as np
    mean_rate = float(np.mean(all_grounding_rates)) if all_grounding_rates else 0.0
    n_below_90 = sum(1 for r in all_grounding_rates if r < 0.90)
    n_below_70 = sum(1 for r in all_grounding_rates if r < 0.70)

    print(f"\n=== SUMMARY ===")
    print(f"Mean grounding rate: {mean_rate:.2%}")
    print(f"Queries below 90% threshold: {n_below_90}/{len(all_grounding_rates)}")
    print(f"Queries below 70% threshold: {n_below_70}/{len(all_grounding_rates)}")

    output = {
        'per_query': compliance_results,
        'aggregate': {
            'mean_grounding_rate': mean_rate,
            'std_grounding_rate': float(np.std(all_grounding_rates)),
            'n_queries': len(compliance_results),
            'n_below_90pct': n_below_90,
            'n_below_70pct': n_below_70,
        }
    }

    Path('eval/results').mkdir(exist_ok=True)
    with open('eval/results/prf_compliance.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("Saved to eval/results/prf_compliance.json")


if __name__ == '__main__':
    main()
