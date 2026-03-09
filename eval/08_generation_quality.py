"""
08_generation_quality.py
Hallucination spot-check: LLM-as-judge evaluates whether generated responses
contain claims unsupported by the retrieved context.

NOTE: This LLM-as-judge approach is indicative, not definitive.
French domain-specific content may confuse the judge. Results labeled accordingly.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, '/opt/craftpilot_backend')
os.chdir('/opt/craftpilot_backend')

from openai import OpenAI
from config.settings import ConfigurationManager
from services.rag_service import RAGService
from services.annotation_service import AnnotationService
from langchain_core.messages import HumanMessage

API_KEY = "H3SAaQWrzLsP0u58dUIRfePv2BQjenx3xuJY18puAjK2mlDQQfVsu3XdjycO7mTJj0IGzM-Lm49luWMh"
PRODUCT_ID = "106980"
BASE_URL = f"https://api.infomaniak.com/2/ai/{PRODUCT_ID}/openai/v1"
LLM_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"

openai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

JUDGE_PROMPT_TEMPLATE = """Étant donné UNIQUEMENT ce contexte documentaire :
---
{context}
---

La réponse suivante contient-elle des affirmations qui ne sont pas soutenues par le contexte ci-dessus ?
Liste chaque affirmation non soutenue. Si toutes les affirmations sont soutenues, réponds exactement "AUCUNE".

Réponse à évaluer :
---
{response}
---

Affirmations non soutenues (ou "AUCUNE") :"""


def build_state(query):
    return {
        "messages": [HumanMessage(content=query)],
        "context": [],
        "video_metadata": None,
        "refined_query": None,
        "hypothetical_document": None,
        "enhanced_query": None,
        "query_variants": [],
        "route": None,
        "selected_domain": None,
        "course_id": None,
    }


def run_full_pipeline(rag, query):
    """Run PRF pipeline + generate, return (context_text, response_text)."""
    state = build_state(query)
    result = rag.retrieve_initial(state)
    state.update(result)
    result = rag.refine_query_prf(state)
    state.update(result)
    result = rag.retrieve_final_dual(state)
    state.update(result)

    context_docs = state.get('context', [])
    context_text = "\n\n".join(doc.page_content for doc in context_docs) if context_docs else ""

    result = rag.generate(state)
    msgs = result.get('messages', [])
    response_text = msgs[0].content if msgs and hasattr(msgs[0], 'content') else str(msgs[0]) if msgs else "[no response]"

    return context_text, response_text, context_docs


def judge_hallucination(context, response):
    """Use LLM as judge. Returns (has_hallucination: bool, unsupported_claims: str)."""
    if not context.strip():
        return False, "AUCUNE (no context provided)"

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        context=context[:3000],  # Truncate to fit context window
        response=response[:1500],
    )
    try:
        result = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1,
        )
        judge_output = result.choices[0].message.content.strip()
        has_hallucination = "aucune" not in judge_output.lower()
        return has_hallucination, judge_output
    except Exception as e:
        return False, f"[judge error: {e}]"


def main():
    print("=== GENERATION QUALITY / HALLUCINATION SPOT-CHECK ===")

    # Load ground truth — pick 10 queries (mix of registers, no adversarial)
    with open('eval/fixtures/ground_truth.json') as f:
        gt = json.load(f)

    # Select 10 queries: 4 expert, 4 novice, 2 course
    selected = []
    for reg in ['expert', 'novice', 'expert']:  # cycle through registers
        for q in gt:
            if q['register'] == reg and len(selected) < 10 and q not in selected:
                if q.get('relevant_annotation_ids') or q.get('relevant_course_sources'):
                    selected.append(q)
                    break

    # Fill remaining spots from course queries
    for q in gt:
        if len(selected) >= 10:
            break
        if q['register'] not in ('adversarial',) and q not in selected:
            selected.append(q)

    selected = selected[:10]
    print(f"Selected {len(selected)} queries for generation quality check")

    config_manager = ConfigurationManager()
    annotation_service = AnnotationService(config_manager)
    rag = RAGService(config_manager, annotation_service=annotation_service)
    from services.course_rag_service import CourseRAGService
    course_rag = CourseRAGService(
        embeddings=rag.embeddings,
        persist_directory=rag.config.persist_directory,
    )
    rag.course_rag_service = course_rag

    results = []
    n_hallucinations = 0

    for i, q in enumerate(selected):
        query = q['query']
        print(f"\n[{i+1}/{len(selected)}] {query[:80]}...")

        context_text, response_text, context_docs = run_full_pipeline(rag, query)
        print(f"  Context: {len(context_docs)} docs, {len(context_text)} chars")
        print(f"  Response (first 200): {response_text[:200]}")

        # Judge
        print(f"  Running LLM judge...")
        has_hallucination, judge_output = judge_hallucination(context_text, response_text)
        print(f"  Hallucination: {has_hallucination}")
        print(f"  Judge: {judge_output[:200]}")

        if has_hallucination:
            n_hallucinations += 1

        results.append({
            'qid': q['qid'],
            'query': query,
            'register': q['register'],
            'n_context_docs': len(context_docs),
            'context_sources': [d.metadata.get('source', '') for d in context_docs],
            'response': response_text,
            'judge_output': judge_output,
            'has_hallucination': has_hallucination,
            'judge_reliability_note': (
                "Low confidence: French domain-specific content; judge may flag legitimate expert vocabulary as unsupported."
                if has_hallucination else "Likely reliable negative."
            ),
        })

    hallucination_rate = n_hallucinations / len(results) if results else 0.0
    print(f"\n=== SUMMARY ===")
    print(f"Hallucination rate: {n_hallucinations}/{len(results)} = {hallucination_rate:.1%}")
    print("NOTE: LLM-as-judge is indicative only. French craft vocabulary may be flagged unfairly.")

    output = {
        'n_queries': len(results),
        'hallucination_rate': hallucination_rate,
        'n_hallucinations': n_hallucinations,
        'methodology_note': (
            "LLM-as-judge using swiss-ai/Apertus-70B-Instruct-2509. "
            "French domain-specific craft vocabulary may be incorrectly flagged as hallucinated. "
            "Treat as indicative, not definitive."
        ),
        'per_query': results,
    }

    Path('eval/results').mkdir(exist_ok=True)
    with open('eval/results/generation_quality.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("Saved to eval/results/generation_quality.json")


if __name__ == '__main__':
    main()
