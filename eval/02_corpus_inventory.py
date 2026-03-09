"""
Step 2: Corpus inventory — collect and print statistics about ChromaDB collections.
Saves results/corpus_stats.json.
"""

import os
import sys
import json
import logging
from collections import defaultdict

os.chdir('/opt/craftpilot_backend')
sys.path.insert(0, '/opt/craftpilot_backend')

logging.basicConfig(level=logging.WARNING)

import chromadb

RESULTS_DIR = '/opt/craftpilot_backend/eval/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_chromadb_client():
    return chromadb.PersistentClient(path='./chroma_langchain_db')


def collect_annotation_stats(client):
    """Stats for moodle_assistant_collection."""
    col = client.get_collection('moodle_assistant_collection')
    count = col.count()

    results = col.get(include=['documents', 'metadatas'])
    docs = results['documents']
    metas = results['metadatas']

    per_annotation = []
    total_duration = 0.0
    for doc, meta in zip(docs, metas):
        chars = len(doc)
        tokens_approx = chars // 4
        dur = float(meta.get('duration', 0.0))
        total_duration += dur
        per_annotation.append({
            'annotation_id': meta.get('annotation_id'),
            'video_filename': meta.get('video_filename'),
            'duration_s': round(dur, 2),
            'transcription_chars': chars,
            'transcription_tokens_approx': tokens_approx,
            'source': meta.get('source'),
        })

    per_annotation.sort(key=lambda x: x['annotation_id'])

    return {
        'total_annotation_docs': count,
        'total_annotated_duration_s': round(total_duration, 2),
        'per_annotation': per_annotation,
    }


def collect_course_stats(client):
    """Stats for all course_* collections."""
    col_names = client.list_collections()
    course_cols = [n for n in col_names if n.startswith('course_')]

    course_stats = []
    all_module_types = set()
    total_chunks = 0

    for col_name in sorted(course_cols):
        col = client.get_collection(col_name)
        count = col.count()
        total_chunks += count

        if count == 0:
            course_stats.append({
                'collection': col_name,
                'doc_count': 0,
                'avg_chunk_len': 0,
                'module_types': [],
            })
            continue

        results = col.get(include=['documents', 'metadatas'])
        docs = results['documents']
        metas = results['metadatas']

        total_len = sum(len(d) for d in docs)
        avg_len = total_len // count if count > 0 else 0

        module_types = list(set(m.get('module_type', 'unknown') for m in metas))
        for mt in module_types:
            all_module_types.add(mt)

        course_stats.append({
            'collection': col_name,
            'doc_count': count,
            'avg_chunk_len': avg_len,
            'module_types': module_types,
        })

    # Sort by doc_count desc for top 5
    course_stats_sorted = sorted(course_stats, key=lambda x: x['doc_count'], reverse=True)
    top_5 = course_stats_sorted[:5]

    # Craft-relevant
    craft_cols = {}
    for cs in course_stats:
        if cs['collection'] in ('course_83', 'course_98'):
            craft_cols[cs['collection']] = cs

    return {
        'total_course_collections': len(course_cols),
        'total_course_chunks': total_chunks,
        'all_unique_module_types': sorted(list(all_module_types)),
        'top_5_by_size': top_5,
        'craft_relevant_courses': craft_cols,
        'per_course': course_stats,
    }


def print_annotation_table(ann_stats):
    print("\n=== Annotation Documents (moodle_assistant_collection) ===")
    print(f"Total docs: {ann_stats['total_annotation_docs']}")
    print(f"Total annotated duration: {ann_stats['total_annotated_duration_s']}s")
    print()
    header = f"{'Ann ID':<8} {'Video Filename':<48} {'Dur(s)':<8} {'Chars':<8} {'~Tokens':<8}"
    print(header)
    print("-" * len(header))
    for ann in ann_stats['per_annotation']:
        print(f"{ann['annotation_id']:<8} {ann['video_filename']:<48} "
              f"{ann['duration_s']:<8} {ann['transcription_chars']:<8} "
              f"{ann['transcription_tokens_approx']:<8}")


def print_course_table(course_stats):
    print("\n=== Course Collections ===")
    print(f"Total course collections: {course_stats['total_course_collections']}")
    print(f"Total chunks across all courses: {course_stats['total_course_chunks']}")
    print(f"\nUnique module types: {', '.join(course_stats['all_unique_module_types'])}")

    print("\n--- Top 5 Courses by Size ---")
    header = f"{'Collection':<20} {'Doc Count':<12} {'Avg Chunk Len':<16} {'Module Types'}"
    print(header)
    print("-" * 80)
    for cs in course_stats['top_5_by_size']:
        mts = ', '.join(cs['module_types'])
        print(f"{cs['collection']:<20} {cs['doc_count']:<12} {cs['avg_chunk_len']:<16} {mts}")

    print("\n--- Craft-Relevant Courses ---")
    for col_name, cs in course_stats['craft_relevant_courses'].items():
        print(f"  {col_name}: {cs['doc_count']} docs, avg chunk len {cs['avg_chunk_len']} chars, "
              f"module types: {', '.join(cs['module_types'])}")


def main():
    print("=" * 60)
    print("STEP 2: Corpus Inventory")
    print("=" * 60)

    client = get_chromadb_client()

    print("\nCollecting annotation stats...")
    ann_stats = collect_annotation_stats(client)

    print("Collecting course stats...")
    course_stats = collect_course_stats(client)

    print_annotation_table(ann_stats)
    print_course_table(course_stats)

    # Save to JSON
    output = {
        'annotations': ann_stats,
        'courses': course_stats,
    }
    out_path = os.path.join(RESULTS_DIR, 'corpus_stats.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_path}")
    print("=" * 60)
    print("COMPLETE: Corpus inventory done.")
    print("=" * 60)


if __name__ == '__main__':
    main()
