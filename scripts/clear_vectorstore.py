"""
Clear the ChromaDB vector store and the local annotation seed SQLite.

Run from /opt/craftpilot_backend:
    python scripts/clear_vectorstore.py

Use this when you want a completely clean slate before real elicitations
are collected via the videoelicit plugin.  After clearing:
  - The vector store will be empty (0 documents).
  - The local SQLite seed DB will be wiped so the pipeline's auto-sync on
    the next restart does NOT re-populate ChromaDB with stale test data.
  - Real elicitations will arrive via the /api/ingest-annotation hook fired
    by the videoelicit backend whenever a transcription completes.
"""

import sqlite3
import sys
import os
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ConfigurationManager
from services.rag_service import RAGService

SQLITE_PATH = "chroma_langchain_db/elicitations_db/annotations.db"


def clear_chroma(rag: RAGService) -> None:
    data_before = rag.get_vector_store_data()
    count_before = len(data_before.get("ids", []))
    print(f"ChromaDB documents before: {count_before}")

    if count_before > 0:
        rag.remove_documents("all")
        count_after = len(rag.get_vector_store_data().get("ids", []))
        print(f"ChromaDB documents after:  {count_after}")
    else:
        print("ChromaDB already empty.")


def clear_sqlite_seeds(db_path: str) -> None:
    """Truncate the local seed SQLite so auto-sync has nothing to replay."""
    if not Path(db_path).exists():
        print(f"SQLite not found at {db_path} — skipping.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables = [row[0] for row in cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    for table in tables:
        cursor.execute(f"DELETE FROM {table}")
        print(f"  Cleared table '{table}'")

    conn.commit()
    conn.close()
    print(f"SQLite seed DB cleared ({db_path})")


def main():
    print("=== CraftPilot — full data reset ===\n")

    config_manager = ConfigurationManager()
    rag = RAGService(config_manager=config_manager)

    clear_chroma(rag)
    print()
    clear_sqlite_seeds(SQLITE_PATH)
    print("\nDone. Restart the craftpilot backend to apply.")


if __name__ == "__main__":
    main()
