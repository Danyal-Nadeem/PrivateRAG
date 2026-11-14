"""Document ingestion script.

Usage:
    python -m src.retrieval.ingest [--recreate]

Reads data/documents/metadata.json and all .md files, chunks each document,
embeds chunks, and upserts into Qdrant (local persistent mode).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a module from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.chunking import chunk_text
from src.retrieval.qdrant_service import ensure_collection, get_collection_info, upsert_chunks

DOCS_DIR = PROJECT_ROOT / "data" / "documents"
METADATA_FILE = DOCS_DIR / "metadata.json"


def main(recreate: bool = False) -> None:
    print("=" * 60)
    print("PrivateRAG — Document Ingestion")
    print("=" * 60)

    if not METADATA_FILE.exists():
        print(f"ERROR: metadata.json not found at {METADATA_FILE}")
        sys.exit(1)

    with open(METADATA_FILE, encoding="utf-8") as f:
        metadata: list[dict] = json.load(f)

    print(f"Found {len(metadata)} documents in metadata.json")
    print(f"Ensuring Qdrant collection (recreate={recreate}) …")
    ensure_collection(recreate=recreate)

    total_chunks = 0
    missing_docs = 0

    for entry in metadata:
        doc_id = entry["doc_id"]
        access_level = entry["access_level"]
        title = entry.get("title", doc_id)
        md_path = DOCS_DIR / f"{doc_id}.md"

        if not md_path.exists():
            print(f"  WARN: {md_path.name} not found — skipping")
            missing_docs += 1
            continue

        text = md_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, doc_id=doc_id, access_level=access_level, title=title)

        if not chunks:
            print(f"  WARN: {doc_id} produced 0 chunks — skipping")
            continue

        upsert_chunks(chunks)
        total_chunks += len(chunks)
        print(f"  {doc_id:<12}  level={access_level:<15}  chunks={len(chunks)}")

    print("-" * 60)
    info = get_collection_info()
    print(f"Ingestion complete.")
    print(f"  Documents processed : {len(metadata) - missing_docs}")
    print(f"  Documents skipped   : {missing_docs}")
    print(f"  Total chunks upserted: {total_chunks}")
    print(f"  Collection points   : {info['points_count']}")
    print(f"  Collection status   : {info['status']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before ingesting.",
    )
    args = parser.parse_args()
    main(recreate=args.recreate)
