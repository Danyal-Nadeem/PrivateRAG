"""Qdrant persistent client and collection management.

Storage path: qdrant_storage/  (gitignored, relative to project root).
Collection : documents
Vector size: 384 (all-MiniLM-L6-v2)
Distance   : Cosine
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from src.retrieval.rbac import build_rbac_filter
from src.retrieval import embedder as _embedder

COLLECTION_NAME = "documents"
VECTOR_SIZE = 384
_STORAGE_PATH = str(Path(__file__).resolve().parent.parent.parent / "qdrant_storage")

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=_STORAGE_PATH)
    return _client


def _make_point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic UUID5 from doc_id + chunk_index (never uses hash())."""
    name = f"{doc_id}_{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


def ensure_collection(recreate: bool = False) -> None:
    """Create the Qdrant collection if it does not exist.

    Parameters
    ----------
    recreate:
        If True, delete any existing collection first (used by ingest script).
    """
    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if recreate and COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        existing.discard(COLLECTION_NAME)

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def upsert_chunks(chunks: list[dict]) -> None:
    """Embed *chunks* and upsert them into the Qdrant collection.

    Each chunk dict must have keys: ``text``, ``doc_id``, ``access_level``,
    ``chunk_index``, ``title``.

    Parameters
    ----------
    chunks:
        List of chunk dicts produced by :func:`src.retrieval.chunking.chunk_text`.
    """
    if not chunks:
        return
    client = _get_client()
    texts = [c["text"] for c in chunks]
    vectors: np.ndarray = _embedder.embed(texts)

    points = [
        models.PointStruct(
            id=_make_point_id(c["doc_id"], c["chunk_index"]),
            vector=vectors[i].tolist(),
            payload={
                "doc_id": c["doc_id"],
                "access_level": c["access_level"],
                "chunk_index": c["chunk_index"],
                "title": c["title"],
                "text": c["text"],
            },
        )
        for i, c in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(query_text: str, role: str, top_k: int = 3) -> list[dict]:
    """Retrieve the top-k chunks visible to *role*, RBAC filter applied at query time.

    Parameters
    ----------
    query_text:
        The user's natural-language question.
    role:
        One of ``intern``, ``employee``, ``hr``, ``admin``.
    top_k:
        Number of results to retrieve.

    Returns
    -------
    list[dict]
        Each dict has keys: ``doc_id``, ``access_level``, ``chunk_index``,
        ``title``, ``text``, ``score``.
    """
    client = _get_client()
    q_vec = _embedder.embed([query_text])[0].tolist()
    rbac_filter = build_rbac_filter(role)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        query_filter=rbac_filter,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "doc_id": p.payload["doc_id"],
            "access_level": p.payload["access_level"],
            "chunk_index": p.payload["chunk_index"],
            "title": p.payload["title"],
            "text": p.payload["text"],
            "score": p.score,
        }
        for p in results.points
    ]


def search_no_filter(query_text: str, top_k: int = 3) -> list[dict]:
    """Retrieve top-k chunks with NO RBAC filter.

    For eval baseline only — never exposed through the API.
    """
    client = _get_client()
    q_vec = _embedder.embed([query_text])[0].tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "doc_id": p.payload["doc_id"],
            "access_level": p.payload["access_level"],
            "chunk_index": p.payload["chunk_index"],
            "title": p.payload["title"],
            "text": p.payload["text"],
            "score": p.score,
        }
        for p in results.points
    ]


def get_collection_info() -> dict:
    """Return basic stats about the collection for the /health endpoint."""
    client = _get_client()
    info = client.get_collection(COLLECTION_NAME)
    # points_count is reliable; vectors_count was removed in newer Qdrant versions
    points = getattr(info, "points_count", None)
    vectors = getattr(info, "vectors_count", points)
    return {
        "name": COLLECTION_NAME,
        "vectors_count": vectors,
        "points_count": points,
        "status": str(info.status),
    }


def scroll_all_payloads(batch_size: int = 256) -> list[dict]:
    """Return all payload dicts from the collection (used by unit tests)."""
    client = _get_client()
    all_payloads: list[dict] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=batch_size,
            with_payload=True,
        )
        for r in records:
            all_payloads.append(r.payload)
        if offset is None:
            break
    return all_payloads
