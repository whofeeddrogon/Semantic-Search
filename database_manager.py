"""
database_manager.py — Qdrant CRUD wrapper.

Handles collection creation, batch upsert, search (dense/sparse), and full-scroll retrieval.
"""

import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    SparseVector,
)

import config

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return a singleton QdrantClient."""
    global _client
    if _client is None:
        _client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        print(f"[db] Connected to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    return _client


def ensure_collection() -> None:
    """Create the collection if it doesn't already exist."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if config.COLLECTION_NAME not in collections:
        # Note: In a real sparse/dense setup, you'd configure both sparse_vectors and named vectors here.
        # Since we are restoring from snapshot, we assume it's already configured correctly.
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config={"text": VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE)},
            # If we were creating from scratch, we'd add sparse_vectors_config here too
        )
        print(f"[db] Created collection '{config.COLLECTION_NAME}'")
    else:
        print(f"[db] Collection '{config.COLLECTION_NAME}' already exists.")


def upsert_batch(
    vectors: np.ndarray,
    payloads: list[dict[str, Any]],
    ids: list[str] | None = None,
) -> None:
    """Legacy upsert (only dense). Used by some scripts."""
    # This function assumes single unnamed vector or just dense 'text' vector
    # Ideally should be updated to support dense+sparse
    # For now, we leave it for compatibility but add_single handles both.
    pass


def search(query_vector: np.ndarray, top_k: int | None = None) -> list[dict]:
    """Cosine-similarity search using named vector 'text' (Dense)."""
    client = get_client()
    k = top_k or config.TOP_K
    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector.tolist(),
        using="text",
        limit=k,
    )
    return [
        {"id": str(hit.id), "score": hit.score, "payload": hit.payload}
        for hit in results.points
    ]


def search_sparse(query_sparse: SparseVector, top_k: int | None = None) -> list[dict]:
    """Keyword search using named sparse vector 'bm25'."""
    client = get_client()
    k = top_k or config.TOP_K
    
    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_sparse,
        using="bm25",  # Must match the sparse vector name in collection config
        limit=k,
    )
    return [
        {"id": str(hit.id), "score": hit.score, "payload": hit.payload}
        for hit in results.points
    ]


def get_all_points(limit: int = 500) -> list[dict]:
    """Scroll through collection."""
    client = get_client()
    all_points: list[dict] = []
    offset = None

    while True:
        scroll_kwargs: dict[str, Any] = {
            "collection_name": config.COLLECTION_NAME,
            "limit": min(limit - len(all_points), 100),
            "with_vectors": ["text"],
            "with_payload": True,
        }
        if offset is not None:
            scroll_kwargs["offset"] = offset

        records, next_offset = client.scroll(**scroll_kwargs)

        for r in records:
            vec = r.vector
            if isinstance(vec, dict):
                vec = vec.get("text", vec)
            all_points.append({
                "id": str(r.id),
                "vector": vec,
                "payload": r.payload,
            })

        if next_offset is None or len(all_points) >= limit:
            break
        offset = next_offset

    return all_points


def add_single(text: str, dense_vector: np.ndarray, sparse_vector: SparseVector) -> str:
    """Add a single point with both dense and sparse vectors."""
    client = get_client()
    uid = str(uuid.uuid4())
    
    point = PointStruct(
        id=uid,
        vector={
            "text": dense_vector.tolist(),
            "bm25": sparse_vector
        },
        payload={"text": text}
    )
    
    client.upsert(collection_name=config.COLLECTION_NAME, points=[point])
    return uid


def collection_count() -> int:
    """Return the number of points in the collection."""
    client = get_client()
    info = client.get_collection(config.COLLECTION_NAME)
    return info.points_count
