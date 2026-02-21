"""
search_engine.py — High-level search and document-add logic.

Combines model_loader and database_manager into simple public functions
that the API and UI layers can call.
"""

import numpy as np
import config
import model_loader
import database_manager
from qdrant_client.models import SparseVector


def search(query: str, top_k: int | None = None, mode: str | None = None) -> tuple[list[dict], np.ndarray | SparseVector]:
    """
    Encode the query and run Top-K search using 'sparse' (keyword) or 'dense' (semantic) vectors.
    """
    mode = mode or config.SEARCH_MODE
    k = top_k or config.TOP_K
    
    # Keyword search (BM25)
    if mode == "sparse":
        print(f"[search] Using SPARSE (keyword) search for: '{query}'")
        sparse_vec = model_loader.encode_sparse(query)
        results = database_manager.search_sparse(sparse_vec, k)
        return results, sparse_vec

    # Default dense search (Semantic)
    else:
        print(f"[search] Using DENSE (semantic) search for: '{query}'")
        dense_vec = model_loader.encode_dense([query])[0]
        results = database_manager.search(dense_vec, k)
        return results, dense_vec


def add_document(text: str) -> str:
    """
    Encode a single text and upsert it into Qdrant.
    Note: Now requires generating both Dense and Sparse vectors.
    """
    # 1. Dense vector
    dense = model_loader.encode_dense([text])[0]
    
    # 2. Sparse vector
    sparse = model_loader.encode_sparse(text)
    
    uid = database_manager.add_single(text, dense, sparse)
    return uid
