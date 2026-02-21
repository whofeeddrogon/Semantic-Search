"""
model_loader.py — Shared loader for Dense (BGE-M3) and Sparse (BM25) models.
"""

import numpy as np
import requests
import config

# Dense Model (SentenceTransformers)
from sentence_transformers import SentenceTransformer
_dense_model: SentenceTransformer | None = None

# Sparse Model (FastEmbed)
# We use FastEmbed because Qdrant's sparse vectors are typically generated using it (or Splade)
try:
    from fastembed import SparseTextEmbedding
    _sparse_model: SparseTextEmbedding | None = None
except ImportError:
    _sparse_model = None
    print("[model_loader] Warning: 'fastembed' not installed. Sparse mode will fail.")


def get_dense_model() -> SentenceTransformer:
    global _dense_model
    if _dense_model is None:
        use_fp16 = config.DEVICE == "cuda"
        precision = "fp16" if use_fp16 else "fp32"
        print(f"[model_loader] Loading Dense Model ({config.MODEL_NAME}) on {config.DEVICE} ({precision})…")
        _dense_model = SentenceTransformer(config.MODEL_NAME, device=config.DEVICE)
        if use_fp16:
            _dense_model.half()
    return _dense_model


def get_sparse_model():
    """Load Qdrant/bm25 (or Splade) from FastEmbed."""
    global _sparse_model
    if _sparse_model is None:
        print(f"[model_loader] Loading Sparse Model (Qdrant/bm25)…")
        # Snapshot says 'bm25', so likely Qdrant/bm25 or Qdrant/bm42
        # If it fails, we fall back to Splade
        try:
            from fastembed import SparseTextEmbedding
            # Using standard BM25 model supported by Qdrant
            _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25") 
        except Exception as e:
            print(f"[model_loader] Failed to load sparse model: {e}")
            raise e
    return _sparse_model


def encode_dense(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    """Encode text into dense vectors (truncated if needed)."""
    if config.USE_TEI:
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            response = requests.post(
                config.TEI_URL,
                json={"inputs": texts},
                timeout=30
            )
            response.raise_for_status()
            embeddings = np.array(response.json(), dtype=np.float32)
        except Exception as e:
            print(f"[model_loader] TEI Server failed at {config.TEI_URL}. Error: {e}")
            raise e
    else:
        model = get_dense_model()
        bs = batch_size or config.BATCH_SIZE
        embeddings = model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=len(texts) > bs,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)

    # Truncate if larger (e.g. 1024 -> 512)
    if embeddings.shape[1] > config.EMBEDDING_DIM:
        embeddings = embeddings[:, :config.EMBEDDING_DIM]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
    return embeddings


def encode_sparse(text: str):
    """Encode a single text into a SparseVector (indices, values)."""
    model = get_sparse_model()
    # FastEmbed returns a generator of sparse vectors
    # We take the first one since we encode one by one here for search
    embedding_gen = model.embed([text])
    sparse_vec = next(embedding_gen)
    
    # Qdrant expects SparseVector(indices=..., values=...)
    from qdrant_client.models import SparseVector
    return SparseVector(
        indices=sparse_vec.indices.tolist(),
        values=sparse_vec.values.tolist()
    )
