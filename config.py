"""
Central configuration for the Semantic Search Engine.
All tunable parameters live here.
"""

import os
import torch

# ── TEI (Text Embeddings Inference) ────────────────────
USE_TEI = os.getenv("USE_TEI", "True").lower() in ("true", "1", "yes")
TEI_URL = os.getenv("TEI_URL", "http://8.211.22.117:9090/embed")

# ── Qdrant ──────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = 6333
COLLECTION_NAME = "products"

# ── Embedding Model ────────────────────────────────────
MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 512           # snapshot uses 512-dim dense vectors (named "text")

# ── Device & Precision ─────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16   # fp16 for 8 GB VRAM optimisation

# ── Batch / Search ─────────────────────────────────────
BATCH_SIZE = 32               # ingestion batch size
# ── Batch / Search ─────────────────────────────────────
BATCH_SIZE = 32               # ingestion batch size
TOP_K = 5                     # default number of search results
SEARCH_MODE = "sparse"        # 'dense', 'sparse', or 'hybrid'

# ── FastAPI ────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
