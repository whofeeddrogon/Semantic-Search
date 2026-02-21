"""
app.py — FastAPI backend for the Semantic Search Engine.

Endpoints
---------
POST /search   — query text → Top-5 results
POST /add      — new text → embed & upsert
GET  /points   — all points (for visualization)
GET  /health   — health check
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import model_loader
import database_manager
import search_engine


# ── Lifespan: warm up model & DB on startup ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not config.USE_TEI:
        model_loader.get_dense_model()      # pre-load dense model (BGE-M3) locally
    else:
        print(f"[app] Using remote TEI server: {config.TEI_URL}")
        
    model_loader.get_sparse_model()     # pre-load sparse model (BM25)
    database_manager.ensure_collection()
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(title="Semantic Search Engine", lifespan=lifespan)

# Allow CORS for Streamlit / any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    mode: str | None = None  # 'dense', 'sparse'


class AddRequest(BaseModel):
    text: str


# ── Endpoints ────────────────────────────────────────────
@app.get("/health")
def health():
    count = database_manager.collection_count()
    return {
        "status": "ok", 
        "collection": config.COLLECTION_NAME, 
        "points": count,
        "mode": config.SEARCH_MODE
    }


@app.post("/search")
def do_search(req: SearchRequest):
    # Pass mode if provided, else use config default inside search_engine
    results, query_vector = search_engine.search(req.query, req.top_k, req.mode)
    
    # Handle serialization (SparseVector is not JSON serializable directly)
    vec_out = query_vector.tolist() if hasattr(query_vector, "tolist") else "sparse_vector"
    
    return {
        "query": req.query,
        "results": results,
        "query_vector": vec_out,
    }


@app.post("/add")
def add_document(req: AddRequest):
    uid = search_engine.add_document(req.text)
    return {"id": uid, "text": req.text}


from fastapi import UploadFile, File
import shutil
import os

@app.post("/upload_snapshot")
async def upload_snapshot(file: UploadFile = File(...)):
    """Temporarily save snapshot and ask database_manager to restore it."""
    try:
        # Save temp file
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"Snapshot received: {temp_path}")
        
        # We will use requests to forward it locally to Qdrant inside the docker network
        import requests
        qdrant_url = f"http://{config.QDRANT_HOST}:6333"
        restore_endpoint = f"{qdrant_url}/collections/{config.COLLECTION_NAME}/snapshots/upload"
        
        with open(temp_path, "rb") as f:
            res = requests.post(restore_endpoint, files={"snapshot": f})
            
        # Clean up temp file
        os.remove(temp_path)
        
        if res.status_code == 200:
            return {"status": "success", "message": "Snapshot restored successfully!"}
        else:
            return {"status": "error", "message": res.text}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/points")
def get_points():
    points = database_manager.get_all_points()
    return {"count": len(points), "points": points}


# ── Run directly ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
