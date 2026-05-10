"""
main.py — FastAPI backend for the CineMatch movie recommender.

Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from recommender import ProductRecommender, ALGORITHM_INFO
import os

# ─────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="ProductRecommand API",
    description="Product recommendation using 6 different text embedding methods",
    version="1.0.0",
)

# Allow React frontend to call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# Load recommender on startup (builds all 6 models)
# ─────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "imdb_top_1000.csv")
parquet_path = os.path.join(os.path.dirname(__file__), "data", "train-00000-of-00001.parquet")
if os.path.exists(parquet_path):
    print("📂 Loading preprocessed data from Parquet...")
    recommender = ProductRecommender(parquet_path)
else:
    recommender = ProductRecommender(CSV_PATH)


# ─────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    product: str
    method: str = "tfidf"
    top_n: int = 10


class CompareRequest(BaseModel):
    product: str
    top_n: int = 8


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "movies_loaded": len(recommender.get_titles())}


@app.get("/movies")
def list_movies():
    """Return all movie titles (for the search dropdown)."""
    return {"movies": recommender.get_titles()}


@app.get("/products")
def list_products():
    """Return all product titles (for the search dropdown)."""
    return {"products": recommender.get_titles()}


@app.get("/algorithms")
def list_algorithms():
    """Return metadata for each algorithm (displayed in the UI)."""
    return recommender.algorithm_info()


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Get top-N product recommendations for a given product using one method.

    Methods: bow | tfidf | word2vec | glove | fasttext | bge_m3
    """
    valid_methods = ["bow", "tfidf", "word2vec", "glove", "fasttext", "bge_m3"]
    if req.method not in valid_methods:
        raise HTTPException(status_code=400, detail=f"method must be one of {valid_methods}")

    results = recommender.recommend(req.product, method=req.method, top_n=req.top_n)

    if not results:
        raise HTTPException(status_code=404, detail=f"Product '{req.product}' not found")

    return {
        "query": req.product,
        "method": req.method,
        "algorithm": ALGORITHM_INFO[req.method]["name"],
        "recommendations": results,
    }


@app.post("/compare")
def compare(req: CompareRequest):
    """
    Run ALL 5 methods on the same movie and return all results for comparison.
    """
    all_results = recommender.compare_all(req.product, top_n=req.top_n)
    if not any(all_results.values()):
        raise HTTPException(status_code=404, detail=f"Product '{req.product}' not found")

    return {
        "query": req.product,
        "results": all_results,
        "algorithms": ALGORITHM_INFO,
    }
