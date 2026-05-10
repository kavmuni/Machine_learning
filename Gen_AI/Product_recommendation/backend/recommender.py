"""
recommender.py — Text embedding methods for movie recommendation.

Every method converts text → numbers, then uses cosine similarity to find similar movies.
That's the one constant: representation changes, similarity measure stays the same.

Methods covered: BoW · TF-IDF · Word2Vec (CBOW) · GloVe · FastText
Note: Transformer-based embeddings (SBERT/BERT) are covered in the Attention session.
"""

import re
import os
import ssl
import base64
import certifi
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

# Fix macOS SSL certificate issue so gensim can download pre-trained models
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# ─────────────────────────────────────────────────────────
# Algorithm metadata shown in the frontend
# ─────────────────────────────────────────────────────────
ALGORITHM_INFO = {
    "bow": {
        "name": "Bag of Words",
        "year": "1954",
        "color": "#ef4444",
        "tagline": "Counts word occurrences — order doesn't matter",
        "pro": "Simple, fast, interpretable",
        "con": "Ignores word order and semantics; 'dog bites man' = 'man bites dog'",
        "formula": "V[word] = count of word in document",
    },
    "tfidf": {
        "name": "TF-IDF",
        "year": "1972",
        "color": "#f59e0b",
        "tagline": "Rare words matter more than common ones",
        "pro": "Weights important/unique words higher automatically",
        "con": "Still bag-of-words; no semantic understanding",
        "formula": "TF-IDF = TF(t,d) × log(N / df(t))",
    },
    "word2vec": {
        "name": "Word2Vec (Google News) - DISABLED",
        "year": "2013",
        "color": "#10b981",
        "tagline": "Pre-trained on 100B Google News words — 3M vocab, 300d (REQUIRES 1.6GB DOWNLOAD)",
        "pro": "'king - man + woman ≈ queen' — rich real-world semantics",
        "con": "No subword model — OOV words get zero vector; Currently disabled for faster startup (using TF-IDF instead)",
        "formula": "P(center | context) → trained weights become vectors",
        "status": "disabled - falling back to TF-IDF"
    },
    "glove": {
        "name": "GloVe",
        "year": "2014",
        "color": "#6366f1",
        "tagline": "Global co-occurrence statistics + local context",
        "pro": "Stable, rich pre-trained vectors from 6B words",
        "con": "One vector per word — polysemy not handled ('bank')",
        "formula": "u·v ≈ log P(word_i | word_j)",
    },
    "fasttext": {
        "name": "FastText (Wiki News) - DISABLED",
        "year": "2016",
        "color": "#ec4899",
        "tagline": "Pre-trained on Wikipedia + news — 1M vocab, 300d subwords (REQUIRES 958MB DOWNLOAD)",
        "pro": "Handles OOV words via character n-grams — typos still work!",
        "con": "Subword noise can hurt very short product descriptions; Currently disabled for faster startup (using TF-IDF instead)",
        "formula": "word_vector = avg of character n-gram vectors",
        "status": "disabled - falling back to TF-IDF"
    },
    "bge_m3": {
        "name": "BGE-M3 (BAAI)",
        "year": "2024",
        "color": "#8b5cf6",
        "tagline": "Qwen-quality embeddings from BAAI — 1024d multilingual (~2GB)",
        "pro": "State-of-the-art dense embeddings, multilingual, fast, 2GB only",
        "con": "Requires sentence-transformers package (one-time download)",
        "formula": "embedding = BGE_M3_Encoder(text)",
        "status": "Production-ready"
    }
}


# ─────────────────────────────────────────────────────────
# Core Recommender Class
# ─────────────────────────────────────────────────────────
class ProductRecommender:
    """
    Builds 5 different text representations of 1000 movies.
    Then lets you compare how each one ranks similar movies.
    """

    def __init__(self, parquet_path: str):
        print("[*] Loading product data...")
        self.df = pd.read_parquet(parquet_path)
        self.titles = self.df["Product Name"].tolist()
        self._prepare_text()
        self._build_all_models()
        print("[OK] All models ready - recommender is live!")

    # ─── Text preprocessing ───────────────────────────────

    def _clean(self, text: str) -> str:
        """Lowercase, remove punctuation/numbers, strip extra whitespace."""
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _prepare_text(self):
        """Combine relevant metadata into one text field per movie."""
        self.df["combined"] = (
            self.df["description"].fillna("") + " " +
            self.df["Category"].fillna("") + " " +
            self.df["Variants"].fillna("") + " " +
            self.df["Selling Price"].fillna("") + " " +
            self.df["About Product"].fillna("")
        )
        self.df["text_clean"] = self.df["combined"].apply(self._clean)
        self.corpus = self.df["text_clean"].tolist()
        # Tokenized version for Word2Vec / FastText
        self.tokenized = [doc.split() for doc in self.corpus]

    # ─── Build all similarity matrices ───────────────────

    def _build_all_models(self):
        self._build_bow()
        self._build_tfidf()
        self._build_word2vec()
        self._build_glove()
        self._build_fasttext()
        self._build_bge_m3()  # BGE-M3: Qwen-quality embeddings, lightweight


    def _build_bow(self):
        print("  [1/5] Building Bag-of-Words model...")
        vec = CountVectorizer(max_features=8000, stop_words="english")
        matrix = vec.fit_transform(self.corpus)
        self.sim_bow = cosine_similarity(matrix)

    def _build_tfidf(self):
        print("  [2/5] Building TF-IDF model...")
        vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english")
        matrix = vec.fit_transform(self.corpus)
        self.sim_tfidf = cosine_similarity(matrix)

    def _avg_vectors(self, model_wv, size: int) -> np.ndarray:
        """Average word vectors across each document."""
        vecs = []
        for tokens in self.tokenized:
            token_vecs = [model_wv[w] for w in tokens if w in model_wv]
            vecs.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(size))
        return np.array(vecs)

    def _build_word2vec(self):
        print("  [3/5] Skipping Word2Vec (1.6GB download) - using TF-IDF fallback for faster startup...")
        # Word2Vec requires 1.6GB download and 3-5 minutes to load
        # For faster startup, we skip it and use TF-IDF as fallback
        # Users who want Word2Vec can uncomment lines below and wait for first-time download
        self.sim_w2v = self.sim_tfidf.copy()

        # To enable Word2Vec, uncomment below:
        # try:
        #     print("  [3/5] Loading pre-trained Google News Word2Vec (300d, ~1.6 GB)...")
        #     wv = api.load("word2vec-google-news-300")
        #     matrix = self._avg_vectors(wv, 300)
        #     self.sim_w2v = cosine_similarity(matrix)
        # except Exception as e:
        #     print(f"  [WARNING] Word2Vec download failed ({e}). Falling back to TF-IDF.")
        #     self.sim_w2v = self.sim_tfidf.copy()

    def _build_glove(self):
        print("  [4/5] Loading pre-trained GloVe vectors (~66 MB, cached after first run)...")
        try:
            glove = api.load("glove-wiki-gigaword-50")
            matrix = self._avg_vectors(glove, 50)
            self.sim_glove = cosine_similarity(matrix)
        except Exception as e:
            print(f"  [WARNING] GloVe download failed ({e}). Falling back to TF-IDF for GloVe slot.")
            self.sim_glove = self.sim_tfidf.copy()

    def _build_fasttext(self):
        print("  [5/5] Skipping FastText (958MB download) - using TF-IDF fallback for faster startup...")
        # FastText requires 958MB download and 3-5 minutes to load
        # For faster startup, we skip it and use TF-IDF as fallback
        # Users who want FastText can uncomment lines below and wait for first-time download
        self.sim_fasttext = self.sim_tfidf.copy()

        # To enable FastText, uncomment below:
        # try:
        #     print("  [5/5] Loading pre-trained FastText wiki-news vectors (300d, ~958 MB)...")
        #     wv = api.load("fasttext-wiki-news-subwords-300")
        #     matrix = self._avg_vectors(wv, 300)
        #     self.sim_fasttext = cosine_similarity(matrix)
        # except Exception as e:
        #     print(f"  [WARNING] FastText download failed ({e}). Falling back to TF-IDF.")
        #     self.sim_fasttext = self.sim_tfidf.copy()

    def _build_bge_m3(self):
        print("  [6/6] Loading BGE-M3 embeddings (BAAI/bge-m3, ~2GB, cached after first run)...")
        try:
            from sentence_transformers import SentenceTransformer

            # Download and cache the model
            print("       Downloading BAAI/bge-m3 model...")
            model = SentenceTransformer('BAAI/bge-m3')

            # Generate embeddings for all products
            print("       Generating embeddings (this may take 1-2 minutes)...")
            embeddings = model.encode(
                self.corpus,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            # Calculate similarity matrix
            self.sim_bge_m3 = cosine_similarity(embeddings)
            print("       ✓ BGE-M3 embeddings ready!")

        except ImportError:
            print("  [WARNING] sentence-transformers not installed.")
            print("           Run: pip install sentence-transformers")
            print("           Falling back to TF-IDF for BGE-M3 slot.")
            self.sim_bge_m3 = self.sim_tfidf.copy()
        except Exception as e:
            print(f"  [WARNING] BGE-M3 download failed ({e}). Falling back to TF-IDF.")
            self.sim_bge_m3 = self.sim_tfidf.copy()

    # ─── Recommendation logic ─────────────────────────────

    _SIM_MAP = {
        "bow":      lambda self: self.sim_bow,
        "tfidf":    lambda self: self.sim_tfidf,
        "word2vec": lambda self: self.sim_w2v,
        "glove":    lambda self: self.sim_glove,
        "fasttext": lambda self: self.sim_fasttext,
        "bge_m3":   lambda self: self.sim_bge_m3,
    }

    def _product_to_dict(self, row, similarity: float) -> dict:
        # Convert image bytes to base64 data URL for display
        poster = "data:image/jpeg;base64,"
        image_data = row.get("image", {})

        # Handle both dict (with 'bytes' key) and direct bytes
        if isinstance(image_data, dict):
            image_bytes = image_data.get("bytes", b"")
        else:
            image_bytes = image_data if isinstance(image_data, bytes) else b""

        if image_bytes:
            poster += base64.b64encode(image_bytes).decode('utf-8')
        else:
            poster = ""

        return {
            "title":             str(row["Product Name"]),
            "name":              str(row["Product Name"]),
            "category":          str(row.get("Category", "N/A")),
            "price":             str(row.get("Selling Price", "N/A")),
            "weight":            str(row.get("Shipping Weight", "N/A")),
            "description":       str(row.get("About Product", "No overview available")),
            "technical_details": str(row.get("Technical Details", "N/A")),
            "model_number":      str(row.get("Model Number", "N/A")),
            "poster":            poster,
            "rating":            "N/A",
            "similarity":        round(float(similarity), 4),
        }

    def recommend(self, product_name: str, method: str = "tfidf", top_n: int = 10) -> list:
        """Return top_n similar movies using the chosen method."""
        if method not in self._SIM_MAP:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(self._SIM_MAP)}")
        try:
            idx = self.titles.index(product_name)
        except ValueError:
            return []

        sim_matrix = self._SIM_MAP[method](self)
        scores = sim_matrix[idx]
        ranked = np.argsort(-scores)
        top_indices = [i for i in ranked if i != idx][:top_n]

        return [self._product_to_dict(self.df.iloc[i], scores[i]) for i in top_indices]

    def compare_all(self, product_name: str, top_n: int = 8) -> dict:
        """Run all 5 methods and return their top recommendations."""
        return {
            method: self.recommend(product_name, method=method, top_n=top_n)
            for method in self._SIM_MAP
        }

    def get_titles(self) -> list:
        return self.titles

    def algorithm_info(self) -> dict:
        return ALGORITHM_INFO
