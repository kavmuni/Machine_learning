# Why Model Loading is Slow - Explanation & Solutions

## 🔍 Root Cause Analysis

### What's Happening During Startup

When you start the backend, the system loads **5 different embedding models**:

| Model | Size | Time | Purpose |
|-------|------|------|---------|
| **BOW** (Bag of Words) | < 1 MB | ~5 sec | Word counting vectorizer |
| **TF-IDF** | < 1 MB | ~5 sec | Weighted word vectors |
| **Word2Vec** (Google News) | **1.6 GB** | 2-5 min | Pre-trained word embeddings |
| **GloVe** (Wiki Gigaword) | **66 MB** | 30-60 sec | Global co-occurrence vectors |
| **FastText** (Wiki News) | **958 MB** | 3-5 min | Subword embeddings |

### Current Behavior

```
[*] Loading product data...                           (~1 sec)
  [1/5] Building Bag-of-Words model...                (~3 sec)
  [2/5] Building TF-IDF model...                      (~3 sec)
  [3/5] Loading Google News Word2Vec...              (~3-5 min) ⚠️ SLOW
  [4/5] Loading GloVe vectors...                      (~1 min)
  [5/5] Loading FastText wiki-news vectors...        (~3-5 min) ⚠️ VERY SLOW
[OK] All models ready - recommender is live!
```

**Total Time: 5-15 minutes on first run** 😱

---

## Why Is It So Slow?

### 1. **Large Model Downloads**
- Word2Vec: 1.6 GB - requires downloading 3M word vectors
- FastText: 958 MB - requires downloading 1M subword vectors
- These are downloaded via `gensim.downloader` API

### 2. **Memory Operations**
- Converting downloaded models to numpy arrays
- Computing cosine similarity matrices (1,345 × 1,345)
- This requires heavy RAM usage and CPU computation

### 3. **No Lazy Loading**
- Current code loads ALL 5 models upfront
- Even if user only needs TF-IDF, system loads all models
- No caching mechanism for pre-computed similarities

---

## ✅ Solutions - Choose One

### **Solution 1: Accept First Run (Recommended for Learning)**

**Explanation**: First run is slow because models are downloaded, but subsequent runs use cached models.

**What to Do**:
1. Start backend: `python -m uvicorn main:app --reload --port 8000`
2. Wait 5-15 minutes (only first time!)
3. Subsequent startups will be MUCH faster (30 seconds)

**Why**: gensim caches downloaded models in `~/.cache/gensim/` directory

---

### **Solution 2: Skip Downloading Models (Fast Load)**

**What**: Disable Word2Vec and FastText downloads on startup

**Implementation**:

Modify `recommender.py` - Comment out slow model downloads:

```python
def _build_word2vec(self):
    print("  [3/5] Skipping Word2Vec (falling back to TF-IDF for faster startup)...")
    # Try to load but don't block if download takes too long
    self.sim_w2v = self.sim_tfidf.copy()
    
def _build_glove(self):
    print("  [4/5] Loading pre-trained GloVe vectors...")
    try:
        glove = api.load("glove-wiki-gigaword-50")
        matrix = self._avg_vectors(glove, 50)
        self.sim_glove = cosine_similarity(matrix)
    except:
        self.sim_glove = self.sim_tfidf.copy()

def _build_fasttext(self):
    print("  [5/5] Skipping FastText (falling back to TF-IDF for faster startup)...")
    # Skipping 958MB download
    self.sim_fasttext = self.sim_tfidf.copy()
```

**Result**: ✅ Fast startup (30 seconds), but only 3 algorithms available
**Best for**: Development and testing

---

### **Solution 3: Lazy Loading (Best Practice)**

**What**: Load models only when user requests them

**How It Works**:
1. Backend starts with only BOW and TF-IDF (30 seconds)
2. When user requests Word2Vec/GloVe/FastText, system downloads in background
3. User sees "Loading..." message while waiting

**Implementation Steps**:

**Step 1**: Update `recommender.py` to lazily load models:

```python
def __init__(self, parquet_path: str):
    print("[*] Loading product data...")
    self.df = pd.read_parquet(parquet_path)
    self.titles = self.df["Product Name"].tolist()
    self._prepare_text()
    self._build_fast_models()  # Only BOW and TF-IDF
    self.models_loaded = {"bow", "tfidf"}
    print("[OK] Fast models ready - recommender started!")
    print("[*] Other models (Word2Vec, GloVe, FastText) will load on first use...")

def _build_fast_models(self):
    self._build_bow()
    self._build_tfidf()

def _ensure_model_loaded(self, method: str):
    """Load model on demand if not already loaded"""
    if method in self.models_loaded:
        return
    
    print(f"[*] Loading {method} model (first use)...")
    if method == "word2vec":
        self._build_word2vec()
    elif method == "glove":
        self._build_glove()
    elif method == "fasttext":
        self._build_fasttext()
    self.models_loaded.add(method)
```

**Step 2**: Update `recommend()` method:

```python
def recommend(self, product_name: str, method: str = "tfidf", top_n: int = 10) -> list:
    self._ensure_model_loaded(method)  # Load if needed
    # ... rest of code
```

**Result**: ✅ Fast startup (30 sec) + All 5 algorithms on demand
**Best for**: Production use

---

### **Solution 4: Pre-Cache Models (Enterprise)**

**What**: Pre-download and serialize models to disk

**How**:
```python
import pickle

def cache_models(self):
    """Pre-compute and save similarity matrices"""
    cache_dir = "./model_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    for method in ["bow", "tfidf", "word2vec", "glove", "fasttext"]:
        cache_file = f"{cache_dir}/{method}_matrix.pkl"
        if not os.path.exists(cache_file):
            sim_matrix = getattr(self, f"sim_{method.replace('-', '_')}")
            with open(cache_file, 'wb') as f:
                pickle.dump(sim_matrix, f)

def load_from_cache(self, method: str):
    """Load pre-cached similarity matrix"""
    cache_file = f"./model_cache/{method}_matrix.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None
```

**Result**: ✅ Ultra-fast startup (2-3 seconds) after first setup
**Best for**: Production servers

---

## 🚀 Recommended Approach for You

### **Immediate (No Code Changes)**
Just wait 5-15 minutes on first startup. Subsequent startups will be 30 seconds because:
- gensim caches models in `~/.cache/gensim-data/`
- Windows: `C:\Users\muralidharan\AppData\Local\gensim-data\`

**How to verify cache exists**:
```bash
dir %APPDATA%\Local\gensim-data\
```

### **For Development (Solution 2)**
Add this to `main.py` or create environment variable:
```python
# Skip heavy models for development
SKIP_HEAVY_MODELS = os.getenv("SKIP_HEAVY_MODELS", "false").lower() == "true"
```

Then only download heavy models if explicitly needed.

### **For Production (Solution 3 - Lazy Loading)**
Implement lazy loading so:
- Backend starts instantly
- Models download only when requested
- Better user experience (no 10-minute wait)

---

## 📊 Comparison

| Solution | Startup Time | All Features | Code Changes |
|----------|-------------|--------------|--------------|
| **Current** | 5-15 min* | ✅ Yes | None |
| **Solution 2** | 30 sec | ⚠️ 3 algos | Low |
| **Solution 3** | 30 sec | ✅ Yes | Medium |
| **Solution 4** | 2-3 sec | ✅ Yes | High |

*\* First run. Subsequent runs: 30 seconds (models cached)*

---

## 💡 Key Takeaway

**The slow loading is NORMAL and EXPECTED** because:

1. **Word2Vec** (1.6GB) takes time to download and parse
2. **FastText** (958MB) contains 1M+ word vectors  
3. **Cosine similarity matrices** must be pre-computed (1,345² = ~1.8M operations)

**Good news**: 
- ✅ Only happens on FIRST startup
- ✅ Subsequent startups use cached models (30 sec)
- ✅ Not a bug - it's how pre-trained models work

---

## Next Steps

Choose based on your needs:

1. **Learning/Testing**: Wait for models to cache (happens automatically)
2. **Development**: Use Solution 2 (skip heavy models)
3. **Production**: Use Solution 3 (lazy loading)

Would you like me to implement Solution 2 or 3 for you?
