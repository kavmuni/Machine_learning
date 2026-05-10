# BGE-M3 Installation & Testing Guide

## What is BGE-M3?

**BGE-M3** is a state-of-the-art multilingual embedding model from **BAAI (Beijing Academy of Artificial Intelligence)** that provides:
- ✅ Qwen-quality semantic embeddings
- ✅ 1024-dimensional dense vectors
- ✅ Multilingual support (100+ languages)
- ✅ Only 2GB download size
- ✅ Fast CPU inference
- ✅ Production-ready

Think of it as: **"Qwen-level quality in a lightweight package"**

---

## Installation Steps

### Step 1: Install sentence-transformers package
```bash
pip install sentence-transformers torch
```

**What this installs:**
- `sentence-transformers`: Framework for using pre-trained embedding models
- `torch`: PyTorch library needed by transformers
- Total size: ~500MB (libraries only; model downloads separately on first use)

### Step 2: Verify installation
```bash
python -c "from sentence_transformers import SentenceTransformer; print('✓ Installation successful')"
```

### Step 3: First-time model download
When you run the backend for the first time:
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

**Expected output:**
```
[*] Loading product data...
[1/5] Building Bag-of-Words model...
[2/5] Building TF-IDF model...
[3/5] Skipping Word2Vec (1.6GB download)...
[4/5] Loading pre-trained GloVe vectors...
[5/5] Skipping FastText (958MB download)...
[6/6] Loading BGE-M3 embeddings (BAAI/bge-m3, ~2GB, cached after first run)...
     Downloading BAAI/bge-m3 model...
     Generating embeddings (this may take 1-2 minutes)...
     ✓ BGE-M3 embeddings ready!
[OK] All models ready - recommender is live!
```

⏱️ **First run:** ~2-3 minutes (downloading + processing)  
⚡ **Subsequent runs:** <30 seconds (cached)

---

## Testing BGE-M3 Recommendations

### Test 1: Start the Backend
```bash
cd C:\Users\muralidharan\PycharmProjects\zero-to-genai-engineer\01_text_to_numbers\Product_recommendation\backend
python -m uvicorn main:app --reload --port 8000
```

### Test 2: Start the Frontend
```bash
cd C:\Users\muralidharan\PycharmProjects\zero-to-genai-engineer\01_text_to_numbers\Product_recommendation\frontend
npm run dev
```

### Test 3: Open in Browser
Navigate to: http://localhost:5173

### Test 4: Test BGE-M3 Recommendations
1. Select a product from dropdown (e.g., "Barbie Fashionistas Doll Wear Your Heart")
2. Look for new button: **"🎯 Recommend with BGE-M3"**
3. Click it
4. Compare results with other methods (BoW, TF-IDF, GloVe)

**Expected Results:**
- BGE-M3 should show semantically similar products
- Better quality recommendations than BoW/TF-IDF
- Similar or better quality to GloVe
- Results based on semantic meaning, not just keyword matching

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'sentence_transformers'` | Run `pip install sentence-transformers torch` |
| `Connection timeout downloading model` | Check internet connection, retry (will resume from checkpoint) |
| `CUDA out of memory` | Normal - BGE-M3 will use CPU fallback automatically |
| `Slow embedding generation` | Normal first time (~2 min). Subsequent loads are cached (<30 sec) |
| Frontend shows no BGE-M3 button | Ensure backend is running and logs show "✓ BGE-M3 embeddings ready!" |

---

## Performance Comparison

| Method | Quality | Speed | Memory | Download |
|--------|---------|-------|--------|----------|
| BoW | ⭐⭐ | ⚡⚡⚡ | 1MB | None |
| TF-IDF | ⭐⭐ | ⚡⚡⚡ | 2MB | None |
| Word2Vec | ⭐⭐⭐ | ⚡⚡ | 1.6GB | 1.6GB |
| GloVe | ⭐⭐⭐ | ⚡⚡ | 66MB | 66MB |
| FastText | ⭐⭐⭐ | ⚡⚡ | 958MB | 958MB |
| **BGE-M3** | **⭐⭐⭐⭐⭐** | **⚡⚡** | **2GB** | **2GB** |

**BGE-M3 sweet spot:** Best quality with reasonable resource requirements

---

## Model Details

**Official Name:** `BAAI/bge-m3`  
**Organization:** Beijing Academy of Artificial Intelligence (BAAI)  
**Model Type:** Dense embedding model  
**Embedding Dimension:** 1024D  
**Languages:** 100+ languages  
**Training Data:** ~400M relevance pairs  
**License:** MIT (free to use)  

**Paper & Resources:**
- GitHub: https://github.com/FlagOpen/FlagEmbedding
- HuggingFace: https://huggingface.co/BAAI/bge-m3

---

## What Changed in the Code

### Before (Broken):
```python
def _qwen2vec_placeholder(self):
    wv = api.load("qwen2vec-qwen-2.5b-512d")  # ❌ Model doesn't exist!
```

### After (Working):
```python
def _build_bge_m3(self):
    model = SentenceTransformer('BAAI/bge-m3')  # ✅ Real, working model
    embeddings = model.encode(self.corpus, batch_size=32)
    self.sim_bge_m3 = cosine_similarity(embeddings)
```

---

## Next Steps (Optional)

If you want even higher quality embeddings later:
1. **Qwen2-7B** - Use full Qwen model (14GB, slower)
2. **E5-Large** - Another BAAI model optimized for similarity
3. **Jina Embeddings** - Task-specific fine-tuned models

But **BGE-M3 is the sweet spot** for this project! 🎯

---

**Status:** ✅ Ready to test  
**Installation Time:** ~5 minutes  
**First Run Time:** ~2-3 minutes  
**Subsequent Runs:** <30 seconds
