# Quick Start: BGE-M3 Testing

## 1️⃣ Install Package (One-time)
```bash
pip install sentence-transformers torch
```

## 2️⃣ Run Backend
```bash
cd C:\Users\muralidharan\PycharmProjects\zero-to-genai-engineer\01_text_to_numbers\Product_recommendation\backend
python -m uvicorn main:app --reload --port 8000
```

**Wait for output:**
```
[6/6] Loading BGE-M3 embeddings...
     ✓ BGE-M3 embeddings ready!
[OK] All models ready - recommender is live!
```

⏱️ First time: ~2-3 minutes (downloads model)  
⚡ Next times: <30 seconds (cached)

## 3️⃣ Run Frontend (New Terminal)
```bash
cd C:\Users\muralidharan\PycharmProjects\zero-to-genai-engineer\01_text_to_numbers\Product_recommendation\frontend
npm run dev
```

## 4️⃣ Test in Browser
Open: http://localhost:5173

1. Select a product from dropdown
2. Click **"🎯 Recommend with BGE-M3"** button
3. See Qwen-quality embeddings in action! 🚀

---

## 📊 Model Comparison

| Method | Quality | Size | Speed |
|--------|---------|------|-------|
| TF-IDF | ⭐⭐ | Instant | Instant |
| GloVe | ⭐⭐⭐ | 66MB | ~30s |
| **BGE-M3** | **⭐⭐⭐⭐⭐** | **2GB** | **~2min first, <30s after** |

---

## ✅ What You Get

✅ State-of-the-art embeddings (Qwen-quality)  
✅ Multilingual support  
✅ 1024-dimensional vectors  
✅ Production-ready  
✅ Easy to use  

---

## 📚 More Info

See `BGE_M3_SETUP.md` for detailed documentation
