# BGE-M3 in React UI - Quick Reference 🚀

## What is BGE-M3?

**BGE-M3** (BAAI General Embedding - Multi-lingual, Multi-granularity, Multi-representation) is a state-of-the-art embedding model released in 2024 by BAAI (Beijing Academy of Artificial Intelligence).

- **Dimensions:** 1024d (vs. 300d for Word2Vec/FastText)
- **Languages:** Multi-lingual support
- **Strength:** Semantic understanding + cross-lingual capabilities
- **Trade-off:** Slower inference, requires more VRAM

---

## Where to Find BGE-M3 in the UI

### 1. Algorithm Selector (Left Panel)
```
┌─────────────────────────────────────────┐
│ Embedding Method                         │
│ 6 methods · from word counting (1954)    │
│ to multi-lingual embeddings (2024)       │
│                                          │
│ [BOW]  [TF-IDF]  [Word2Vec]  [GloVe]   │
│ [FastText]  [🚀 BGE-M3]                │
│           Selected: BGE-M3 (2024)        │
│           Tagline: Multi-lingual...      │
│           ✓ State-of-the-art semantic    │
│           ✗ Slower inference, more VRAM  │
└─────────────────────────────────────────┘
```

### 2. Recommendation Button
```
🎯 Recommend with BGE_M3
```
Click to get recommendations using only BGE-M3 embeddings

### 3. Comparison Tab
Click **"⚡ Compare All 6 Methods"** to see:
- **Bar Chart:** BGE-M3 in purple bars
- **Radar Chart:** BGE-M3 as 6th data point
- **Per-Method Grids:** BGE-M3 results in purple

### 4. Footer
See how BGE-M3 works:
```
BAAI multi-lingual 1024d vectors 
→ cross-lingual semantic sim
```

---

## How to Use BGE-M3

### Step 1: Select a Product
```
Search: [Enter product name...]
```

### Step 2: Choose BGE-M3
```
Click the purple [🚀 BGE-M3] button in the Algorithm Selector
```

### Step 3: Get Recommendations
```
Click: 🎯 Recommend with BGE_M3
```

### Step 4: View Results
```
Ranked by BGE-M3 (BAAI) cosine similarity
- Shows top 10 products with similarity %
- Purple color indicates BGE-M3 method
```

---

## Comparing BGE-M3 with Other Methods

### Use BGE-M3 When You Need:
✅ **Most accurate semantic matches**
✅ **Multi-lingual understanding**
✅ **State-of-the-art embeddings** (2024)
✅ **Dense representations** (1024 dimensions)

### Use Other Methods When:
- **Word Counting (BOW):** Fast, simple interpretability
- **TF-IDF:** Highlight rare keywords
- **Word2Vec:** Pre-trained on news corpus
- **GloVe:** Fast, proven semantic similarity
- **FastText:** Handle misspellings & unusual words

---

## Visual Identification

### BGE-M3 Color: Purple 🟣
```
Color Code: #7C3AED
Background: #F3E8FF (Light Purple)
Icon: 🚀
Year: 2024
```

### Where You'll See Purple:
- ✅ Algorithm selector button
- ✅ Bar chart bars in comparison
- ✅ Radar chart data point
- ✅ Result cards in grids
- ✅ Per-method section header

---

## Example Use Cases

### Case 1: Finding Exact Semantic Matches
**Task:** Find products most semantically similar to "luxury smartphone"

**Steps:**
1. Enter: `luxury smartphone`
2. Select: `BGE-M3`
3. Click: `Recommend`
4. **Result:** Gets truly semantic matches (not just keyword matches)

### Case 2: Comparing All 6 Methods
**Task:** See how different algorithms score the same products

**Steps:**
1. Enter: any product
2. Click: `Compare All 6 Methods`
3. View: Bar chart with BGE-M3 in purple
4. **Result:** Understand trade-offs between methods

### Case 3: Checking Multi-lingual Understanding
**Task:** Test if BGE-M3 understands products in different languages

**Steps:**
1. Enter: `テレビ` (TV in Japanese) or `télévision` (TV in French)
2. Select: `BGE-M3`
3. Click: `Recommend`
4. **Result:** See if semantic understanding works cross-lingually

---

## Performance Notes

### Processing Time
```
Method          | Avg Time per Product | Total for 1000
─────────────────────────────────────────────────
BOW             | ~1ms                 | ~1 second
TF-IDF          | ~1ms                 | ~1 second
Word2Vec        | ~5ms                 | ~5 seconds
GloVe           | ~8ms                 | ~8 seconds
FastText        | ~15ms                | ~15 seconds
BGE-M3          | ~50-100ms            | ~50-100 seconds ⚠️
─────────────────────────────────────────────────
```

### Memory Usage
```
Method          | Approx Model Size
─────────────────────────────────────
BOW             | ~5MB (sparse matrix)
TF-IDF          | ~10MB (sparse matrix)
Word2Vec        | ~375MB (300d × 3M vocab)
GloVe           | ~150MB (50d pre-trained)
FastText        | ~1.1GB (300d × 1M vocab)
BGE-M3          | ~1.2GB (1024d × 500k vocab) ⚠️
─────────────────────────────────────
```

---

## Batch Processing During BGE-M3 Loading

When the backend loads BGE-M3, you'll see:
```
[6/6] Loading BGE-M3 embeddings...
     Generating embeddings (this may take 1-2 minutes)...
Batches: 100%|████████████| 43/43 [01:23<00:00,  1.95s/batch]
     ✓ BGE-M3 embeddings ready!
```

**What this means:**
- **43/43:** Processing 43 batches of products (1000 products ÷ 32 per batch)
- **100%:** All batches complete
- **1:23:** Total processing time (~1 min 23 sec)
- **1.95s/batch:** Average time per batch

📖 **Read more:** See `BATCH_SIZE_43_EXPLAINED.md` for detailed explanation

---

## Troubleshooting

### Q: BGE-M3 button not showing?
**A:** Make sure:
- Frontend is running (`npm run dev`)
- Backend is running (`uvicorn main:app`)
- Page is fully loaded (refresh if needed)

### Q: BGE-M3 takes too long?
**A:** 
- BGE-M3 is naturally slower (state-of-the-art trade-off)
- First load includes batch processing
- Subsequent recommendations are cached
- Use other methods for faster testing

### Q: Compare button shows only 5 methods?
**A:** 
- Refresh the page
- Check browser console for errors
- Restart backend server

### Q: Purple color not showing?
**A:** 
- Clear browser cache
- Hard refresh: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)

---

## Technical Details

### Backend Endpoint
```
POST /recommend
{
  "product": "Galaxy S24",
  "method": "bge_m3",      ← Use this value
  "top_n": 10
}
```

### Response
```json
{
  "query": "Galaxy S24",
  "method": "bge_m3",
  "recommendations": [
    {
      "title": "iPhone 15 Pro",
      "similarity": 0.87,
      "poster": "https://..."
    },
    ...
  ]
}
```

### Comparison Endpoint
```
POST /compare
{
  "product": "Galaxy S24",
  "top_n": 8
}
```

Returns results for all 6 methods including BGE-M3.

---

## Architecture

```
React Frontend (UI)
    ↓
AlgorithmSelector (shows BGE-M3 as option)
    ↓
App.jsx (handles BGE-M3 selection)
    ↓
MovieGrid + ComparisonChart (displays results)
    ↓
FastAPI Backend
    ↓
ProductRecommender.recommend(method="bge_m3")
    ↓
BGE-M3 Model (BAAI)
    ↓
Embeddings (1024d vectors)
    ↓
Cosine Similarity Calculation
    ↓
Ranked Results (back to UI)
```

---

## Summary

| Feature | Status |
|---------|--------|
| BGE-M3 in Algorithm Selector | ✅ Implemented |
| Individual Recommendations | ✅ Implemented |
| Comparison with other methods | ✅ Implemented |
| Color Coding (Purple) | ✅ Implemented |
| Documentation | ✅ Implemented |
| Backend Support | ✅ Ready |

**Status:** 🎉 **Fully Integrated**

You can now use BGE-M3 for product recommendations in the React UI!

---

## See Also
- `BGE_M3_REACT_INTEGRATION.md` - Detailed integration changes
- `BATCH_SIZE_43_EXPLAINED.md` - Batch processing explanation
- `BGE_M3_SETUP.md` - Backend setup guide
- `QWEN_MODEL_IMPLEMENTATION.md` - Qwen model integration

---

**For questions or issues, check the documentation files above!** 📚
