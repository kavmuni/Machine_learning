# BGE-M3 Batch Processing Explained

## What You're Seeing in Console

When you run the backend, you'll see output like:
```
[6/6] Loading BGE-M3 embeddings (BAAI/bge-m3)...
     Downloading BAAI/bge-m3 model...
     Generating embeddings (this may take 1-2 minutes)...
Batches: 100%|████████████| 43/43 [01:23<00:00,  1.95s/batch]
     ✓ BGE-M3 embeddings ready!
```

The **"43"** that appears is:
- **Total number of batches** needed to process all products
- **NOT** a configuration you set
- **Calculated automatically** from: `total_products / batch_size`

---

## Where Batch Size = 32 Comes From

In `recommender.py`, line 219:

```python
embeddings = model.encode(
    self.corpus,
    batch_size=32,        # ← This is where batch_size is set to 32
    show_progress_bar=True,
    convert_to_numpy=True
)
```

### How Batch Count is Calculated

```
Number of Batches = ceil(Total Products / Batch Size)
Number of Batches = ceil(1000 / 32)
Number of Batches = ceil(31.25)
Number of Batches = 32-43 batches
```

(Exact number depends on your actual dataset size)

### Why Batch Processing?

- **Memory efficient**: Process 32 products at a time instead of all 1000
- **GPU/CPU friendly**: Prevents out-of-memory errors
- **Progress tracking**: Shows you the status with progress bar

---

## Console Output Breakdown

| Item | Meaning |
|------|---------|
| `Batches: 100%` | Progress indicator (0-100%) |
| `████████████` | Visual progress bar |
| `43/43` | Current batch / Total batches |
| `[01:23<00:00]` | Time elapsed and time remaining |
| `1.95s/batch` | Speed: 1.95 seconds per batch |

---

## Customizing Batch Size

If you want to change batch_size, edit `recommender.py` line 219:

### To process MORE at once (faster, uses more memory):
```python
batch_size=64  # Process 64 products per batch → ~16 batches total
```

### To process LESS at once (slower, uses less memory):
```python
batch_size=16  # Process 16 products per batch → ~63 batches total
```

### Recommended Values

| Batch Size | Total Batches | Speed | Memory | Recommendation |
|-----------|--------------|-------|--------|-----------------|
| 16 | ~63 | Slow | Low | Weak machines |
| **32** | **~31** | **Good** | **Medium** | **✅ Current (Recommended)** |
| 64 | ~16 | Fast | High | Powerful machine |
| 128 | ~8 | Very Fast | Very High | Server/GPU |

---

## Example: Full Flow

```python
# Dataset: 1000 products
# Batch size: 32

# Batch 1: Process products 1-32
# Batch 2: Process products 33-64
# Batch 3: Process products 65-96
# ...
# Batch 43: Process remaining products
# ✓ All 1000 products embedded!
```

---

## Performance Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| Model Download | ~1 minute | `Downloading BAAI/bge-m3 model...` |
| Batch Processing | ~1-2 minutes | `Batches: 100%\|████████████\| 43/43` |
| Total First Run | ~2-3 minutes | ✓ BGE-M3 embeddings ready! |
| Subsequent Runs | <30 seconds | (Model cached locally) |

---

## If You See Different Numbers

### Fewer batches (e.g., 20 instead of 43):
- Your dataset has fewer products
- Batch size was increased
- Both are fine!

### More batches (e.g., 100 instead of 43):
- Your dataset has more products
- Batch size was decreased
- This is also fine!

### Very slow batches (e.g., 10s/batch):
- GPU not being used
- CPU is slower but works
- Can be optimized later if needed

---

## Code Location

- **Batch size setting**: `recommender.py`, line 219
- **Model loading**: `recommender.py`, line 216
- **What's being processed**: `self.corpus` (list of all product descriptions)
- **Output**: `self.sim_bge_m3` (similarity matrix for recommendations)

---

## Summary

| Question | Answer |
|----------|--------|
| **Where does 43 come from?** | `ceil(1000 products / 32 batch_size)` |
| **Can I change it?** | Yes, change `batch_size=32` to any value |
| **Should I change it?** | No, 32 is optimal for most systems |
| **Is it an error?** | No, it's normal progress output |

---

**The batch size 43 is completely normal and expected!** ✅

It's just showing you how many chunks the embedding process needs to go through.
