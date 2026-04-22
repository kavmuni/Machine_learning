# Product Recommendation Engine - Setup & Troubleshooting Guide

## ✅ Issue Fixed
The "Request Failed" error when clicking recommend was caused by:
1. **Field name mismatch** - Backend was returning `"name"` but frontend expected `"title"`
2. **Encoding issues** - Unicode emoji characters caused Python to crash on Windows
3. **Missing backend server** - Backend wasn't running when trying to call the API

## 🚀 Quick Start

### Step 1: Start the FastAPI Backend
Open Command Prompt or PowerShell and run:

```bash
cd "C:\Users\muralidharan\PycharmProjects\zero-to-genai-engineer\01_text_to_numbers\Product_recommendation\backend"
python -m uvicorn main:app --reload --port 8000
```

**Expected Output:**
```
[*] Loading product data...
[*] Building Bag-of-Words model...
[*] Building TF-IDF model...
...
[OK] All models ready - recommender is live!
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Test the Backend:**
- Browser: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

### Step 2: Start the React Frontend
**In a NEW Command Prompt/PowerShell**, run:

```bash
cd "C:\Users\muralidharan\PycharmProjects\zero-to-genai-engineer\01_text_to_numbers\Product_recommendation\frontend"
npm run dev
```

**Expected Output:**
```
  VITE v5.3.4  ready in XXX ms

  ➜  Local:   http://localhost:3000/
  ➜  press h to show help
```

**Access the App:**
- Browser: http://localhost:3000

---

## 🛠️ Using Batch Files (Easier Way)

Instead of manually running commands, use the batch files:

**Option A: Start Backend**
Double-click: `C:\...\Product_recommendation\start_backend.bat`

**Option B: Start Frontend**  
Double-click: `C:\...\Product_recommendation\start_product_app.bat`

---

## ✅ What Works Now

✓ **Frontend loads correctly** at http://localhost:3000  
✓ **Product list loads** from backend API  
✓ **Recommend button works** - returns similar products  
✓ **All 5 algorithms available** - BOW, TF-IDF, Word2Vec, GloVe, FastText  
✓ **Compare button works** - runs all methods simultaneously  

---

## 🔧 Changes Made to Fix the Issue

### Backend (recommender.py)
- Fixed `_product_to_dict()` method to return `"title"` instead of `"name"`
- Removed Unicode emoji characters to fix Windows encoding issues
- Updated field mapping to match frontend expectations

### Frontend (App.jsx, Header.jsx)
- Changed API endpoint from `/movies` to `/products`
- Updated all variable names: `movies` → `products`, `selectedMovie` → `selectedProduct`
- Updated UI text to reflect "Product Recommendation" instead of "Movie Recommendation"
- Updated emoji from 🎬 to 🛍️

---

## 🐛 Troubleshooting

### **"Request Failed" when clicking Recommend**
- ✓ Ensure **Backend is running** on port 8000
- ✓ Check browser console (F12) for error details
- ✓ Verify API is accessible: http://localhost:8000/health

### **Frontend won't load**
- Kill old Node processes: `taskkill /F /IM node.exe`
- Clear browser cache (Ctrl+Shift+Delete)
- Hard refresh: Ctrl+F5
- Restart npm dev server

### **"Cannot find module" errors**
- Reinstall dependencies: `npm install --legacy-peer-deps`
- Delete node_modules and package-lock.json, then reinstall

### **Port already in use**
```bash
# Kill processes on port 8000 (backend)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Kill processes on port 3000 (frontend)
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

---

## 📊 API Endpoints Available

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check API status |
| `/products` | GET | List all products |
| `/algorithms` | GET | Get algorithm information |
| `/recommend` | POST | Get recommendations for one product |
| `/compare` | POST | Compare all 5 algorithms |

---

## 🎯 Next Steps

1. **Test Recommendations**
   - Select a product from dropdown
   - Click "Recommend with TFIDF"
   - View similar products

2. **Compare Algorithms**
   - Select a product
   - Click "Compare All 5 Methods"
   - See how different embeddings rank similarity

3. **Check Documentation**
   - API Docs: http://localhost:8000/docs
   - Interactive Swagger UI available

---

## 📝 Notes

- First run may be slow (downloading pre-trained word vectors)
- GloVe, Word2Vec, and FastText models are ~2GB total (cached after first download)
- If downloads fail, system automatically falls back to TF-IDF
- Product data: 1,345 products from Amazon catalog

---

**Setup Complete!** Your Product Recommendation Engine is ready to use. 🚀
