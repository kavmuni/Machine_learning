#!/usr/bin/env python
"""Test script to verify the ProductRecommender loads correctly."""

import sys
import os

sys.path.insert(0, r"C:\Users\muralidharan\PycharmProjects\zero-to-genai-engineer\01_text_to_numbers\Product_recommendation\backend")

try:
    from recommender import ProductRecommender

    parquet_path = r"C:\Users\muralidharan\PycharmProjects\zero-to-genai-engineer\01_text_to_numbers\Product_recommendation\backend\data\train-00000-of-00001.parquet"

    print("=" * 60)
    print("Testing ProductRecommender initialization...")
    print("=" * 60)

    recommender = ProductRecommender(parquet_path)

    print("\n[OK] Recommender loaded successfully!")
    print(f"   Total products: {len(recommender.get_titles())}")
    print(f"   First 5 products:")
    for i, title in enumerate(recommender.get_titles()[:5], 1):
        print(f"     {i}. {title}")

    print("\n" + "=" * 60)
    print("Testing recommendation for first product...")
    print("=" * 60)

    first_product = recommender.get_titles()[0]
    results = recommender.recommend(first_product, method="tfidf", top_n=5)

    print(f"\nRecommendations for '{first_product}' (TF-IDF):")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['title']} ({result['similarity']*100:.1f}%)")

    print("\n[OK] All tests passed!")

except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
