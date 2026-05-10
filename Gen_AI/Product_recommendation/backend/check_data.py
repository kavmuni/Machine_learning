import pandas as pd
import os

backend_path = r"C:\Users\muralidharan\PycharmProjects\zero-to-genai-engineer\01_text_to_numbers\Product_recommendation\backend"
parquet_path = os.path.join(backend_path, "data", "train-00000-of-00001.parquet")

print(f"Loading parquet file from: {parquet_path}")
df = pd.read_parquet(parquet_path)

print(f"\nDataFrame shape: {df.shape}")
print(f"\nColumn names:")
print(df.columns.tolist())

print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)
