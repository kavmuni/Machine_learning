import pandas as pd

df = pd.read_parquet('data/train-00000-of-00001.parquet')
print('Columns:', df.columns.tolist())
print('\n--- Sample 1 ---')
print('Product Name:', df.iloc[0]['Product Name'])
print('Image:', df.iloc[0].get('image', 'N/A'))
print('\n--- Sample 2 ---')
print('Product Name:', df.iloc[1]['Product Name'])
print('Image:', df.iloc[1].get('image', 'N/A'))
print('\n--- Sample 3 ---')
print('Product Name:', df.iloc[2]['Product Name'])
print('Image:', df.iloc[2].get('image', 'N/A'))
