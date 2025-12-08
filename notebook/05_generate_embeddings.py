"""
Module 1 - Task 2: Generate Product Embeddings
This script generates vector embeddings for all unique products.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from datetime import datetime

print("="*80)
print("PRODUCT EMBEDDINGS GENERATION")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Step 1: Load cleaned dataset
print("\n" + "="*80)
print("STEP 1: LOADING CLEANED DATASET")
print("="*80)

df = pd.read_csv('data/dataset_cleaned.csv', low_memory=False)
print(f"✓ Dataset loaded: {len(df):,} rows")

# Step 2: Get unique products
print("\n" + "="*80)
print("STEP 2: EXTRACTING UNIQUE PRODUCTS")
print("="*80)

# Group by StockCode to get unique products with their info
products = df.groupby('StockCode').agg({
    'Description': 'first',
    'UnitPrice': 'mean',  # Average price
    'Country': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # Most common country
}).reset_index()

print(f"✓ Unique products: {len(products):,}")
print(f"\nSample products:")
print(products.head(10)[['StockCode', 'Description', 'UnitPrice']].to_string(index=False))

# Step 3: Load embedding model
print("\n" + "="*80)
print("STEP 3: LOADING EMBEDDING MODEL")
print("="*80)

model_name = 'all-MiniLM-L6-v2'
print(f"Model: {model_name}")
print(f"Dimension: 384")
print(f"Loading model...")

model = SentenceTransformer(model_name)
print(f"✓ Model loaded successfully!")

# Step 4: Generate embeddings
print("\n" + "="*80)
print("STEP 4: GENERATING EMBEDDINGS")
print("="*80)

# Fill any None or NaN descriptions with "UNKNOWN"
products['Description'] = products['Description'].fillna('UNKNOWN')
print(f"✓ Filled {products['Description'].isnull().sum()} null descriptions")

descriptions = products['Description'].tolist()
print(f"Generating embeddings for {len(descriptions):,} products...")
print(f"This may take a few minutes...")

embeddings = model.encode(
    descriptions,
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True  # Normalize for cosine similarity
)

print(f"\n✓ Generated {len(embeddings):,} embeddings")
print(f"✓ Shape: {embeddings.shape}")
print(f"✓ Embedding dimension: {embeddings.shape[1]}")

# Step 5: Save embeddings and product data
print("\n" + "="*80)
print("STEP 5: SAVING DATA")
print("="*80)

# Save embeddings as numpy array
embeddings_file = 'data/product_embeddings.npy'
np.save(embeddings_file, embeddings)
print(f"✓ Embeddings saved to: {embeddings_file}")

# Save product metadata
products_file = 'data/product_metadata.csv'
products.to_csv(products_file, index=False)
print(f"✓ Product metadata saved to: {products_file}")

# Save as pickle for quick loading
pickle_file = 'data/product_embeddings.pkl'
with open(pickle_file, 'wb') as f:
    pickle.dump({
        'embeddings': embeddings,
        'products': products,
        'model_name': model_name
    }, f)
print(f"✓ Complete data saved to: {pickle_file}")

# Step 6: Statistics
print("\n" + "="*80)
print("STEP 6: EMBEDDING STATISTICS")
print("="*80)

print(f"\nEmbedding statistics:")
print(f"  - Min value: {embeddings.min():.4f}")
print(f"  - Max value: {embeddings.max():.4f}")
print(f"  - Mean value: {embeddings.mean():.4f}")
print(f"  - Std deviation: {embeddings.std():.4f}")

print(f"\nProduct statistics:")
print(f"  - Total unique products: {len(products):,}")
print(f"  - Avg price: ${products['UnitPrice'].mean():.2f}")
print(f"  - Price range: ${products['UnitPrice'].min():.2f} - ${products['UnitPrice'].max():.2f}")
print(f"  - Unique countries: {products['Country'].nunique()}")

# Sample embeddings
print(f"\nSample embedding (first product):")
print(f"  Product: {products.iloc[0]['Description']}")
print(f"  Vector (first 10 values): {embeddings[0][:10]}")

print("\n" + "="*80)
print("EMBEDDING GENERATION COMPLETE!")
print("="*80)
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nReady to upload to Pinecone!")
print("="*80)
