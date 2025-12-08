"""
Module 1 - Task 2: Upload Embeddings to Pinecone
This script uploads product vectors to Pinecone vector database.
"""

import os
import numpy as np
import pandas as pd
from pinecone import Pinecone
from dotenv import load_dotenv
from datetime import datetime
import time

# Load environment variables
load_dotenv()

print("="*80)
print("UPLOAD VECTORS TO PINECONE")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Step 1: Load embeddings and metadata
print("\n" + "="*80)
print("STEP 1: LOADING EMBEDDINGS AND METADATA")
print("="*80)

embeddings = np.load('data/product_embeddings.npy')
products = pd.read_csv('data/product_metadata.csv')

print(f"✓ Loaded {len(embeddings):,} embeddings")
print(f"✓ Loaded {len(products):,} product records")
print(f"✓ Embedding dimension: {embeddings.shape[1]}")

# Step 2: Initialize Pinecone
print("\n" + "="*80)
print("STEP 2: CONNECTING TO PINECONE")
print("="*80)

api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)

INDEX_NAME = "ecommerce-products"
index = pc.Index(INDEX_NAME)

print(f"✓ Connected to Pinecone")
print(f"✓ Index: {INDEX_NAME}")

# Check initial stats
initial_stats = index.describe_index_stats()
print(f"✓ Current vectors in index: {initial_stats['total_vector_count']:,}")

# Step 3: Prepare vectors for upload
print("\n" + "="*80)
print("STEP 3: PREPARING VECTORS FOR UPLOAD")
print("="*80)

vectors_to_upsert = []

for i, row in products.iterrows():
    vectors_to_upsert.append({
        "id": str(row['StockCode']),
        "values": embeddings[i].tolist(),
        "metadata": {
            "description": str(row['Description']),
            "price": float(row['UnitPrice']),
            "country": str(row['Country']),
            "stock_code": str(row['StockCode'])
        }
    })

print(f"✓ Prepared {len(vectors_to_upsert):,} vectors for upload")
print(f"\nSample vector:")
print(f"  ID: {vectors_to_upsert[0]['id']}")
print(f"  Description: {vectors_to_upsert[0]['metadata']['description']}")
print(f"  Price: ${vectors_to_upsert[0]['metadata']['price']:.2f}")
print(f"  Vector length: {len(vectors_to_upsert[0]['values'])}")

# Step 4: Upload in batches
print("\n" + "="*80)
print("STEP 4: UPLOADING TO PINECONE")
print("="*80)

BATCH_SIZE = 100
total_batches = (len(vectors_to_upsert) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Uploading {len(vectors_to_upsert):,} vectors in {total_batches} batches...")
print(f"Batch size: {BATCH_SIZE}")

uploaded = 0
for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
    batch = vectors_to_upsert[i:i+BATCH_SIZE]

    # Upload batch
    index.upsert(vectors=batch)

    uploaded += len(batch)
    batch_num = (i // BATCH_SIZE) + 1

    if batch_num % 10 == 0 or batch_num == total_batches:
        print(f"  ✓ Batch {batch_num}/{total_batches} - Uploaded {uploaded:,}/{len(vectors_to_upsert):,} vectors ({uploaded/len(vectors_to_upsert)*100:.1f}%)")

    # Small delay to avoid rate limiting
    time.sleep(0.1)

print(f"\n✓ All vectors uploaded successfully!")

# Step 5: Verify upload
print("\n" + "="*80)
print("STEP 5: VERIFYING UPLOAD")
print("="*80)

# Wait a moment for indexing
print("Waiting 5 seconds for indexing...")
time.sleep(5)

final_stats = index.describe_index_stats()
print(f"\n✓ Final vector count: {final_stats['total_vector_count']:,}")
print(f"✓ Dimension: {final_stats['dimension']}")
print(f"✓ Index fullness: {final_stats.get('index_fullness', 0):.4f}")

if final_stats['total_vector_count'] == len(vectors_to_upsert):
    print(f"\n✓✓✓ SUCCESS: All {len(vectors_to_upsert):,} vectors uploaded correctly!")
else:
    print(f"\n⚠️  Warning: Expected {len(vectors_to_upsert):,}, got {final_stats['total_vector_count']:,}")

# Step 6: Test query
print("\n" + "="*80)
print("STEP 6: TESTING VECTOR SEARCH")
print("="*80)

# Load model for test query
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

test_query = "pink lunch bag"
print(f"\nTest query: '{test_query}'")

# Generate query embedding
query_vector = model.encode([test_query], normalize_embeddings=True)[0].tolist()

# Search
results = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True
)

print(f"\nTop 5 results:")
for i, match in enumerate(results['matches'], 1):
    print(f"\n{i}. Score: {match['score']:.4f}")
    print(f"   Product: {match['metadata']['description']}")
    print(f"   Price: ${match['metadata']['price']:.2f}")
    print(f"   StockCode: {match['metadata']['stock_code']}")

print("\n" + "="*80)
print("PINECONE UPLOAD COMPLETE!")
print("="*80)
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nIndex: {INDEX_NAME}")
print(f"Total vectors: {final_stats['total_vector_count']:,}")
print(f"Ready for product recommendation queries!")
print("="*80)
