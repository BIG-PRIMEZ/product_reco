"""
Module 1 - Task 2: Pinecone Index Setup
This script creates a Pinecone index for product vectors.
"""

import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*80)
print("PINECONE INDEX SETUP")
print("="*80)

# Initialize Pinecone
api_key = os.getenv('PINECONE_API_KEY')
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in .env file")

print("\n1. Initializing Pinecone client...")
pc = Pinecone(api_key=api_key)

# Index configuration
INDEX_NAME = "ecommerce-products"
DIMENSION = 384  # For all-MiniLM-L6-v2 model
METRIC = "cosine"

print(f"\n2. Checking if index '{INDEX_NAME}' exists...")
existing_indexes = pc.list_indexes()
index_names = [index['name'] for index in existing_indexes]

if INDEX_NAME in index_names:
    print(f"   ⚠️  Index '{INDEX_NAME}' already exists!")
    print(f"   Deleting existing index...")
    pc.delete_index(INDEX_NAME)
    print(f"   ✓ Deleted")

print(f"\n3. Creating new index '{INDEX_NAME}'...")
print(f"   - Dimension: {DIMENSION}")
print(f"   - Metric: {METRIC}")
print(f"   - Cloud: AWS")
print(f"   - Region: us-east-1")

pc.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    metric=METRIC,
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

print(f"\n✓ Index '{INDEX_NAME}' created successfully!")

# Get index stats
index = pc.Index(INDEX_NAME)
stats = index.describe_index_stats()

print(f"\n4. Index Statistics:")
print(f"   - Total vectors: {stats['total_vector_count']}")
print(f"   - Dimension: {stats['dimension']}")
print(f"   - Index fullness: {stats.get('index_fullness', 0)}")

print("\n" + "="*80)
print("PINECONE INDEX SETUP COMPLETE!")
print("="*80)
print(f"\nIndex Name: {INDEX_NAME}")
print(f"Ready to upload product vectors!")
