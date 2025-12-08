"""
Module 1 - Task 3: Test Similarity Metrics
Compare cosine, dotproduct, and euclidean distance metrics.
"""

import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time

load_dotenv()

print("="*80)
print("SIMILARITY METRICS COMPARISON")
print("="*80)

# Initialize
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test query
test_query = "pink lunch bag"
query_vector = model.encode([test_query], normalize_embeddings=True)[0].tolist()

print(f"\nTest Query: '{test_query}'")

# Metrics to test
metrics = ["cosine", "dotproduct", "euclidean"]

results_by_metric = {}

for metric in metrics:
    print(f"\n{'='*80}")
    print(f"Testing: {metric.upper()}")
    print(f"{'='*80}")

    index_name = f"test-{metric}"

    # Create temporary index
    print(f"Creating index with {metric} metric...")

    # Delete if exists
    existing = [idx['name'] for idx in pc.list_indexes()]
    if index_name in existing:
        pc.delete_index(index_name)
        time.sleep(1)

    # Create new index
    pc.create_index(
        name=index_name,
        dimension=384,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    # Wait for index to be ready
    time.sleep(5)

    index = pc.Index(index_name)

    # Upload sample vectors from main index
    print("Uploading sample vectors...")
    main_index = pc.Index("ecommerce-products")

    # Fetch some vectors from main index
    sample_ids = ["22384", "22727", "22112", "23298", "20726"]
    fetch_result = main_index.fetch(ids=sample_ids)

    # Upload to test index
    vectors_to_upload = []
    for vec_id, vec_data in fetch_result['vectors'].items():
        vectors_to_upload.append({
            "id": vec_id,
            "values": vec_data['values'],
            "metadata": vec_data['metadata']
        })

    index.upsert(vectors=vectors_to_upload)
    time.sleep(2)

    # Query
    print(f"Querying with {metric}...")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    # Store results
    results_by_metric[metric] = results['matches']

    # Display
    print(f"\nTop 3 results:")
    for i, match in enumerate(results['matches'], 1):
        print(f"{i}. {match['metadata']['description']}")
        print(f"   Score: {match['score']:.4f}")

    # Cleanup
    print(f"\nCleaning up {index_name}...")
    pc.delete_index(index_name)
    time.sleep(1)

# Summary comparison
print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}")

print(f"\nQuery: '{test_query}'")
print(f"Expected: LUNCH BAG PINK POLKADOT")

for metric in metrics:
    top_result = results_by_metric[metric][0]
    print(f"\n{metric.upper()}:")
    print(f"  Winner: {top_result['metadata']['description']}")
    print(f"  Score: {top_result['score']:.4f}")
    is_correct = "PINK POLKADOT" in top_result['metadata']['description']
    print(f"  Correct: {'✓' if is_correct else '✗'}")

print(f"\n{'='*80}")
print("RECOMMENDATION: COSINE")
print(f"{'='*80}")
print("Cosine similarity is best for text embeddings because:")
print("1. Measures semantic similarity (angle between vectors)")
print("2. Normalized scores (0-1 range)")
print("3. Not affected by vector magnitude")
print("4. Standard for sentence-transformers models")
print("="*80)
