"""
Test Vector Database Service
This script tests the VectorDBService class.
"""

import sys
sys.path.append('.')

from services.vector_db_service import VectorDBService

print("="*80)
print("TESTING VECTOR DATABASE SERVICE")
print("="*80)

# Initialize service
print("\n1. Initializing VectorDBService...")
service = VectorDBService()
print("✓ Service initialized")

# Get index stats
print("\n2. Index Statistics:")
stats = service.get_index_stats()
print(f"   Total vectors: {stats['total_vectors']:,}")
print(f"   Dimension: {stats['dimension']}")
print(f"   Index fullness: {stats['index_fullness']:.4f}")

# Test queries
test_queries = [
    "pink lunch bag",
    "red alarm clock",
    "chocolate hot water bottle",
    "spotty bunting decoration",
    "jumbo storage bag"
]

print("\n3. Testing Search Queries:")
for query in test_queries:
    print(f"\n   Query: '{query}'")

    # Validate query
    is_valid, error = service.validate_query(query)
    if not is_valid:
        print(f"   ✗ Invalid: {error}")
        continue

    # Search
    results = service.search_products(query, top_k=3, min_score=0.5)

    print(f"   Results: {len(results)} products found")
    for i, product in enumerate(results, 1):
        print(f"     {i}. {product['description']}")
        print(f"        Score: {product['similarity_score']:.4f} | Price: ${product['price']:.2f}")

# Test similar products
print("\n4. Testing Similar Products:")
stock_code = "22384"  # LUNCH BAG PINK POLKADOT
print(f"   Finding products similar to StockCode: {stock_code}")

similar = service.get_similar_products(stock_code, top_k=5)
print(f"   Found {len(similar)} similar products:")
for i, product in enumerate(similar, 1):
    print(f"     {i}. {product['description']}")
    print(f"        Score: {product['similarity_score']:.4f} | Price: ${product['price']:.2f}")

# Test query validation
print("\n5. Testing Query Validation:")
test_cases = [
    ("", "Empty query"),
    ("a", "Too short"),
    ("x" * 501, "Too long"),
    ("<script>alert('xss')</script>", "XSS attempt"),
    ("normal query", "Valid query")
]

for query, description in test_cases:
    is_valid, error = service.validate_query(query)
    status = "✓" if is_valid else "✗"
    result = "Valid" if is_valid else f"Invalid: {error}"
    print(f"   {status} {description}: {result}")

print("\n" + "="*80)
print("ALL TESTS COMPLETED!")
print("="*80)
