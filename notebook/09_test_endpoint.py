"""
Test Product Recommendation Endpoint
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"

print("="*80)
print("TESTING PRODUCT RECOMMENDATION ENDPOINT")
print("="*80)

# Test queries
test_queries = [
    "pink lunch bag",
    "red alarm clock",
    "chocolate hot water bottle",
    "",  # Invalid: empty
    "a",  # Invalid: too short
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    print("-" * 60)

    response = requests.post(
        f"{BASE_URL}/product-recommendation",
        data={"query": query}
    )

    print(f"Status: {response.status_code}")

    result = response.json()
    print(f"Response: {result['response']}")
    print(f"Products found: {len(result['products'])}")

    if result['products']:
        print("\nTop 3 products:")
        for i, product in enumerate(result['products'][:3], 1):
            print(f"  {i}. {product['Description']}")
            print(f"     Price: {product['UnitPrice']} | Code: {product['StockCode']}")

print("\n" + "="*80)
print("TESTS COMPLETE")
print("="*80)
