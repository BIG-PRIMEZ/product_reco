"""
Test /image-product-search endpoint with sample product images
"""
import requests
from pathlib import Path
import json

print("="*70)
print("IMAGE PRODUCT SEARCH ENDPOINT TEST")
print("="*70)

# API endpoint
url = "http://127.0.0.1:5000/image-product-search"

# Get sample images from test set
test_dir = Path('data/cnn_dataset/test')

# Select one image from each class for testing
sample_images = []
class_names = {}

for class_dir in sorted(test_dir.iterdir()):
    if class_dir.is_dir():
        stock_code = class_dir.name
        images = list(class_dir.glob('*.jpg'))
        if images:
            sample_images.append((stock_code, images[0]))

print(f"\nTesting with {len(sample_images)} sample images")
print(f"API endpoint: {url}\n")

successful_tests = 0
failed_tests = 0

for stock_code, img_path in sample_images[:5]:  # Test first 5 for brevity
    print("-" * 70)
    print(f"Testing: {img_path.name}")
    print(f"True Class: {stock_code}")

    try:
        # Send POST request with image
        with open(img_path, 'rb') as f:
            files = {'product_image': f}
            response = requests.post(url, files=files)

        if response.status_code == 200:
            result = response.json()

            print(f"Status: SUCCESS (200)")
            print(f"\nResponse: {result['response']}")

            if 'cnn_prediction' in result:
                cnn = result['cnn_prediction']
                print(f"\nCNN Prediction:")
                print(f"  Product: {cnn['predicted_class']}")
                print(f"  Stock Code: {cnn['stock_code']}")
                print(f"  Confidence: {cnn['confidence']}")

                if 'top_predictions' in cnn and len(cnn['top_predictions']) > 1:
                    print(f"  Alternatives:")
                    for pred in cnn['top_predictions'][1:]:
                        print(f"    - {pred['product']} ({pred['confidence']})")

            products = result.get('products', [])
            print(f"\nFound {len(products)} similar products")
            if products:
                print(f"Top 3:")
                for i, p in enumerate(products[:3], 1):
                    print(f"  {i}. {p['Description']} - {p['UnitPrice']}")

            # Check if prediction matches true class
            if 'cnn_prediction' in result:
                predicted = result['cnn_prediction']['stock_code']
                if predicted == stock_code:
                    print(f"\n✓ CNN prediction CORRECT")
                    successful_tests += 1
                else:
                    print(f"\n✗ CNN prediction INCORRECT (predicted {predicted})")
                    failed_tests += 1
            else:
                failed_tests += 1

        else:
            print(f"Status: ERROR ({response.status_code})")
            print(f"Response: {response.json()}")
            failed_tests += 1

    except Exception as e:
        print(f"Status: EXCEPTION")
        print(f"Error: {e}")
        failed_tests += 1

    print()

print("="*70)
print("TEST SUMMARY")
print("="*70)
print(f"Total Tests: {successful_tests + failed_tests}")
print(f"Successful: {successful_tests}")
print(f"Failed: {failed_tests}")
print("="*70)
