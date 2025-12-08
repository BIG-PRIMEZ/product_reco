"""
Test OCR with handwritten-style images
"""

import sys
sys.path.insert(0, '.')

from services.ocr_service import OCRService
import os
import requests

print("="*80)
print("TESTING OCR WITH HANDWRITTEN-STYLE IMAGES")
print("="*80)

# Initialize OCR service
ocr_service = OCRService()

# Test with local images first
test_images_dir = "images/test"
if os.path.exists(test_images_dir):
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith('.png')]

    if test_images:
        print(f"\nFound {len(test_images)} test images\n")

        for img_file in sorted(test_images):
            img_path = os.path.join(test_images_dir, img_file)
            print(f"{'-'*60}")
            print(f"Image: {img_file}")

            # Extract text with OCR
            text, confidence = ocr_service.extract_text(img_path)

            print(f"Extracted: '{text}'")
            print(f"Confidence: {confidence:.1%}")

            # Test via endpoint
            print("\nTesting via /ocr-query endpoint...")
            with open(img_path, 'rb') as f:
                response = requests.post(
                    "http://127.0.0.1:5000/ocr-query",
                    files={"image_data": f}
                )

            if response.status_code == 200:
                result = response.json()
                print(f"Status: {response.status_code}")
                print(f"Products found: {len(result['products'])}")

                if result['products']:
                    print(f"Top match: {result['products'][0]['Description']}")
                    print(f"Price: {result['products'][0]['UnitPrice']}")
            else:
                print(f"Status: {response.status_code}")
                print(f"Error: {response.json().get('response', 'Unknown error')}")

            print()
    else:
        print(f"\nNo test images found in {test_images_dir}")
        print("Run: python3 notebook/12_create_handwritten_test.py")
else:
    print(f"\nDirectory {test_images_dir} not found")
    print("Run: python3 notebook/12_create_handwritten_test.py")

print("="*80)
print("OCR IMAGE TESTING COMPLETE")
print("="*80)
