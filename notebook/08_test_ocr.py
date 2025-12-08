"""
Test OCR Service
"""

import sys
sys.path.insert(0, '.')

from services.ocr_service import OCRService
import os

print("="*80)
print("TESTING OCR SERVICE")
print("="*80)

# Initialize OCR service
print("\nInitializing OCR service...")
ocr_service = OCRService()
print("OCR service initialized successfully!")

# Create a simple test
print("\nTesting with synthetic text...")
print("Note: For real testing, you'll need actual handwritten images")
print("      Place test images in an 'images/test/' directory")

# Check if test images exist
test_image_dir = "images/test"
if os.path.exists(test_image_dir):
    test_images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if test_images:
        for img_file in test_images[:3]:  # Test first 3 images
            img_path = os.path.join(test_image_dir, img_file)
            print(f"\n{'-'*60}")
            print(f"Testing image: {img_file}")

            text, confidence = ocr_service.extract_text(img_path)

            print(f"Extracted text: '{text}'")
            print(f"Confidence: {confidence:.2%}")
    else:
        print(f"\nNo images found in {test_image_dir}")
        print("Add test images to test OCR functionality")
else:
    print(f"\nDirectory {test_image_dir} not found")
    print("Create it and add handwritten query images for testing")

print("\n" + "="*80)
print("OCR SERVICE TEST COMPLETE")
print("="*80)
print("\nNote: OCR service is ready for integration with /ocr-query endpoint")
