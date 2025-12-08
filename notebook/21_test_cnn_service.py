"""
Test CNN service with sample images
"""
import sys
sys.path.insert(0, '.')

from services.cnn_service import CNNService
from pathlib import Path

print("="*70)
print("CNN SERVICE TEST")
print("="*70)

# Initialize service
print("\nInitializing CNN service...")
cnn_service = CNNService()

print("\n" + "="*70)
print("AVAILABLE CLASSES")
print("="*70)
all_classes = cnn_service.get_all_classes()
for cls in all_classes:
    print(f"Class {cls['class_id']}: {cls['stock_code']} - {cls['product_name']}")

# Test with a few sample images from test set
print("\n" + "="*70)
print("TESTING PREDICTIONS")
print("="*70)

test_dir = Path('data/cnn_dataset/test')

# Get one image from each class
sample_images = []
for class_dir in sorted(test_dir.iterdir()):
    if class_dir.is_dir():
        # Get first image from this class
        images = list(class_dir.glob('*.jpg'))
        if images:
            sample_images.append(images[0])

print(f"\nTesting with {len(sample_images)} sample images...\n")

correct = 0
total = 0

for img_path in sample_images:
    # Get true class from directory name (stock code)
    true_stock_code = img_path.parent.name

    # Get prediction
    result = cnn_service.predict(str(img_path), return_top_k=3)

    total += 1
    is_correct = result['predicted_stock_code'] == true_stock_code
    if is_correct:
        correct += 1

    print(f"Image: {img_path.name}")
    print(f"  True Class: {true_stock_code}")
    print(f"  Predicted: {result['predicted_stock_code']} - {result['predicted_product']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

    if len(result['top_predictions']) > 1:
        print(f"  Top 3 predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"    {i}. {pred['stock_code']} ({pred['confidence']:.2%})")
    print()

print("="*70)
print("TEST SUMMARY")
print("="*70)
print(f"Tested: {total} images")
print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
print("="*70)
