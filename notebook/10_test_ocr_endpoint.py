"""
Test OCR Query Endpoint
"""

import requests
from PIL import Image, ImageDraw, ImageFont
import io

BASE_URL = "http://127.0.0.1:5000"

print("="*80)
print("TESTING OCR QUERY ENDPOINT")
print("="*80)

# Create a simple test image with text
print("\nCreating test image with text 'pink lunch bag'...")
img = Image.new('RGB', (400, 100), color='white')
d = ImageDraw.Draw(img)

# Try to use a font, fallback to default if not available
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
except:
    font = ImageFont.load_default()

d.text((20, 30), "pink lunch bag", fill='black', font=font)

# Save to bytes
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

# Test the endpoint
print("\nSending image to /ocr-query endpoint...")
print("-" * 60)

response = requests.post(
    f"{BASE_URL}/ocr-query",
    files={"image_data": ("test.png", img_bytes, "image/png")}
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
print("OCR ENDPOINT TEST COMPLETE")
print("="*80)
print("\nNote: For best results, test with actual handwritten images")
print("      Place images in 'images/test/' directory and modify script")
