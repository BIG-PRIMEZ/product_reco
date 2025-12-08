"""
Quick OCR Test - Check if OCR is functional
"""

import sys
sys.path.insert(0, '.')

print("="*80)
print("QUICK OCR FUNCTIONALITY TEST")
print("="*80)

try:
    print("\n1. Importing OCR service...")
    from services.ocr_service import OCRService
    print("   ✓ Import successful")

    print("\n2. Initializing OCR service (may take time on first run)...")
    print("   Note: This downloads ~100MB of models on first initialization")
    ocr_service = OCRService()
    print("   ✓ OCR service initialized")

    print("\n3. Creating test image...")
    from PIL import Image, ImageDraw
    import io

    img = Image.new('RGB', (300, 80), color='white')
    d = ImageDraw.Draw(img)
    d.text((10, 25), "hello world", fill='black')

    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    print("   ✓ Test image created")

    print("\n4. Testing OCR extraction...")
    text, confidence = ocr_service.extract_text_from_bytes(img_bytes.getvalue())
    print(f"   Extracted: '{text}'")
    print(f"   Confidence: {confidence:.2%}")

    if text:
        print("\n" + "="*80)
        print("✓ OCR IS FUNCTIONAL!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠ OCR initialized but no text extracted (may be font issue)")
        print("="*80)

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\n" + "="*80)
    print("OCR NOT YET FUNCTIONAL")
    print("="*80)
    print("\nPossible reasons:")
    print("- Models still downloading (check internet connection)")
    print("- First run takes 2-5 minutes to download models")
    print("- Network timeout (try again)")
