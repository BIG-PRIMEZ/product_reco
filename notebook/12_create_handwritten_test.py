"""
Create handwritten-style test images for OCR testing
"""

from PIL import Image, ImageDraw, ImageFont
import random

def create_handwritten_style_image(text, filename):
    """Create an image with text that looks somewhat handwritten"""
    # Create larger image with white background
    img = Image.new('RGB', (600, 150), color='white')
    d = ImageDraw.Draw(img)

    # Try to use a handwriting-style font, fallback to default
    try:
        # Try system fonts that look handwritten
        fonts_to_try = [
            "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttc",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]
        font = None
        for font_path in fonts_to_try:
            try:
                font = ImageFont.truetype(font_path, 50)
                break
            except:
                continue
        if not font:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Add slight randomness to position for handwritten feel
    x_offset = random.randint(-5, 5)
    y_offset = random.randint(-5, 5)

    # Draw text with slight gray (not pure black)
    d.text((30 + x_offset, 50 + y_offset), text, fill=(40, 40, 40), font=font)

    # Add slight noise for realism
    pixels = img.load()
    for _ in range(100):
        x = random.randint(0, img.width - 1)
        y = random.randint(0, img.height - 1)
        pixels[x, y] = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))

    img.save(filename)
    print(f"Created: {filename}")

# Create test images
print("="*60)
print("CREATING HANDWRITTEN-STYLE TEST IMAGES")
print("="*60)

test_queries = [
    ("pink lunch bag", "images/test/query1_pink_lunch_bag.png"),
    ("red alarm clock", "images/test/query2_red_alarm_clock.png"),
    ("chocolate", "images/test/query3_chocolate.png"),
]

for text, filename in test_queries:
    create_handwritten_style_image(text, filename)

print("\n" + "="*60)
print("TEST IMAGES CREATED")
print("="*60)
print("\nYou can now test OCR with these images:")
print("python3 notebook/13_test_ocr_with_images.py")
