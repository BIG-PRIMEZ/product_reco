"""
Augment product images to reach target count for CNN training
"""
import sys
sys.path.insert(0, '.')

from services.augmentation_service import AugmentationService
from pathlib import Path
import pandas as pd

# Initialize augmentation service
augmenter = AugmentationService()

# Get all product directories
base_dir = Path('data/product_images')
product_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

print(f"Found {len(product_dirs)} product directories")
print(f"Target: 150 images per product\n")

results = {}

for product_dir in sorted(product_dirs):
    product_name = product_dir.name
    print(f"Processing: {product_name}")

    # Augment to reach 150 images per product
    final_count = augmenter.augment_product_directory(product_dir, target_per_product=150)

    results[product_name] = final_count
    print()

# Print summary
print("="*70)
print("AUGMENTATION SUMMARY")
print("="*70)
for product_name, count in sorted(results.items()):
    print(f"{product_name}: {count} images")

total_images = sum(results.values())
print(f"\nTotal Images: {total_images}")

# Update CNN_Model_Train_Data.csv with all images
print("\nUpdating CNN_Model_Train_Data.csv...")

# Load stock code mapping
stock_codes_df = pd.read_csv('data/CNN_Model_Train_Data_cleaned.csv')
products_df = pd.read_csv('data/dataset_cleaned.csv')

# Create mapping of product descriptions to stock codes
product_mapping = {}
for stock_code in stock_codes_df['StockCode'].tolist():
    product_row = products_df[products_df['StockCode'] == stock_code].iloc[0]
    description = product_row['Description']
    product_mapping[description] = stock_code

# Create CSV data
mapping_data = []
for product_dir in product_dirs:
    product_name = product_dir.name

    # Find matching stock code
    stock_code = None
    for desc, code in product_mapping.items():
        # Match by sanitized name
        sanitized_desc = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in desc).strip()
        if sanitized_desc == product_name or desc == product_name:
            stock_code = code
            break

    if not stock_code:
        print(f"Warning: Could not find stock code for {product_name}")
        continue

    # Add all images for this product
    image_files = list(product_dir.glob('*.jpg'))
    for img_path in image_files:
        mapping_data.append({
            'StockCode': stock_code,
            'Description': list(product_mapping.keys())[list(product_mapping.values()).index(stock_code)],
            'ImagePath': str(img_path)
        })

# Save to CSV
mapping_df = pd.DataFrame(mapping_data)
mapping_df.to_csv('data/CNN_Model_Train_Data.csv', index=False)
print(f"Updated CNN_Model_Train_Data.csv with {len(mapping_df)} images")
