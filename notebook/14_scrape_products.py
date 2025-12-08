"""
Scrape product images from Google Images using Selenium
Downloads 150 images per product for CNN training
"""
import sys
sys.path.insert(0, '.')

from services.scraper_service import ScraperService
import pandas as pd
import time

# Load stock codes for CNN training
stock_codes_df = pd.read_csv('data/CNN_Model_Train_Data_cleaned.csv')
stock_codes = stock_codes_df['StockCode'].tolist()

# Load product descriptions
products_df = pd.read_csv('data/dataset_cleaned.csv')

# Create mapping of stock codes to descriptions
product_mapping = {}
for stock_code in stock_codes:
    # Get first description for this stock code
    product_row = products_df[products_df['StockCode'] == stock_code].iloc[0]
    description = product_row['Description']
    product_mapping[stock_code] = description

print(f"Found {len(product_mapping)} products to scrape:")
for stock_code, description in product_mapping.items():
    print(f"  {stock_code}: {description}")

# Initialize scraper
scraper = ScraperService(base_dir="data/product_images", headless=True)

# Scrape images for each product
results = {}
for stock_code, description in product_mapping.items():
    print(f"\n{'='*70}")
    print(f"Processing: {description} (Stock: {stock_code})")
    print(f"{'='*70}")

    try:
        # Scrape images
        image_paths = scraper.scrape_google_images(description, max_images=150)

        # Validate images
        product_dir = scraper.base_dir / scraper._sanitize_filename(description)
        valid_count, invalid_files = scraper.validate_images(product_dir)

        results[stock_code] = {
            'description': description,
            'downloaded': len(image_paths),
            'valid': valid_count,
            'invalid': len(invalid_files)
        }

        print(f"\n  Summary:")
        print(f"    Downloaded: {len(image_paths)} images")
        print(f"    Valid: {valid_count} images")
        print(f"    Invalid: {len(invalid_files)} images")

        # Sleep between products to avoid rate limiting
        time.sleep(3)

    except Exception as e:
        print(f"  Error processing {description}: {e}")
        results[stock_code] = {
            'description': description,
            'downloaded': 0,
            'valid': 0,
            'invalid': 0
        }

# Close browser
scraper.close()

# Print final summary
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")
for stock_code, result in results.items():
    print(f"{stock_code}: {result['description']}")
    print(f"  Downloaded: {result['downloaded']}, Valid: {result['valid']}, Invalid: {result['invalid']}")

total_downloaded = sum(r['downloaded'] for r in results.values())
total_valid = sum(r['valid'] for r in results.values())
print(f"\nTotal Images Downloaded: {total_downloaded}")
print(f"Total Valid Images: {total_valid}")

# Create CSV mapping for CNN training
print("\nCreating CNN_Model_Train_Data.csv...")
mapping_data = []
for stock_code, description in product_mapping.items():
    product_dir = scraper.base_dir / scraper._sanitize_filename(description)
    if product_dir.exists():
        image_files = list(product_dir.glob('*.jpg'))
        for img_path in image_files:
            mapping_data.append({
                'StockCode': stock_code,
                'Description': description,
                'ImagePath': str(img_path)
            })

mapping_df = pd.DataFrame(mapping_data)
mapping_df.to_csv('data/CNN_Model_Train_Data.csv', index=False)
print(f"Created CNN_Model_Train_Data.csv with {len(mapping_df)} images")
