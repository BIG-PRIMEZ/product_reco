"""
Test scraping one product to verify it works
"""
import sys
sys.path.insert(0, '.')

from services.scraper_service import ScraperService

# Initialize scraper
scraper = ScraperService(base_dir="data/product_images", headless=True)

# Test with one product
print("Testing with LUNCH BAG PINK POLKADOT...")
image_paths = scraper.scrape_google_images("LUNCH BAG PINK POLKADOT", max_images=20)

print(f"\nTotal downloaded: {len(image_paths)} images")
if image_paths:
    print("First 3 image paths:")
    for path in image_paths[:3]:
        print(f"  {path}")

# Close browser
scraper.close()
