# Web Scraping Strategy for Product Images

## Overview
This document explains how we collected images for training the CNN model. We needed 100-150 images per product for 10 different product categories.

## Image Source
**Google Images** - We chose Google Images because:
- It has a huge collection of product images
- Images are diverse (different angles, backgrounds, lighting)
- It's free and accessible
- No API key required

## How We Scraped the Images

### Tools Used
- **Selenium WebDriver** - Automates a web browser to visit Google Images
- **Chrome (Headless Mode)** - Runs browser without showing the window
- **webdriver-manager** - Automatically downloads and manages ChromeDriver

### The Scraping Process

1. **Navigate to Google Images**
   - Search for each product by name (e.g., "LUNCH BAG PINK POLKADOT")
   - Example URL: `https://www.google.com/search?q=LUNCH+BAG+PINK+POLKADOT&tbm=isch`

2. **Load More Images**
   - Scroll down the page 10 times to load more image thumbnails
   - Wait 0.8 seconds between each scroll
   - Try to click "Show more results" button if it appears

3. **Find Image Thumbnails**
   - Look for all `<img>` tags on the page
   - Filter to only Google Images thumbnails (URLs containing "encrypted-tbn")
   - Skip logos and other non-product images

4. **Download Images**
   - Get the image URL from the thumbnail
   - Download the image using HTTP request
   - Validate the image (must be at least 100x100 pixels)
   - Convert to RGB format and save as JPEG
   - Add rate limiting (0.3 seconds between downloads) to avoid being blocked

5. **Organize by Product**
   - Create a folder for each product
   - Save images with numbered filenames and hash for uniqueness
   - Example: `0001_45140f56.jpg`

### Rate Limiting & Safety
- Wait 0.3 seconds between downloading each image
- Wait 3 seconds between scraping different products
- Use realistic browser user agent
- Run in headless mode to reduce resource usage

## Results from Initial Scraping

We scraped **187 original images** from Google Images:
- LUNCH BAG PINK POLKADOT: 19 images
- ALARM CLOCK BAKELIKE RED: 17 images
- CHOCOLATE HOT WATER BOTTLE: 19 images
- SPOTTY BUNTING: 17 images
- LUNCH BAG WOODLAND: 20 images
- REX CASH+CARRY JUMBO SHOPPER: 20 images
- JUMBO STORAGE BAG SUKI: 20 images
- RETROSPOT TEA SET CERAMIC 11 PC: 20 images
- 6 RIBBONS RUSTIC CHARM: 19 images
- REGENCY CAKESTAND 3 TIER: 16 images

### Why Only ~20 Images Per Product?
Google Images only loads about 20 thumbnail images initially, even with aggressive scrolling. To load more, you typically need to click on individual images to view full versions, which is slower and more complex.

## Data Augmentation Solution

Since we only got 187 images (not enough for good CNN training), we used **data augmentation** to create more training examples from the original images.

### Augmentation Techniques Applied

For each original image, we created 6-8 variations using:

1. **Rotations**
   - Rotate 90° clockwise
   - Rotate 180°
   - Rotate 270° clockwise

2. **Flips**
   - Flip horizontally (mirror image)
   - Flip vertically

3. **Brightness Adjustments**
   - Increase brightness by 30%
   - Decrease brightness by 30%

4. **Contrast Adjustments**
   - Increase contrast by 30%
   - Decrease contrast by 30%

5. **Filters**
   - Apply blur effect
   - Apply sharpening effect

### Final Dataset

After augmentation, we achieved **1,375 total images**:
- LUNCH BAG WOODLAND: 140 images
- REX CASH+CARRY JUMBO SHOPPER: 140 images
- JUMBO STORAGE BAG SUKI: 140 images
- 6 RIBBONS RUSTIC CHARM: 133 images
- CHOCOLATE HOT WATER BOTTLE: 133 images
- RETROSPOT TEA SET CERAMIC 11 PC: 140 images
- LUNCH BAG PINK POLKADOT: 133 images
- REGENCY CAKESTAND 3 TIER: 144 images
- ALARM CLOCK BAKELIKE RED: 136 images
- SPOTTY BUNTING: 136 images

Each product has **133-144 images**, which is sufficient for CNN training.

## File Organization

```
data/product_images/
├── LUNCH BAG PINK POLKADOT/
│   ├── 0000_be52db9d.jpg          (original)
│   ├── 0001_45140f56.jpg          (original)
│   ├── 0020_aug_90056ff7.jpg      (augmented)
│   └── ...
├── ALARM CLOCK BAKELIKE RED/
│   └── ...
└── ...
```

## Training Data CSV

All images are cataloged in `data/CNN_Model_Train_Data.csv`:

```csv
StockCode,Description,ImagePath
22384,LUNCH BAG PINK POLKADOT,data/product_images/LUNCH BAG PINK POLKADOT/0000_be52db9d.jpg
22384,LUNCH BAG PINK POLKADOT,data/product_images/LUNCH BAG PINK POLKADOT/0001_45140f56.jpg
...
```

This CSV maps each image to its product (StockCode and Description) for training the CNN model.

## Scripts Created

1. **services/scraper_service.py** - Selenium scraper for Google Images
2. **services/augmentation_service.py** - Image augmentation with 11 techniques
3. **notebook/14_scrape_products.py** - Main script to scrape all 10 products
4. **notebook/17_augment_images.py** - Script to augment images to target count

## Summary

- **Source**: Google Images
- **Method**: Selenium browser automation
- **Original Images**: 187
- **Augmented Images**: 1,188
- **Total Dataset**: 1,375 images across 10 product classes
- **Average per Product**: 137.5 images
- **Ready for**: CNN model training in Module 3
