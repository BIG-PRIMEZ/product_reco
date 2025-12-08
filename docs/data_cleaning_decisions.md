# Data Cleaning Decisions

## Overview

We cleaned the e-commerce dataset from 541,909 rows down to 533,772 rows (lost only 1.5% of data). The dataset covers one year of transactions from December 2010 to December 2011.

## What We Fixed

### Special Characters Problem
About half the data had weird characters everywhere - `ä`, `ö`, `^`, `$`, `@`, `Ww`, `&`, `#`, `XxY`, and emoji. We removed all of them. For example:
- `536365ä` became `536365`
- `ö84406B^` became `84406B` (then `84406` after removing letters)
- `XxYUnited Kingdom☺️` became `United Kingdom`

### StockCode - Numbers Only
StockCodes had letters mixed with numbers like `85123A` or `84406B`. We decided to keep only the numbers, so:
- `85123A` → `85123`
- `84406B` → `84406`

This removed 2,796 rows where the StockCode was entirely letters.

### Missing Descriptions
1,025 products had no description. We filled these with "UNKNOWN" since we still have the StockCode to identify the product.

### Missing CustomerIDs
About 25% of transactions (133,992 rows) don't have a CustomerID. We kept these as null because they're probably guest purchases - perfectly normal in e-commerce.

### CustomerID Decimal Points
All CustomerIDs had `.0` at the end like `17850.0`. We converted them to clean integers: `17850`.

### Duplicates
Found and removed 5,341 duplicate rows - probably data entry mistakes.

### Negative Quantities
10,005 transactions have negative quantities. We kept them because they represent returns and refunds, which are important for analysis.

### Zero Prices
2,496 products have zero price. We kept them - they could be promotional items or free samples.

### Negative Prices
Only 2 rows had negative prices. These were clearly errors, so we removed them.

## Final Results

**Dataset**: 533,772 rows, 8 columns
**Data Loss**: 8,137 rows (1.5%)
**Missing Values**: Only CustomerID has nulls (by design)
**Special Characters**: All removed
**Data Types**: All correct

## Where the Data Lives

Most transactions are from the UK (91.5%). Germany, France, and Ireland make up most of the rest.

Top selling products are home decor items like heart t-light holders, cake stands, and retrospot bags.

## CNN Training Data

We also cleaned the CNN training dataset - 10 StockCodes that will be used to train the image recognition model. All 10 exist in the main dataset with plenty of transaction history.

## What's Ready

- `data/dataset_cleaned.csv` - Main dataset ready for vectorization
- `data/CNN_Model_Train_Data_cleaned.csv` - 10 StockCodes for CNN training

Everything is clean and ready for the next steps: building the vector database and training the CNN model.
