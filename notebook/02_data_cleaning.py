"""
Module 1 - Task 1: E-commerce Dataset Cleaning
This script performs comprehensive data cleaning on the e-commerce dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("E-COMMERCE DATASET CLEANING SCRIPT")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# STEP 1: LOAD THE DATASET
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATASET")
print("="*80)

df = pd.read_csv('data/dataset.csv')
print(f"✓ Dataset loaded successfully")
print(f"  Initial shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Store original statistics for comparison
original_rows = len(df)
original_dtypes = df.dtypes.to_dict()
original_missing = df.isnull().sum().to_dict()
original_duplicates = df.duplicated().sum()

# ============================================================================
# STEP 2: REMOVE SPECIAL CHARACTERS
# ============================================================================
print("\n" + "="*80)
print("STEP 2: REMOVING SPECIAL CHARACTERS")
print("="*80)

# InvoiceNo: Remove 'ä'
invoice_before = df['InvoiceNo'].str.contains('ä', na=False).sum()
df['InvoiceNo'] = df['InvoiceNo'].str.replace('ä', '', regex=False)
print(f"✓ InvoiceNo: Cleaned {invoice_before:,} rows (removed 'ä')")

# StockCode: Remove ALL special characters and letters, keep only numbers
stock_before = df['StockCode'].str.contains('[^0-9]', regex=True, na=False).sum()
# Remove everything except digits (0-9)
df['StockCode'] = df['StockCode'].str.replace('[^0-9]', '', regex=True)
print(f"✓ StockCode: Cleaned {stock_before:,} rows (removed all special characters and letters, kept only numbers)")

# Description: Remove '$'
desc_before = df['Description'].str.contains('$', regex=False, na=False).sum()
df['Description'] = df['Description'].str.replace('$', '', regex=False)
print(f"✓ Description: Cleaned {desc_before:,} rows (removed '$')")

# Quantity: Remove '@'
qty_before = df['Quantity'].str.contains('@', na=False).sum()
df['Quantity'] = df['Quantity'].str.replace('@', '', regex=False)
print(f"✓ Quantity: Cleaned {qty_before:,} rows (removed '@')")

# UnitPrice: Remove 'Ww'
price_before = df['UnitPrice'].str.contains('Ww', na=False).sum()
df['UnitPrice'] = df['UnitPrice'].str.replace('Ww', '', regex=False)
print(f"✓ UnitPrice: Cleaned {price_before:,} rows (removed 'Ww')")

# CustomerID: Remove '&' and '#'
customer_before = df['CustomerID'].str.contains('[&#]', regex=True, na=False).sum()
df['CustomerID'] = df['CustomerID'].str.replace('&', '', regex=False)
df['CustomerID'] = df['CustomerID'].str.replace('#', '', regex=False)
print(f"✓ CustomerID: Cleaned {customer_before:,} rows (removed '&' and '#')")

# Country: Remove 'XxY' and emoji
country_before = df['Country'].str.contains('XxY', na=False).sum()
df['Country'] = df['Country'].str.replace('XxY', '', regex=False)
df['Country'] = df['Country'].str.replace('☺️', '', regex=False)
print(f"✓ Country: Cleaned {country_before:,} rows (removed 'XxY' and emoji)")

# ============================================================================
# STEP 3: STANDARDIZE TEXT FIELDS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: STANDARDIZING TEXT FIELDS")
print("="*80)

# Trim whitespace from all string columns
df['InvoiceNo'] = df['InvoiceNo'].str.strip()
df['StockCode'] = df['StockCode'].str.strip()
df['Description'] = df['Description'].str.strip()
df['Country'] = df['Country'].str.strip()
print(f"✓ Trimmed whitespace from text columns")

# Standardize country names (title case)
df['Country'] = df['Country'].str.title()
print(f"✓ Standardized country names to title case")
print(f"  Unique countries: {df['Country'].nunique()}")

# ============================================================================
# STEP 4: CONVERT DATA TYPES
# ============================================================================
print("\n" + "="*80)
print("STEP 4: CONVERTING DATA TYPES")
print("="*80)

# Convert Quantity to integer
qty_errors_before = pd.to_numeric(df['Quantity'], errors='coerce').isnull().sum()
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
print(f"✓ Quantity converted to numeric ({qty_errors_before:,} conversion errors)")

# Convert UnitPrice to float
price_errors_before = pd.to_numeric(df['UnitPrice'], errors='coerce').isnull().sum()
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
print(f"✓ UnitPrice converted to numeric ({price_errors_before:,} conversion errors)")

# Convert CustomerID to float first, then to Int64 (nullable integer) to remove .0
customer_errors_before = pd.to_numeric(df['CustomerID'], errors='coerce').isnull().sum()
df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
df['CustomerID'] = df['CustomerID'].astype('Int64')  # Int64 allows NaN and removes .0
print(f"✓ CustomerID converted to integer ({customer_errors_before:,} null values, removed .0)")

# Convert InvoiceDate to datetime
date_errors_before = pd.to_datetime(df['InvoiceDate'], errors='coerce').isnull().sum()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
print(f"✓ InvoiceDate converted to datetime ({date_errors_before:,} conversion errors)")

print(f"\nData types after conversion:")
for col, dtype in df.dtypes.items():
    print(f"  {col}: {dtype}")

# ============================================================================
# STEP 5: HANDLE MISSING VALUES
# ============================================================================
print("\n" + "="*80)
print("STEP 5: HANDLING MISSING VALUES")
print("="*80)

missing_before = df.isnull().sum()
print("Missing values per column (before):")
for col in df.columns:
    if missing_before[col] > 0:
        pct = (missing_before[col] / len(df)) * 100
        print(f"  {col}: {missing_before[col]:,} ({pct:.2f}%)")

# StockCode: Remove rows where StockCode became empty after removing letters
stockcode_empty_before = (df['StockCode'].isnull() | (df['StockCode'].str.strip() == '')).sum()
df = df[df['StockCode'].notna() & (df['StockCode'].str.strip() != '')]
print(f"\n✓ Removed {stockcode_empty_before:,} rows with empty StockCode (all letters removed)")

# Description: Fill missing AND empty descriptions with 'UNKNOWN'
desc_missing = df['Description'].isnull().sum()
desc_empty = (df['Description'].str.strip() == '').sum() if df['Description'].dtype == 'object' else 0
df['Description'] = df['Description'].fillna('UNKNOWN')
df.loc[df['Description'].str.strip() == '', 'Description'] = 'UNKNOWN'
print(f"✓ Filled {desc_missing:,} missing and {desc_empty:,} empty descriptions with 'UNKNOWN'")

# CustomerID: Keep as NaN (guest purchases are legitimate)
customer_missing = df['CustomerID'].isnull().sum()
print(f"✓ Kept {customer_missing:,} missing CustomerIDs (guest purchases)")

# Remove rows with missing critical fields
critical_missing_before = df[['InvoiceNo', 'Quantity', 'UnitPrice']].isnull().any(axis=1).sum()
df = df.dropna(subset=['InvoiceNo', 'Quantity', 'UnitPrice'])
print(f"✓ Removed {critical_missing_before:,} rows with missing other critical fields")
print(f"  Current shape: {df.shape}")

# ============================================================================
# STEP 6: REMOVE DUPLICATES
# ============================================================================
print("\n" + "="*80)
print("STEP 6: REMOVING DUPLICATES")
print("="*80)

duplicates_before = df.duplicated().sum()
df = df.drop_duplicates()
duplicates_removed = duplicates_before - df.duplicated().sum()
print(f"✓ Removed {duplicates_removed:,} duplicate rows")
print(f"  Current shape: {df.shape}")

# ============================================================================
# STEP 7: REMOVE INVALID DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 7: REMOVING INVALID DATA")
print("="*80)

# Remove negative prices (data errors)
negative_prices = (df['UnitPrice'] < 0).sum()
df = df[df['UnitPrice'] >= 0]
print(f"✓ Removed {negative_prices:,} rows with negative prices")

# Keep zero prices (could be promotional items)
zero_prices = (df['UnitPrice'] == 0).sum()
print(f"✓ Kept {zero_prices:,} rows with zero prices (promotional items/samples)")

# Keep negative quantities (represent returns/refunds)
negative_qty = (df['Quantity'] < 0).sum()
print(f"✓ Kept {negative_qty:,} rows with negative quantities (returns/refunds)")

print(f"  Final shape: {df.shape}")

# ============================================================================
# STEP 8: DATA VALIDATION
# ============================================================================
print("\n" + "="*80)
print("STEP 8: DATA VALIDATION")
print("="*80)

# Check for remaining special characters
print("\nChecking for remaining special characters:")
special_chars_check = {
    'InvoiceNo (ä)': df['InvoiceNo'].str.contains('ä', na=False).sum(),
    'StockCode (ö, ^)': df['StockCode'].str.contains('[öä^]', regex=True, na=False).sum(),
    'Description ($)': df['Description'].str.contains('$', regex=False, na=False).sum(),
    'Quantity (@)': df['Quantity'].astype(str).str.contains('@', na=False).sum(),
    'Country (XxY, emoji)': df['Country'].str.contains('XxY|☺️', regex=True, na=False).sum(),
}

all_clean = True
for check, count in special_chars_check.items():
    status = "✓" if count == 0 else "✗"
    print(f"  {status} {check}: {count:,}")
    if count > 0:
        all_clean = False

if all_clean:
    print("\n✓ All special characters removed successfully!")
else:
    print("\n✗ Warning: Some special characters remain")

# Check data types
print("\nData type validation:")
expected_types = {
    'InvoiceNo': 'object',
    'StockCode': 'object',
    'Description': 'object',
    'Quantity': ['int64', 'float64'],
    'InvoiceDate': 'datetime64[ns]',
    'UnitPrice': 'float64',
    'CustomerID': ['Int64', 'float64'],  # Int64 is nullable integer
    'Country': 'object'
}

for col, expected in expected_types.items():
    actual = str(df[col].dtype)
    if isinstance(expected, list):
        is_correct = actual in expected
    else:
        is_correct = actual == expected
    status = "✓" if is_correct else "✗"
    print(f"  {status} {col}: {actual}")

# ============================================================================
# STEP 9: SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STEP 9: SUMMARY STATISTICS")
print("="*80)

print("\nBEFORE vs AFTER Comparison:")
print(f"  Total rows: {original_rows:,} → {len(df):,} (removed {original_rows - len(df):,})")
print(f"  Duplicates: {original_duplicates:,} → {df.duplicated().sum():,}")
print(f"  Missing CustomerIDs: {original_missing['CustomerID']:,} → {df['CustomerID'].isnull().sum():,}")
print(f"  Missing Descriptions: {original_missing['Description']:,} → {df['Description'].isnull().sum():,}")

print("\nNumeric columns summary:")
print(df[['Quantity', 'UnitPrice']].describe())

print("\nDate range:")
print(f"  Earliest: {df['InvoiceDate'].min()}")
print(f"  Latest: {df['InvoiceDate'].max()}")

print("\nTop 10 countries by transaction count:")
print(df['Country'].value_counts().head(10))

print("\nTop 10 products by transaction count:")
print(df['Description'].value_counts().head(10))

# ============================================================================
# STEP 10: SAVE CLEANED DATASET
# ============================================================================
print("\n" + "="*80)
print("STEP 10: SAVING CLEANED DATASET")
print("="*80)

output_path = 'data/dataset_cleaned.csv'

# Final cleanup: Ensure Description has no nulls or empty strings before saving
df['Description'] = df['Description'].fillna('UNKNOWN')
df.loc[df['Description'].astype(str).str.strip() == '', 'Description'] = 'UNKNOWN'

df.to_csv(output_path, index=False)
print(f"✓ Cleaned dataset saved to: {output_path}")
print(f"  File size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"  Final Description nulls: {df['Description'].isnull().sum()}")

# ============================================================================
# STEP 11: CLEAN CNN_Model_Train_Data.csv
# ============================================================================
print("\n" + "="*80)
print("STEP 11: CLEANING CNN MODEL TRAINING DATA")
print("="*80)

# Load CNN training data
cnn_file = 'data/CNN_Model_Train_Data.csv'
print(f"\nLoading CNN training data from: {cnn_file}")
df_cnn = pd.read_csv(cnn_file)
print(f"✓ CNN data loaded successfully")
print(f"  Initial shape: {df_cnn.shape}")
print(f"  Columns: {list(df_cnn.columns)}")

# Store original stats
cnn_original_rows = len(df_cnn)

# Display sample data before cleaning
print(f"\nSample StockCodes BEFORE cleaning:")
print(f"  {df_cnn['StockCode'].head(10).tolist()}")

# Convert StockCode to string first (in case it's numeric)
df_cnn['StockCode'] = df_cnn['StockCode'].astype(str)

# Check for special characters
stock_with_special = df_cnn['StockCode'].str.contains('[^0-9]', regex=True, na=False).sum()
print(f"\n✓ StockCodes with non-numeric characters: {stock_with_special:,}")

# Clean StockCode: Remove ALL special characters and letters, keep only numbers
df_cnn['StockCode'] = df_cnn['StockCode'].str.replace('[^0-9]', '', regex=True)
print(f"✓ Removed all special characters and letters from StockCode (kept only numbers)")

# Trim whitespace
df_cnn['StockCode'] = df_cnn['StockCode'].str.strip()
print(f"✓ Trimmed whitespace")

# Remove empty rows
df_cnn = df_cnn[df_cnn['StockCode'].notna() & (df_cnn['StockCode'] != '')]
print(f"✓ Removed empty rows")

# Remove duplicates
cnn_duplicates = df_cnn.duplicated().sum()
df_cnn = df_cnn.drop_duplicates()
print(f"✓ Removed {cnn_duplicates:,} duplicate StockCodes")

# Display sample data after cleaning
print(f"\nSample StockCodes AFTER cleaning:")
print(f"  {df_cnn['StockCode'].head(10).tolist()}")

# Validation
print(f"\nValidation:")
remaining_special = df_cnn['StockCode'].str.contains('[^0-9]', regex=True, na=False).sum()
status = "✓" if remaining_special == 0 else "✗"
print(f"  {status} StockCodes with non-numeric characters: {remaining_special:,}")

# Save cleaned CNN data
cnn_output_path = 'data/CNN_Model_Train_Data_cleaned.csv'
df_cnn.to_csv(cnn_output_path, index=False)
print(f"\n✓ Cleaned CNN training data saved to: {cnn_output_path}")
print(f"  Total rows: {cnn_original_rows:,} → {len(df_cnn):,}")
print(f"  Unique StockCodes: {df_cnn['StockCode'].nunique():,}")

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "="*80)
print("DATA CLEANING COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nCleaned datasets:")
print(f"  1. data/dataset_cleaned.csv - Main e-commerce dataset")
print(f"  2. data/CNN_Model_Train_Data_cleaned.csv - CNN training StockCodes")
print(f"\nAll data ready for vectorization and model training.")
print("="*80)
