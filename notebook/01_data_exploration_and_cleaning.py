"""
Module 1 - Task 1: E-commerce Dataset Exploration and Cleaning
This script explores the dataset and identifies all data quality issues.
"""

import pandas as pd
import numpy as np
import re

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/dataset.csv')

print(f"\n{'='*80}")
print("DATASET EXPLORATION")
print(f"{'='*80}")

# Basic information
print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nDataset shape: {df.shape}")

# Display first few rows
print(f"\n{'='*80}")
print("SAMPLE DATA (First 10 rows)")
print(f"{'='*80}")
print(df.head(10))

# Data types
print(f"\n{'='*80}")
print("DATA TYPES")
print(f"{'='*80}")
print(df.dtypes)

# Missing values analysis
print(f"\n{'='*80}")
print("MISSING VALUES ANALYSIS")
print(f"{'='*80}")
missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_count.index,
    'Missing Count': missing_count.values,
    'Percentage': missing_percent.values
})
print(missing_df[missing_df['Missing Count'] > 0])

# Duplicate analysis
print(f"\n{'='*80}")
print("DUPLICATE ANALYSIS")
print(f"{'='*80}")
duplicate_rows = df.duplicated().sum()
print(f"Total duplicate rows: {duplicate_rows:,}")
print(f"Percentage of duplicates: {(duplicate_rows/len(df)*100):.2f}%")

# Unique values per column
print(f"\n{'='*80}")
print("UNIQUE VALUES PER COLUMN")
print(f"{'='*80}")
for col in df.columns:
    print(f"{col}: {df[col].nunique():,} unique values")

# Detect data quality issues in each column
print(f"\n{'='*80}")
print("DATA QUALITY ISSUES DETECTED")
print(f"{'='*80}")

# InvoiceNo issues
print("\n1. InvoiceNo Column:")
invoice_with_special = df[df['InvoiceNo'].astype(str).str.contains('[^0-9C]', regex=True, na=False)]
print(f"   - Rows with special characters (ä, etc.): {len(invoice_with_special):,}")
print(f"   - Sample: {invoice_with_special['InvoiceNo'].head(3).tolist()}")

# StockCode issues
print("\n2. StockCode Column:")
stock_with_special = df[df['StockCode'].astype(str).str.contains('[öä^]', regex=True, na=False)]
print(f"   - Rows with special characters (ö, ^, etc.): {len(stock_with_special):,}")
print(f"   - Sample: {stock_with_special['StockCode'].head(5).tolist()}")

# Description issues
print("\n3. Description Column:")
desc_with_dollar = df[df['Description'].astype(str).str.startswith('$', na=False)]
print(f"   - Descriptions starting with '$': {len(desc_with_dollar):,}")
desc_with_special = df[df['Description'].astype(str).str.contains('[öä]', na=False)]
print(f"   - Descriptions with special chars: {len(desc_with_special):,}")

# Quantity issues
print("\n4. Quantity Column:")
qty_with_at = df[df['Quantity'].astype(str).str.contains('@', na=False)]
print(f"   - Quantities with '@' symbol: {len(qty_with_at):,}")
print(f"   - Sample: {qty_with_at['Quantity'].head(3).tolist()}")
try:
    qty_numeric = pd.to_numeric(df['Quantity'], errors='coerce')
    negative_qty = (qty_numeric < 0).sum()
    print(f"   - Negative quantities: {negative_qty:,}")
except:
    print("   - Cannot convert to numeric - contains non-numeric values")

# UnitPrice issues
print("\n5. UnitPrice Column:")
price_with_ww = df[df['UnitPrice'].astype(str).str.contains('Ww', na=False)]
print(f"   - Prices with 'Ww' prefix: {len(price_with_ww):,}")
print(f"   - Sample: {price_with_ww['UnitPrice'].head(3).tolist()}")
try:
    price_numeric = pd.to_numeric(df['UnitPrice'], errors='coerce')
    negative_price = (price_numeric < 0).sum()
    zero_price = (price_numeric == 0).sum()
    print(f"   - Negative prices: {negative_price:,}")
    print(f"   - Zero prices: {zero_price:,}")
except:
    print("   - Cannot convert to numeric - contains non-numeric values")

# CustomerID issues
print("\n6. CustomerID Column:")
customer_with_ampersand = df[df['CustomerID'].astype(str).str.contains('[&#]', na=False)]
print(f"   - CustomerIDs with '&' or '#': {len(customer_with_ampersand):,}")
print(f"   - Sample: {customer_with_ampersand['CustomerID'].head(3).tolist()}")

# Country issues
print("\n7. Country Column:")
country_with_prefix = df[df['Country'].astype(str).str.contains('XxY', na=False)]
print(f"   - Countries with 'XxY' prefix: {len(country_with_prefix):,}")
country_with_emoji = df[df['Country'].astype(str).str.contains('☺️', na=False)]
print(f"   - Countries with emoji: {len(country_with_emoji):,}")
print(f"   - Unique countries: {df['Country'].nunique()}")
print(f"   - Sample countries: {df['Country'].unique()[:10].tolist()}")

# Statistical summary
print(f"\n{'='*80}")
print("STATISTICAL SUMMARY")
print(f"{'='*80}")
print(df.describe())

print(f"\n{'='*80}")
print("CLEANING STRATEGY SUMMARY")
print(f"{'='*80}")
print("""
Based on the analysis, we need to:
1. Remove special characters (ä) from InvoiceNo
2. Remove special characters (ö, ^) from StockCode
3. Remove '$' prefix from Description
4. Remove '@' symbol from Quantity and convert to numeric
5. Remove 'Ww' prefix from UnitPrice and convert to numeric
6. Remove '&' and '#' from CustomerID and convert to numeric
7. Remove 'XxY' prefix and emoji from Country
8. Handle missing values in CustomerID and Description
9. Remove duplicate rows
10. Remove rows with negative or zero prices/quantities (if returns)
11. Standardize country names
""")
