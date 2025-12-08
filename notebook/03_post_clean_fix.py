"""
Post-cleaning fix: Ensure no missing values in Description column and CustomerID is integer
"""
import pandas as pd

print("Loading cleaned dataset...")
df = pd.read_csv('data/dataset_cleaned.csv')

print(f"Initial missing descriptions: {df['Description'].isnull().sum()}")
print(f"Initial CustomerID type: {df['CustomerID'].dtype}")

# Fill any remaining nulls in Description
df['Description'] = df['Description'].fillna('UNKNOWN')

# Convert CustomerID to Int64 (nullable integer to remove .0)
df['CustomerID'] = df['CustomerID'].astype('Int64')

# Save back
df.to_csv('data/dataset_cleaned.csv', index=False)

print(f"Final missing descriptions: {df['Description'].isnull().sum()}")
print(f"Final CustomerID type: {df['CustomerID'].dtype}")
print("✓ Dataset fixed and saved!")
