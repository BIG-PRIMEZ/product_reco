"""
Prepare CNN training data by splitting into train/validation/test sets
Split ratio: 70% train, 15% validation, 15% test
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import json

# Load the CNN training data
df = pd.read_csv('data/CNN_Model_Train_Data.csv')

print("="*70)
print("CNN DATA PREPARATION")
print("="*70)
print(f"\nTotal images: {len(df)}")
print(f"Total classes: {df['StockCode'].nunique()}")
print("\nImages per class:")
print(df.groupby('StockCode')['ImagePath'].count().sort_index())

# Create organized dataset directory structure
dataset_dir = Path('data/cnn_dataset')
dataset_dir.mkdir(exist_ok=True)

for split in ['train', 'val', 'test']:
    split_dir = dataset_dir / split
    split_dir.mkdir(exist_ok=True)

# Split data for each class to maintain class balance
train_data = []
val_data = []
test_data = []

class_mapping = {}
class_names = {}

for idx, (stock_code, group) in enumerate(df.groupby('StockCode')):
    # Get class name
    class_name = group.iloc[0]['Description']
    class_mapping[stock_code] = idx
    class_names[idx] = class_name

    # Create class directories
    for split in ['train', 'val', 'test']:
        class_dir = dataset_dir / split / str(stock_code)
        class_dir.mkdir(exist_ok=True)

    # Split data: 70% train, 15% val, 15% test
    images = group['ImagePath'].tolist()

    # First split: 70% train, 30% temp
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)

    # Second split: 50% val, 50% test (of the 30% temp = 15% each)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    print(f"\nClass {stock_code} ({class_name}):")
    print(f"  Total: {len(images)} images")
    print(f"  Train: {len(train_imgs)} images ({len(train_imgs)/len(images)*100:.1f}%)")
    print(f"  Val: {len(val_imgs)} images ({len(val_imgs)/len(images)*100:.1f}%)")
    print(f"  Test: {len(test_imgs)} images ({len(test_imgs)/len(images)*100:.1f}%)")

    # Copy images to respective directories
    for img_path in train_imgs:
        src = Path(img_path)
        dst = dataset_dir / 'train' / str(stock_code) / src.name
        shutil.copy2(src, dst)
        train_data.append({
            'ImagePath': str(dst),
            'StockCode': stock_code,
            'Description': class_name,
            'ClassID': idx
        })

    for img_path in val_imgs:
        src = Path(img_path)
        dst = dataset_dir / 'val' / str(stock_code) / src.name
        shutil.copy2(src, dst)
        val_data.append({
            'ImagePath': str(dst),
            'StockCode': stock_code,
            'Description': class_name,
            'ClassID': idx
        })

    for img_path in test_imgs:
        src = Path(img_path)
        dst = dataset_dir / 'test' / str(stock_code) / src.name
        shutil.copy2(src, dst)
        test_data.append({
            'ImagePath': str(dst),
            'StockCode': stock_code,
            'Description': class_name,
            'ClassID': idx
        })

# Save split datasets as CSV files
train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)
test_df = pd.DataFrame(test_data)

train_df.to_csv('data/cnn_train.csv', index=False)
val_df.to_csv('data/cnn_val.csv', index=False)
test_df.to_csv('data/cnn_test.csv', index=False)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Train set: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test set: {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
print(f"\nTotal: {len(train_df) + len(val_df) + len(test_df)} images")

# Save class mapping
mapping_info = {
    'class_mapping': class_mapping,
    'class_names': class_names,
    'num_classes': len(class_mapping)
}

with open('data/class_mapping.json', 'w') as f:
    json.dump(mapping_info, f, indent=2)

print(f"\nClass mapping saved to: data/class_mapping.json")
print(f"Train data saved to: data/cnn_train.csv")
print(f"Validation data saved to: data/cnn_val.csv")
print(f"Test data saved to: data/cnn_test.csv")

print("\n" + "="*70)
print("Dataset Structure:")
print("="*70)
print("data/cnn_dataset/")
print("├── train/")
print("│   ├── 20726/  (LUNCH BAG WOODLAND)")
print("│   ├── 21034/  (REX CASH+CARRY JUMBO SHOPPER)")
print("│   └── ...")
print("├── val/")
print("│   ├── 20726/")
print("│   └── ...")
print("└── test/")
print("    ├── 20726/")
print("    └── ...")
print("\n" + "="*70)
print("DATA PREPARATION COMPLETE")
print("="*70)
