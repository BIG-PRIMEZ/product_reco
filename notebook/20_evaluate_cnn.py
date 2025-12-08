"""
Evaluate trained CNN model on test set
Final performance metrics on unseen data
"""
import sys
sys.path.insert(0, '.')

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

print("="*70)
print("CNN MODEL EVALUATION ON TEST SET")
print("="*70)

# Load class mapping
with open('data/class_mapping.json', 'r') as f:
    mapping = json.load(f)

num_classes = mapping['num_classes']
class_names_dict = mapping['class_names']
class_names = [class_names_dict[str(i)] for i in range(num_classes)]

print(f"\nNumber of classes: {num_classes}")
print(f"Model: models/cnn_product_classifier.keras")

# Load test dataset
img_height = 128
img_width = 128
batch_size = 32

test_dir = Path('data/cnn_dataset/test')
print(f"\nLoading test data from: {test_dir}")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int',
    shuffle=False  # Don't shuffle for evaluation
)

# Normalize
normalization_layer = keras.layers.Rescaling(1./255)
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Load trained model
print("\nLoading trained model...")
model = keras.models.load_model('models/cnn_product_classifier.keras')

print("\n" + "="*70)
print("EVALUATING MODEL")
print("="*70)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

print("\n" + "="*70)
print("TEST SET RESULTS")
print("="*70)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Get predictions for detailed metrics
print("\nGenerating predictions for detailed analysis...")
y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# Classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)
print(report)

# Confusion matrix
print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)
cm = confusion_matrix(y_true, y_pred)
print("\nRows = True labels, Columns = Predicted labels")
print(f"Classes: {list(range(num_classes))}")
print(cm)

# Per-class accuracy
print("\n" + "="*70)
print("PER-CLASS ACCURACY")
print("="*70)
for i in range(num_classes):
    class_correct = cm[i, i]
    class_total = cm[i, :].sum()
    class_acc = class_correct / class_total if class_total > 0 else 0
    stock_code = list(mapping['class_mapping'].keys())[i]
    print(f"Class {i} ({stock_code}): {class_acc:.4f} ({class_acc*100:.2f}%) - {class_names[i]}")
    print(f"  Correct: {class_correct}/{class_total}")

# Save results
results = {
    'test_loss': float(test_loss),
    'test_accuracy': float(test_accuracy),
    'num_test_samples': len(y_true),
    'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
    'confusion_matrix': cm.tolist(),
    'class_names': class_names
}

results_path = 'models/test_evaluation.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n" + "="*70)
print(f"Evaluation results saved to: {results_path}")
print("="*70)

# Summary
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)
print(f"Test Samples: {len(y_true)}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"\nModel Performance:")
if test_accuracy >= 0.7:
    print("  ✓ Good - Model performs well on unseen data")
elif test_accuracy >= 0.5:
    print("  ~ Acceptable - Model has reasonable performance")
else:
    print("  ✗ Needs Improvement - Consider more training data or tuning")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
