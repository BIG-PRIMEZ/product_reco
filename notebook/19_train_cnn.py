"""
Train CNN model from scratch for product classification
Simple architecture - no pre-trained models
"""
import sys
sys.path.insert(0, '.')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

print("="*70)
print("CNN MODEL TRAINING")
print("="*70)

# Load class mapping
with open('data/class_mapping.json', 'r') as f:
    mapping = json.load(f)

num_classes = mapping['num_classes']
print(f"\nNumber of classes: {num_classes}")

# Image parameters
img_height = 128
img_width = 128
batch_size = 32

print(f"Image size: {img_height}x{img_width}")
print(f"Batch size: {batch_size}")

# Load datasets
train_dir = Path('data/cnn_dataset/train')
val_dir = Path('data/cnn_dataset/val')

print(f"\nLoading training data from: {train_dir}")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

print(f"Loading validation data from: {val_dir}")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

# Normalize pixel values to [0, 1]
normalization_layer = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("\n" + "="*70)
print("BUILDING CNN ARCHITECTURE")
print("="*70)

# Simple CNN architecture from scratch
model = keras.Sequential([
    # Input layer
    layers.Input(shape=(img_height, img_width, 3)),

    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Fourth convolutional block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and fully connected layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

print("\nModel Architecture:")
model.summary()

print("\n" + "="*70)
print("COMPILING MODEL")
print("="*70)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nOptimizer: Adam (learning_rate=0.001)")
print("Loss: Sparse Categorical Crossentropy")
print("Metrics: Accuracy")

print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

# Training parameters
epochs = 30

print(f"\nEpochs: {epochs}")
print("\nCallbacks:")
print("  - Early Stopping (patience=5, monitor=val_loss)")
print("  - Model Checkpoint (save best model)")

# Callbacks
checkpoint_path = 'models/cnn_product_classifier.keras'
Path('models').mkdir(exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train model
print("\n" + "-"*70)
print("Starting training...")
print("-"*70 + "\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

# Save final model
final_model_path = 'models/cnn_product_classifier_final.keras'
model.save(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")
print(f"Best model saved to: {checkpoint_path}")

# Plot training history
print("\n" + "="*70)
print("TRAINING METRICS")
print("="*70)

# Get final metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

best_val_acc = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"\nBest Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")

# Save training history
history_path = 'models/training_history.json'
history_dict = {
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'epochs_trained': len(history.history['accuracy']),
    'best_val_accuracy': float(best_val_acc),
    'best_epoch': int(best_epoch)
}

with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)

print(f"\nTraining history saved to: {history_path}")

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Training')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'], label='Training')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Model Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plot_path = 'models/training_plots.png'
plt.savefig(plot_path, dpi=100)
print(f"Training plots saved to: {plot_path}")

print("\n" + "="*70)
print("CNN TRAINING COMPLETE")
print("="*70)
