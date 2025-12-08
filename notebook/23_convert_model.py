"""
Convert Keras model to weights-only format for better compatibility
"""
import tensorflow as tf
from tensorflow import keras

print("="*70)
print("CONVERTING MODEL TO WEIGHTS FORMAT")
print("="*70)

# Load model with tensorflow.keras (not tf-keras)
print("\nLoading model with tensorflow.keras...")
model = keras.models.load_model('models/cnn_product_classifier.keras')

print("Model loaded successfully")
print(f"Model summary:")
model.summary()

# Save just the weights in H5 format
print("\nSaving model weights...")
model.save_weights('models/cnn_weights.weights.h5')
print("Weights saved to: models/cnn_weights.weights.h5")

# Also save architecture as JSON
print("\nSaving model architecture...")
import json
architecture = model.to_json()
with open('models/cnn_architecture.json', 'w') as f:
    f.write(architecture)
print("Architecture saved to: models/cnn_architecture.json")

print("\n" + "="*70)
print("CONVERSION COMPLETE")
print("="*70)
print("\nFiles created:")
print("  - models/cnn_weights.weights.h5 (model weights)")
print("  - models/cnn_architecture.json (model architecture)")
