"""
CNN Service - Product image classification
Uses trained CNN model to identify products from images
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import logging
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from io import BytesIO
import json
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class CNNService:
    """
    Service for classifying product images using trained CNN model.
    """

    def __init__(self, weights_path: str = 'models/cnn_weights.weights.h5',
                 class_mapping_path: str = 'data/class_mapping.json'):
        """
        Initialize CNN service with trained model.

        Args:
            weights_path: Path to model weights file
            class_mapping_path: Path to class mapping JSON file
        """
        self.weights_path = weights_path
        self.class_mapping_path = class_mapping_path
        self.model = None
        self.class_mapping = None
        self.class_names = None
        self.reverse_mapping = None
        self.img_height = 128
        self.img_width = 128

        # Load model and mappings
        self._load_model()
        self._load_class_mapping()

    def _build_model_architecture(self):
        """Build the CNN model architecture (matches training script)."""
        # Build model layer by layer to avoid keras/tf-keras mixing issues
        model = keras.Sequential()

        # First convolutional block
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))

        # Second convolutional block
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D((2, 2)))

        # Third convolutional block
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D((2, 2)))

        # Fourth convolutional block
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D((2, 2)))

        # Flatten and fully connected layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10, activation='softmax'))

        return model

    def _load_model(self):
        """Load the trained CNN model."""
        try:
            logger.info("Building CNN model architecture...")
            self.model = self._build_model_architecture()
            logger.debug(f"Model architecture: {self.model.count_params()} total parameters")

            logger.info(f"Loading model weights from {self.weights_path}...")
            if not os.path.exists(self.weights_path):
                logger.error(f"Model weights file not found: {self.weights_path}")
                raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

            self.model.load_weights(self.weights_path)
            logger.info("CNN model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading CNN model: {str(e)}", exc_info=True)
            raise

    def _load_class_mapping(self):
        """Load class ID to stock code and product name mappings."""
        try:
            logger.info(f"Loading class mappings from {self.class_mapping_path}...")

            if not os.path.exists(self.class_mapping_path):
                logger.error(f"Class mapping file not found: {self.class_mapping_path}")
                raise FileNotFoundError(f"Class mapping not found: {self.class_mapping_path}")

            with open(self.class_mapping_path, 'r') as f:
                mapping_data = json.load(f)

            self.class_mapping = mapping_data['class_mapping']  # stock_code -> class_id
            self.class_names = mapping_data['class_names']  # class_id -> product_name

            # Create reverse mapping: class_id -> stock_code
            self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}

            logger.info(f"Loaded {len(self.class_names)} product classes")
            logger.debug(f"Classes: {list(self.class_names.values())}")

        except Exception as e:
            logger.error(f"Error loading class mapping: {str(e)}", exc_info=True)
            raise

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for CNN input.

        Steps:
        1. Resize to 128x128
        2. Convert to RGB (in case of grayscale or RGBA)
        3. Convert to numpy array
        4. Normalize pixel values to [0, 1]
        5. Add batch dimension

        Args:
            image: PIL Image object

        Returns:
            Preprocessed image array ready for CNN
        """
        try:
            logger.debug(f"Preprocessing image: size={image.size}, mode={image.mode}")

            # Convert to RGB if needed
            if image.mode != 'RGB':
                logger.debug(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')

            # Resize to model input size
            original_size = image.size
            image = image.resize((self.img_width, self.img_height))
            logger.debug(f"Resized image from {original_size} to ({self.img_width}, {self.img_height})")

            # Convert to numpy array
            img_array = np.array(image)

            # Normalize pixel values to [0, 1]
            img_array = img_array.astype('float32') / 255.0

            # Add batch dimension (model expects batch of images)
            img_array = np.expand_dims(img_array, axis=0)
            logger.debug(f"Preprocessed image shape: {img_array.shape}")

            return img_array

        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}", exc_info=True)
            raise

    def predict(self, image_path: str, return_top_k: int = 3) -> dict:
        """
        Predict product class from image file.

        Args:
            image_path: Path to image file
            return_top_k: Number of top predictions to return (default: 3)

        Returns:
            Dictionary containing:
            - predicted_class_id: Most likely class ID
            - predicted_stock_code: Stock code of predicted product
            - predicted_product: Product name
            - confidence: Confidence score (0-1)
            - top_predictions: List of top K predictions with scores
        """
        try:
            logger.info(f"Predicting product from image: {image_path}")

            # Load image
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path)
            logger.debug(f"Loaded image: {image.size}, {image.mode}")

            # Preprocess
            preprocessed = self.preprocess_image(image)

            # Get predictions
            logger.debug("Running CNN inference...")
            predictions = self.model.predict(preprocessed, verbose=0)

            # Get probabilities for all classes
            probabilities = predictions[0]

            # Get top K predictions
            top_k_indices = np.argsort(probabilities)[-return_top_k:][::-1]
            top_predictions = []

            for idx in top_k_indices:
                stock_code = self.reverse_mapping[idx]
                product_name = self.class_names[str(idx)]
                confidence = float(probabilities[idx])

                top_predictions.append({
                    'class_id': int(idx),
                    'stock_code': stock_code,
                    'product_name': product_name,
                    'confidence': confidence
                })
                logger.debug(f"Prediction {len(top_predictions)}: {product_name} ({confidence:.2%})")

            # Best prediction
            best_prediction = top_predictions[0]
            logger.info(
                f"Top prediction: {best_prediction['product_name']} "
                f"({best_prediction['confidence']:.2%} confidence)"
            )

            return {
                'predicted_class_id': best_prediction['class_id'],
                'predicted_stock_code': best_prediction['stock_code'],
                'predicted_product': best_prediction['product_name'],
                'confidence': best_prediction['confidence'],
                'top_predictions': top_predictions
            }

        except Exception as e:
            logger.error(f"Prediction error for image '{image_path}': {str(e)}", exc_info=True)
            return {
                'predicted_class_id': -1,
                'predicted_stock_code': '',
                'predicted_product': '',
                'confidence': 0.0,
                'top_predictions': [],
                'error': str(e)
            }

    def predict_from_bytes(self, image_bytes: bytes, return_top_k: int = 3) -> dict:
        """
        Predict product class from image bytes (for uploaded files).

        Args:
            image_bytes: Image data as bytes
            return_top_k: Number of top predictions to return (default: 3)

        Returns:
            Dictionary containing prediction results (same as predict method)
        """
        try:
            logger.info(f"Predicting product from uploaded image ({len(image_bytes)} bytes)")

            # Convert bytes to PIL Image
            logger.debug("Converting image bytes to PIL Image...")
            image = Image.open(BytesIO(image_bytes))
            logger.debug(f"Loaded image: {image.size}, {image.mode}")

            # Preprocess
            preprocessed = self.preprocess_image(image)

            # Get predictions
            logger.debug("Running CNN inference...")
            predictions = self.model.predict(preprocessed, verbose=0)

            # Get probabilities for all classes
            probabilities = predictions[0]

            # Get top K predictions
            top_k_indices = np.argsort(probabilities)[-return_top_k:][::-1]
            top_predictions = []

            for idx in top_k_indices:
                stock_code = self.reverse_mapping[idx]
                product_name = self.class_names[str(idx)]
                confidence = float(probabilities[idx])

                top_predictions.append({
                    'class_id': int(idx),
                    'stock_code': stock_code,
                    'product_name': product_name,
                    'confidence': confidence
                })
                logger.debug(f"Prediction {len(top_predictions)}: {product_name} ({confidence:.2%})")

            # Best prediction
            best_prediction = top_predictions[0]
            logger.info(
                f"Top prediction: {best_prediction['product_name']} "
                f"({best_prediction['confidence']:.2%} confidence)"
            )

            return {
                'predicted_class_id': best_prediction['class_id'],
                'predicted_stock_code': best_prediction['stock_code'],
                'predicted_product': best_prediction['product_name'],
                'confidence': best_prediction['confidence'],
                'top_predictions': top_predictions
            }

        except Exception as e:
            logger.error(f"Prediction error for uploaded image: {str(e)}", exc_info=True)
            return {
                'predicted_class_id': -1,
                'predicted_stock_code': '',
                'predicted_product': '',
                'confidence': 0.0,
                'top_predictions': [],
                'error': str(e)
            }

    def get_class_info(self, class_id: int) -> dict:
        """
        Get information about a specific class.

        Args:
            class_id: Class ID (0-9)

        Returns:
            Dictionary with stock code and product name
        """
        if 0 <= class_id < len(self.class_names):
            stock_code = self.reverse_mapping[class_id]
            product_name = self.class_names[str(class_id)]

            return {
                'class_id': class_id,
                'stock_code': stock_code,
                'product_name': product_name
            }
        else:
            return {
                'class_id': class_id,
                'stock_code': '',
                'product_name': 'Unknown',
                'error': 'Invalid class ID'
            }

    def get_all_classes(self) -> list:
        """
        Get information about all product classes.

        Returns:
            List of dictionaries with class info
        """
        all_classes = []
        for i in range(len(self.class_names)):
            stock_code = self.reverse_mapping[i]
            product_name = self.class_names[str(i)]

            all_classes.append({
                'class_id': i,
                'stock_code': stock_code,
                'product_name': product_name
            })

        return all_classes
