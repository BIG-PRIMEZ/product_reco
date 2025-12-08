"""
OCR Service - Extract text from images
Uses EasyOCR for text recognition (better for handwriting)
"""

import logging
import easyocr
from PIL import Image
from io import BytesIO
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class OCRService:
    """
    Service for extracting text from images using OCR.
    Uses EasyOCR engine which handles handwriting well.
    """

    def __init__(self):
        """Initialize EasyOCR service."""
        try:
            logger.info("Initializing EasyOCR reader...")
            self.reader = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR reader initialized successfully (language: en, gpu: False)")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {str(e)}", exc_info=True)
            raise

    def extract_text(self, image_path: str) -> tuple[str, float]:
        """
        Extract text from an image.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            logger.info(f"Extracting text from image: {image_path}")

            # Read image with EasyOCR
            logger.debug("Running OCR detection...")
            results = self.reader.readtext(image_path)
            logger.debug(f"OCR detected {len(results)} text region(s)")

            # Combine all detected text
            text_parts = []
            confidences = []

            for i, detection in enumerate(results):
                bbox, text, conf = detection
                text_parts.append(text)
                confidences.append(conf)
                logger.debug(f"Region {i+1}: '{text}' (confidence: {conf:.2%})")

            # Combine text with spaces
            extracted_text = ' '.join(text_parts).strip()

            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            if extracted_text:
                logger.info(f"Successfully extracted text: '{extracted_text}' (avg confidence: {avg_confidence:.2%})")
            else:
                logger.warning(f"No text extracted from image: {image_path}")

            return extracted_text, avg_confidence

        except Exception as e:
            logger.error(f"OCR Error for image '{image_path}': {str(e)}", exc_info=True)
            return "", 0.0

    def extract_text_from_bytes(self, image_bytes: bytes) -> tuple[str, float]:
        """
        Extract text from image bytes (for uploaded files).

        Args:
            image_bytes: Image data as bytes

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            logger.info(f"Extracting text from uploaded image ({len(image_bytes)} bytes)")

            # Convert bytes to numpy array
            logger.debug("Converting image bytes to numpy array...")
            image = Image.open(BytesIO(image_bytes))
            logger.debug(f"Image size: {image.size}, mode: {image.mode}")
            image_np = np.array(image)

            # Read with EasyOCR
            logger.debug("Running OCR detection on uploaded image...")
            results = self.reader.readtext(image_np)
            logger.debug(f"OCR detected {len(results)} text region(s)")

            # Combine all detected text
            text_parts = []
            confidences = []

            for i, detection in enumerate(results):
                bbox, text, conf = detection
                text_parts.append(text)
                confidences.append(conf)
                logger.debug(f"Region {i+1}: '{text}' (confidence: {conf:.2%})")

            # Combine text with spaces
            extracted_text = ' '.join(text_parts).strip()

            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            if extracted_text:
                logger.info(f"Successfully extracted text: '{extracted_text}' (avg confidence: {avg_confidence:.2%})")
            else:
                logger.warning("No text extracted from uploaded image")

            return extracted_text, avg_confidence

        except Exception as e:
            logger.error(f"OCR Error for uploaded image: {str(e)}", exc_info=True)
            return "", 0.0
