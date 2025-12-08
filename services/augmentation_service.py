"""
Data Augmentation Service - Expand image dataset for CNN training
Applies various transformations to create multiple variations of each image
"""
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
import hashlib


class AugmentationService:
    """
    Service for augmenting product images to expand training dataset.
    Applies transformations like rotation, flipping, brightness, contrast, etc.
    """

    def __init__(self):
        self.transformations = [
            self._rotate_90,
            self._rotate_180,
            self._rotate_270,
            self._flip_horizontal,
            self._flip_vertical,
            self._brightness_increase,
            self._brightness_decrease,
            self._contrast_increase,
            self._contrast_decrease,
            self._blur,
            self._sharpen,
        ]

    def augment_image(self, image_path: Path, output_dir: Path, base_index: int, target_count: int) -> list:
        """
        Augment a single image to create multiple variations.

        Args:
            image_path: Path to original image
            output_dir: Directory to save augmented images
            base_index: Starting index for naming augmented images
            target_count: Target number of augmented images to create

        Returns:
            List of paths to augmented images
        """
        try:
            original_img = Image.open(image_path)
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')

            augmented_paths = []
            transformations_used = 0

            # Calculate how many transformations to apply
            num_transformations = min(target_count, len(self.transformations))

            for i, transform in enumerate(self.transformations[:num_transformations]):
                if transformations_used >= target_count:
                    break

                try:
                    # Apply transformation
                    augmented_img = transform(original_img)

                    # Generate filename
                    img_hash = hashlib.md5(augmented_img.tobytes()).hexdigest()[:8]
                    filename = f"{base_index + transformations_used:04d}_aug_{img_hash}.jpg"
                    filepath = output_dir / filename

                    # Save augmented image
                    augmented_img.save(filepath, 'JPEG', quality=85)
                    augmented_paths.append(str(filepath))
                    transformations_used += 1

                except Exception:
                    continue

            return augmented_paths

        except Exception:
            return []

    def _rotate_90(self, img: Image.Image) -> Image.Image:
        """Rotate image 90 degrees clockwise."""
        return img.rotate(-90, expand=True)

    def _rotate_180(self, img: Image.Image) -> Image.Image:
        """Rotate image 180 degrees."""
        return img.rotate(180, expand=True)

    def _rotate_270(self, img: Image.Image) -> Image.Image:
        """Rotate image 270 degrees clockwise."""
        return img.rotate(-270, expand=True)

    def _flip_horizontal(self, img: Image.Image) -> Image.Image:
        """Flip image horizontally."""
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def _flip_vertical(self, img: Image.Image) -> Image.Image:
        """Flip image vertically."""
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    def _brightness_increase(self, img: Image.Image) -> Image.Image:
        """Increase brightness by 30%."""
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.3)

    def _brightness_decrease(self, img: Image.Image) -> Image.Image:
        """Decrease brightness by 30%."""
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(0.7)

    def _contrast_increase(self, img: Image.Image) -> Image.Image:
        """Increase contrast by 30%."""
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.3)

    def _contrast_decrease(self, img: Image.Image) -> Image.Image:
        """Decrease contrast by 30%."""
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(0.7)

    def _blur(self, img: Image.Image) -> Image.Image:
        """Apply slight blur."""
        return img.filter(ImageFilter.BLUR)

    def _sharpen(self, img: Image.Image) -> Image.Image:
        """Apply sharpening filter."""
        return img.filter(ImageFilter.SHARPEN)

    def augment_product_directory(self, product_dir: Path, target_per_product: int = 150) -> int:
        """
        Augment all images in a product directory to reach target count.

        Args:
            product_dir: Directory containing original product images
            target_per_product: Target total number of images (original + augmented)

        Returns:
            Total number of images after augmentation
        """
        # Get original images
        original_images = list(product_dir.glob('*[!aug]*.jpg'))
        current_count = len(list(product_dir.glob('*.jpg')))

        if current_count >= target_per_product:
            return current_count

        # Calculate how many augmented images needed
        images_needed = target_per_product - current_count
        augmentations_per_image = images_needed // len(original_images) if original_images else 0

        print(f"    Original images: {len(original_images)}")
        print(f"    Need {images_needed} more images")
        print(f"    Creating {augmentations_per_image} augmentations per original image")

        total_augmented = 0
        next_index = current_count

        for img_path in original_images:
            if total_augmented >= images_needed:
                break

            num_to_create = min(augmentations_per_image, images_needed - total_augmented)
            augmented_paths = self.augment_image(img_path, product_dir, next_index, num_to_create)

            total_augmented += len(augmented_paths)
            next_index += len(augmented_paths)

        final_count = len(list(product_dir.glob('*.jpg')))
        print(f"    Final count: {final_count} images ({total_augmented} augmented)")

        return final_count
