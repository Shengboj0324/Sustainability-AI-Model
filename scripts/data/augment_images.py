"""
Image Data Augmentation

CRITICAL: Expand training dataset with high-quality augmentations
- Horizontal flip (50% probability)
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation)
- Random crop and resize
- Gaussian noise (simulate low-quality cameras)
- Cutout/CutMix (improve robustness)
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CLEAN_DATA_DIR = PROJECT_ROOT / "data" / "clean" / "vision"
AUGMENTED_DATA_DIR = PROJECT_ROOT / "data" / "augmented" / "vision"

# Augmentation parameters
AUGMENTATIONS_PER_IMAGE = 3  # Generate 3 augmented versions per image
TARGET_SIZE = 224  # Standard size for vision models


def create_augmentation_pipeline():
    """Create augmentation pipeline using Albumentations"""
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        
        # Color transformations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        
        # Noise and blur
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        
        # Cutout for robustness
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        
        # Resize to target size
        A.Resize(TARGET_SIZE, TARGET_SIZE),
    ])


def augment_image(image_path: Path, output_dir: Path, num_augmentations: int = AUGMENTATIONS_PER_IMAGE):
    """Augment a single image"""
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load {image_path}")
            return 0
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create augmentation pipeline
        transform = create_augmentation_pipeline()
        
        # Generate augmented versions
        base_name = image_path.stem
        ext = image_path.suffix
        
        augmented_count = 0
        for i in range(num_augmentations):
            try:
                # Apply augmentation
                augmented = transform(image=image)
                augmented_image = augmented['image']
                
                # Save augmented image
                output_path = output_dir / f"{base_name}_aug_{i}{ext}"
                augmented_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), augmented_rgb)
                
                augmented_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to augment {image_path} (iteration {i}): {e}")
        
        return augmented_count
        
    except Exception as e:
        logger.error(f"Error augmenting {image_path}: {e}")
        return 0


def augment_dataset(dataset_name: str):
    """Augment a single dataset"""
    logger.info(f"Augmenting dataset: {dataset_name}")
    
    dataset_dir = CLEAN_DATA_DIR / dataset_name
    if not dataset_dir.exists():
        logger.warning(f"Dataset not found: {dataset_dir}")
        return
    
    # Create output directory
    output_dir = AUGMENTED_DATA_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(dataset_dir.rglob(f"*{ext}"))
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Copy original images
    logger.info("Copying original images...")
    for image_path in tqdm(image_paths, desc="Copying originals"):
        output_path = output_dir / image_path.name
        import shutil
        shutil.copy2(image_path, output_path)
    
    # Augment images
    total_augmented = 0
    for image_path in tqdm(image_paths, desc="Augmenting images"):
        count = augment_image(image_path, output_dir)
        total_augmented += count
    
    # Report statistics
    total_images = len(image_paths) + total_augmented
    logger.info(f"Augmentation results:")
    logger.info(f"  Original images: {len(image_paths)}")
    logger.info(f"  Augmented images: {total_augmented}")
    logger.info(f"  Total images: {total_images}")
    logger.info(f"  Expansion factor: {total_images / len(image_paths):.2f}x")
    
    logger.info(f"✅ Augmented dataset saved to {output_dir}")


def create_train_val_test_split(dataset_name: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split dataset into train/val/test sets"""
    logger.info(f"Creating train/val/test split for {dataset_name}")
    
    dataset_dir = AUGMENTED_DATA_DIR / dataset_name
    if not dataset_dir.exists():
        logger.warning(f"Dataset not found: {dataset_dir}")
        return
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(dataset_dir.glob(f"*{ext}"))
    
    # Shuffle
    random.shuffle(image_paths)
    
    # Calculate split indices
    n = len(image_paths)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split
    train_images = image_paths[:train_end]
    val_images = image_paths[train_end:val_end]
    test_images = image_paths[val_end:]
    
    # Create split directories
    for split_name, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
        split_dir = dataset_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        # Move images
        for image_path in split_images:
            dest_path = split_dir / image_path.name
            image_path.rename(dest_path)
    
    logger.info(f"Split results:")
    logger.info(f"  Train: {len(train_images)} ({len(train_images)/n*100:.1f}%)")
    logger.info(f"  Val: {len(val_images)} ({len(val_images)/n*100:.1f}%)")
    logger.info(f"  Test: {len(test_images)} ({len(test_images)/n*100:.1f}%)")


def main():
    """Main augmentation function"""
    logger.info("=" * 60)
    logger.info("Image Data Augmentation")
    logger.info("=" * 60)
    
    # Find all datasets
    datasets = [d.name for d in CLEAN_DATA_DIR.iterdir() if d.is_dir()]
    logger.info(f"Found {len(datasets)} datasets: {datasets}")
    
    # Augment each dataset
    for dataset_name in datasets:
        augment_dataset(dataset_name)
        create_train_val_test_split(dataset_name)
        logger.info("")
    
    logger.info("=" * 60)
    logger.info("✅ All datasets augmented!")
    logger.info(f"Augmented data location: {AUGMENTED_DATA_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

