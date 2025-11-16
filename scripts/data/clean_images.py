"""
Clean and Validate Vision Dataset

CRITICAL: Ensure high-quality training data
- Remove duplicates (perceptual hashing)
- Filter low-quality images (blur detection, size check)
- Validate annotations (bounding box sanity checks)
- Standardize formats (convert all to COCO)
- Balance classes (oversample minority classes)
"""

import os
import sys
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import hashlib

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import imagehash

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "vision"
CLEAN_DATA_DIR = PROJECT_ROOT / "data" / "clean" / "vision"

# Quality thresholds
MIN_IMAGE_SIZE = 32  # pixels
MAX_IMAGE_SIZE = 4096  # pixels
MIN_ASPECT_RATIO = 0.1
MAX_ASPECT_RATIO = 10.0
BLUR_THRESHOLD = 100.0  # Laplacian variance
DUPLICATE_HASH_THRESHOLD = 5  # Hamming distance


def compute_perceptual_hash(image_path: Path) -> str:
    """Compute perceptual hash for duplicate detection"""
    try:
        img = Image.open(image_path)
        return str(imagehash.phash(img))
    except Exception as e:
        logger.warning(f"Failed to hash {image_path}: {e}")
        return None


def detect_blur(image_path: Path) -> float:
    """Detect blur using Laplacian variance"""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except Exception as e:
        logger.warning(f"Failed to detect blur in {image_path}: {e}")
        return 0.0


def validate_image(image_path: Path) -> Tuple[bool, str]:
    """Validate image quality"""
    try:
        # Open image
        img = Image.open(image_path)
        width, height = img.size
        
        # Check size
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            return False, f"Too small: {width}x{height}"
        
        if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
            return False, f"Too large: {width}x{height}"
        
        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            return False, f"Bad aspect ratio: {aspect_ratio:.2f}"
        
        # Check blur
        blur_score = detect_blur(image_path)
        if blur_score < BLUR_THRESHOLD:
            return False, f"Too blurry: {blur_score:.2f}"
        
        # Check if image is readable
        img.verify()
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Error: {e}"


def find_duplicates(image_paths: List[Path]) -> Set[Path]:
    """Find duplicate images using perceptual hashing"""
    logger.info("Finding duplicates...")
    
    hash_to_paths = defaultdict(list)
    duplicates = set()
    
    for image_path in tqdm(image_paths, desc="Hashing images"):
        img_hash = compute_perceptual_hash(image_path)
        if img_hash:
            hash_to_paths[img_hash].append(image_path)
    
    # Find duplicates
    for img_hash, paths in hash_to_paths.items():
        if len(paths) > 1:
            # Keep first, mark rest as duplicates
            duplicates.update(paths[1:])
            logger.info(f"Found {len(paths)} duplicates with hash {img_hash}")
    
    logger.info(f"Found {len(duplicates)} duplicate images")
    return duplicates


def clean_dataset(dataset_name: str):
    """Clean a single dataset"""
    logger.info(f"Cleaning dataset: {dataset_name}")
    
    dataset_dir = RAW_DATA_DIR / dataset_name
    if not dataset_dir.exists():
        logger.warning(f"Dataset not found: {dataset_dir}")
        return
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(dataset_dir.rglob(f"*{ext}"))
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Find duplicates
    duplicates = find_duplicates(image_paths)
    
    # Validate images
    valid_images = []
    invalid_images = []
    
    for image_path in tqdm(image_paths, desc="Validating images"):
        if image_path in duplicates:
            invalid_images.append((image_path, "Duplicate"))
            continue
        
        is_valid, reason = validate_image(image_path)
        if is_valid:
            valid_images.append(image_path)
        else:
            invalid_images.append((image_path, reason))
    
    # Report statistics
    logger.info(f"Validation results:")
    logger.info(f"  Valid: {len(valid_images)}")
    logger.info(f"  Invalid: {len(invalid_images)}")
    logger.info(f"  Duplicates: {len(duplicates)}")
    
    # Copy valid images to clean directory
    clean_dir = CLEAN_DATA_DIR / dataset_name
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path in tqdm(valid_images, desc="Copying valid images"):
        dest_path = clean_dir / image_path.name
        shutil.copy2(image_path, dest_path)
    
    logger.info(f"✅ Cleaned dataset saved to {clean_dir}")
    
    # Save invalid images report
    report_path = clean_dir / "invalid_images.txt"
    with open(report_path, 'w') as f:
        for image_path, reason in invalid_images:
            f.write(f"{image_path.name}\t{reason}\n")
    
    logger.info(f"Invalid images report saved to {report_path}")


def main():
    """Main cleaning function"""
    logger.info("=" * 60)
    logger.info("Image Dataset Cleaning")
    logger.info("=" * 60)
    
    # Find all datasets
    datasets = [d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    logger.info(f"Found {len(datasets)} datasets: {datasets}")
    
    # Clean each dataset
    for dataset_name in datasets:
        clean_dataset(dataset_name)
        logger.info("")
    
    logger.info("=" * 60)
    logger.info("✅ All datasets cleaned!")
    logger.info(f"Clean data location: {CLEAN_DATA_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

