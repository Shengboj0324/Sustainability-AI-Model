"""
Download Kaggle Waste Classification Datasets

CRITICAL: Multiple high-quality datasets for comprehensive training
- Recyclable and Household Waste Classification (15,000+ images)
- Waste Classification Dataset (25,000+ images)
- Garbage Classification V2 (15,000+ images)
- TrashNet (2,527 images)
"""

import os
import sys
import logging
import json
import shutil
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "vision" / "kaggle"

# Kaggle datasets to download
DATASETS = [
    {
        "name": "recyclable-household-waste",
        "kaggle_id": "alistairking/recyclable-and-household-waste-classification",
        "priority": "CRITICAL",
        "expected_images": 15000
    },
    {
        "name": "waste-classification",
        "kaggle_id": "adithyachalla/waste-classification",
        "priority": "CRITICAL",
        "expected_images": 25000
    },
    {
        "name": "garbage-classification-v2",
        "kaggle_id": "sumn2u/garbage-classification-v2",
        "priority": "HIGH",
        "expected_images": 15000
    },
    {
        "name": "trashnet",
        "kaggle_id": "asdasdasasdas/garbage-classification",
        "priority": "MEDIUM",
        "expected_images": 2500
    }
]


def check_kaggle_api():
    """Check if Kaggle API is configured"""
    try:
        import kaggle
        logger.info("✅ Kaggle API configured")
        return True
    except OSError as e:
        logger.error("❌ Kaggle API not configured")
        logger.error("Please follow these steps:")
        logger.error("1. Go to https://www.kaggle.com/account")
        logger.error("2. Click 'Create New API Token'")
        logger.error("3. Move kaggle.json to ~/.kaggle/kaggle.json")
        logger.error("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    except Exception as e:
        logger.error(f"Kaggle API error: {e}")
        return False


def download_dataset(dataset_info: dict):
    """Download a single Kaggle dataset"""
    try:
        import kaggle
        
        dataset_name = dataset_info["name"]
        kaggle_id = dataset_info["kaggle_id"]
        
        logger.info(f"Downloading {dataset_name}...")
        logger.info(f"  Kaggle ID: {kaggle_id}")
        logger.info(f"  Priority: {dataset_info['priority']}")
        
        # Create dataset directory
        dataset_dir = DATA_DIR / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            kaggle_id,
            path=str(dataset_dir),
            unzip=True,
            quiet=False
        )
        
        logger.info(f"✅ Downloaded {dataset_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {dataset_name}: {e}")
        return False


def count_images(dataset_dir: Path) -> int:
    """Count images in dataset directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    count = 0
    
    for ext in image_extensions:
        count += len(list(dataset_dir.rglob(f"*{ext}")))
    
    return count


def validate_dataset(dataset_info: dict):
    """Validate downloaded dataset"""
    try:
        dataset_name = dataset_info["name"]
        dataset_dir = DATA_DIR / dataset_name
        
        if not dataset_dir.exists():
            logger.error(f"❌ Dataset directory not found: {dataset_dir}")
            return False
        
        # Count images
        num_images = count_images(dataset_dir)
        expected_images = dataset_info["expected_images"]
        
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"  Images found: {num_images}")
        logger.info(f"  Expected: ~{expected_images}")
        
        # Check if we have at least 50% of expected images
        if num_images < expected_images * 0.5:
            logger.warning(f"⚠️  Low image count for {dataset_name}")
            return False
        
        logger.info(f"✅ Validation passed for {dataset_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed for {dataset_name}: {e}")
        return False


def create_dataset_manifest():
    """Create manifest file with dataset information"""
    try:
        manifest = {
            "datasets": [],
            "total_images": 0,
            "download_date": str(Path.ctime(DATA_DIR))
        }
        
        for dataset_info in DATASETS:
            dataset_name = dataset_info["name"]
            dataset_dir = DATA_DIR / dataset_name
            
            if dataset_dir.exists():
                num_images = count_images(dataset_dir)
                manifest["datasets"].append({
                    "name": dataset_name,
                    "kaggle_id": dataset_info["kaggle_id"],
                    "priority": dataset_info["priority"],
                    "num_images": num_images,
                    "path": str(dataset_dir.relative_to(PROJECT_ROOT))
                })
                manifest["total_images"] += num_images
        
        # Save manifest
        manifest_file = DATA_DIR / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✅ Created manifest: {manifest_file}")
        logger.info(f"Total images: {manifest['total_images']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create manifest: {e}")
        return False


def main():
    """Main download function"""
    logger.info("=" * 60)
    logger.info("Kaggle Waste Classification Datasets Download")
    logger.info("=" * 60)
    
    # Check Kaggle API
    if not check_kaggle_api():
        return False
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    success_count = 0
    for dataset_info in DATASETS:
        if download_dataset(dataset_info):
            if validate_dataset(dataset_info):
                success_count += 1
        logger.info("")
    
    # Create manifest
    create_dataset_manifest()
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Downloaded {success_count}/{len(DATASETS)} datasets successfully")
    logger.info(f"Location: {DATA_DIR}")
    logger.info("=" * 60)
    
    return success_count > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

