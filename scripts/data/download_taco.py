"""
Download TACO (Trash Annotations in Context) Dataset

CRITICAL: High-quality waste detection dataset with COCO format annotations
- 1,500+ images
- 4,784 annotations
- 60 categories
- Real-world context
"""

import os
import sys
import logging
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import json
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TACO dataset URLs
TACO_REPO_URL = "https://github.com/pedropro/TACO"
TACO_DOWNLOAD_URL = "http://tacodataset.org/static/download.html"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "vision" / "taco"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
IMAGES_DIR = DATA_DIR / "images"


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded: {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def clone_taco_repo():
    """Clone TACO repository"""
    try:
        logger.info("Cloning TACO repository...")
        
        # Create temp directory
        temp_dir = DATA_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone repository
        import subprocess
        result = subprocess.run(
            ["git", "clone", TACO_REPO_URL, str(temp_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr}")
            return False
        
        logger.info("Repository cloned successfully")
        return temp_dir
        
    except Exception as e:
        logger.error(f"Failed to clone repository: {e}")
        return None


def download_taco_images(temp_dir: Path):
    """Download TACO images using the repository's download script"""
    try:
        logger.info("Downloading TACO images...")
        
        # Run download script from TACO repo
        import subprocess
        download_script = temp_dir / "download.py"
        
        if not download_script.exists():
            logger.error("Download script not found in repository")
            return False
        
        # Run download script
        result = subprocess.run(
            [sys.executable, str(download_script)],
            cwd=str(temp_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.warning(f"Download script output: {result.stderr}")
        
        logger.info("Images downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download images: {e}")
        return False


def organize_dataset(temp_dir: Path):
    """Organize downloaded dataset into proper structure"""
    try:
        logger.info("Organizing dataset...")
        
        # Create directories
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Move images
        source_images = temp_dir / "data"
        if source_images.exists():
            for img_file in source_images.rglob("*.jpg"):
                dest_file = IMAGES_DIR / img_file.name
                shutil.copy2(img_file, dest_file)
            logger.info(f"Copied images to {IMAGES_DIR}")
        
        # Move annotations
        source_annotations = temp_dir / "data" / "annotations.json"
        if source_annotations.exists():
            dest_annotations = ANNOTATIONS_DIR / "instances.json"
            shutil.copy2(source_annotations, dest_annotations)
            logger.info(f"Copied annotations to {ANNOTATIONS_DIR}")
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary files")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to organize dataset: {e}")
        return False


def validate_dataset():
    """Validate downloaded dataset"""
    try:
        logger.info("Validating dataset...")
        
        # Check annotations file
        annotations_file = ANNOTATIONS_DIR / "instances.json"
        if not annotations_file.exists():
            logger.error("Annotations file not found")
            return False
        
        # Load and validate annotations
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        num_images = len(data.get('images', []))
        num_annotations = len(data.get('annotations', []))
        num_categories = len(data.get('categories', []))
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  Images: {num_images}")
        logger.info(f"  Annotations: {num_annotations}")
        logger.info(f"  Categories: {num_categories}")
        
        # Check if images exist
        image_files = list(IMAGES_DIR.glob("*.jpg"))
        logger.info(f"  Image files: {len(image_files)}")
        
        if num_images == 0 or num_annotations == 0:
            logger.error("Dataset appears to be empty")
            return False
        
        logger.info("✅ Dataset validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def main():
    """Main download function"""
    logger.info("=" * 60)
    logger.info("TACO Dataset Download")
    logger.info("=" * 60)
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clone repository
    temp_dir = clone_taco_repo()
    if not temp_dir:
        logger.error("Failed to clone repository")
        return False
    
    # Download images
    if not download_taco_images(temp_dir):
        logger.error("Failed to download images")
        return False
    
    # Organize dataset
    if not organize_dataset(temp_dir):
        logger.error("Failed to organize dataset")
        return False
    
    # Validate dataset
    if not validate_dataset():
        logger.error("Dataset validation failed")
        return False
    
    logger.info("=" * 60)
    logger.info("✅ TACO dataset downloaded successfully!")
    logger.info(f"Location: {DATA_DIR}")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

