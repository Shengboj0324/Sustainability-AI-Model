"""
Package Installation and Verification Script

CRITICAL: Install and verify all required packages
- Check Python version
- Install missing packages
- Verify imports
- Check GPU availability
- Generate installation report
"""

import subprocess
import sys
import importlib.util
import logging
from pathlib import Path
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Minimum Python version
MIN_PYTHON_VERSION = (3, 8)

# Critical packages (must have)
CRITICAL_PACKAGES = {
    'torch': 'torch',
    'torchvision': 'torchvision',
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn',
    'pydantic': 'pydantic',
    'numpy': 'numpy',
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
    'yaml': 'PyYAML',
    'requests': 'requests',
}

# Training packages (needed for model training)
TRAINING_PACKAGES = {
    'timm': 'timm',
    'transformers': 'transformers',
    'peft': 'peft',
    'datasets': 'datasets',
    'wandb': 'wandb',
    'albumentations': 'albumentations',
    'imagehash': 'imagehash',
    'pycocotools': 'pycocotools',
    'tqdm': 'tqdm',
    'pandas': 'pandas',
    'pyarrow': 'pyarrow',
}

# Database packages
DATABASE_PACKAGES = {
    'qdrant_client': 'qdrant-client',
    'neo4j': 'neo4j',
    'psycopg2': 'psycopg2-binary',
    'asyncpg': 'asyncpg',
    'redis': 'redis',
}

# Web scraping packages
SCRAPING_PACKAGES = {
    'bs4': 'beautifulsoup4',
    'lxml': 'lxml',
}


def check_python_version() -> bool:
    """Check if Python version meets minimum requirements"""
    current_version = sys.version_info[:2]
    
    if current_version >= MIN_PYTHON_VERSION:
        logger.info(f"✅ Python {current_version[0]}.{current_version[1]} (meets minimum {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]})")
        return True
    else:
        logger.error(f"❌ Python {current_version[0]}.{current_version[1]} (requires minimum {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]})")
        return False


def check_package(import_name: str) -> bool:
    """Check if a package is installed"""
    return importlib.util.find_spec(import_name) is not None


def install_package(package_name: str) -> bool:
    """Install a package using pip"""
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name, "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Failed to install {package_name}")
        return False


def check_and_install_packages(packages: Dict[str, str], category: str) -> Tuple[List[str], List[str]]:
    """Check and install packages in a category"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Checking {category}")
    logger.info(f"{'='*80}")
    
    available = []
    missing = []
    failed = []
    
    for import_name, package_name in packages.items():
        if check_package(import_name):
            available.append(package_name)
            logger.info(f"✅ {package_name}")
        else:
            missing.append(package_name)
            logger.warning(f"⚠️  {package_name} - MISSING")
    
    # Install missing packages
    if missing:
        logger.info(f"\nInstalling {len(missing)} missing packages...")
        for package_name in missing:
            if install_package(package_name):
                available.append(package_name)
                logger.info(f"✅ Installed {package_name}")
            else:
                failed.append(package_name)
                logger.error(f"❌ Failed to install {package_name}")
    
    logger.info(f"\nSummary: {len(available)}/{len(packages)} available, {len(failed)} failed")
    
    return available, failed


def check_gpu() -> Dict[str, any]:
    """Check GPU availability"""
    logger.info(f"\n{'='*80}")
    logger.info("Checking GPU Availability")
    logger.info(f"{'='*80}")
    
    gpu_info = {
        'available': False,
        'device_count': 0,
        'device_name': None,
        'cuda_version': None
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            gpu_info['cuda_version'] = torch.version.cuda
            
            logger.info(f"✅ GPU Available")
            logger.info(f"  Device Count: {gpu_info['device_count']}")
            logger.info(f"  Device Name: {gpu_info['device_name']}")
            logger.info(f"  CUDA Version: {gpu_info['cuda_version']}")
        else:
            logger.warning("⚠️  No GPU available - will use CPU")
    
    except Exception as e:
        logger.error(f"❌ Error checking GPU: {e}")
    
    return gpu_info


def verify_critical_imports() -> bool:
    """Verify critical imports work"""
    logger.info(f"\n{'='*80}")
    logger.info("Verifying Critical Imports")
    logger.info(f"{'='*80}")
    
    critical_imports = [
        ('torch', 'import torch'),
        ('fastapi', 'from fastapi import FastAPI'),
        ('pydantic', 'from pydantic import BaseModel'),
        ('numpy', 'import numpy as np'),
        ('PIL', 'from PIL import Image'),
    ]
    
    all_success = True
    
    for name, import_stmt in critical_imports:
        try:
            exec(import_stmt)
            logger.info(f"✅ {name}")
        except Exception as e:
            logger.error(f"❌ {name}: {e}")
            all_success = False
    
    return all_success


def main():
    """Main installation and verification"""
    logger.info("=" * 80)
    logger.info("RELEAF AI - PACKAGE INSTALLATION & VERIFICATION")
    logger.info("=" * 80)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check and install packages
    all_failed = []
    
    _, failed = check_and_install_packages(CRITICAL_PACKAGES, "Critical Packages")
    all_failed.extend(failed)
    
    _, failed = check_and_install_packages(TRAINING_PACKAGES, "Training Packages")
    all_failed.extend(failed)
    
    _, failed = check_and_install_packages(DATABASE_PACKAGES, "Database Packages")
    all_failed.extend(failed)
    
    _, failed = check_and_install_packages(SCRAPING_PACKAGES, "Web Scraping Packages")
    all_failed.extend(failed)
    
    # Check GPU
    gpu_info = check_gpu()
    
    # Verify imports
    imports_ok = verify_critical_imports()
    
    # Final report
    logger.info(f"\n{'='*80}")
    logger.info("FINAL REPORT")
    logger.info(f"{'='*80}")
    
    if all_failed:
        logger.error(f"❌ {len(all_failed)} packages failed to install:")
        for pkg in all_failed:
            logger.error(f"  - {pkg}")
        logger.info("\nTry installing manually:")
        logger.info(f"  pip install {' '.join(all_failed)}")
        sys.exit(1)
    elif not imports_ok:
        logger.error("❌ Some critical imports failed")
        sys.exit(1)
    else:
        logger.info("✅ ALL PACKAGES INSTALLED AND VERIFIED!")
        logger.info(f"✅ GPU Available: {gpu_info['available']}")
        logger.info("\n🎉 System ready for training and deployment!")


if __name__ == "__main__":
    main()

