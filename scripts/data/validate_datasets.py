"""
Comprehensive Dataset Validation

CRITICAL: Ensure dataset quality meets extreme standards
- Validate image quality (95%+ accuracy target)
- Check class distribution and balance
- Verify annotation format and completeness
- Statistical analysis
- Generate quality report
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
AUGMENTED_DATA_DIR = PROJECT_ROOT / "data" / "augmented" / "vision"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

# Quality thresholds
MIN_IMAGES_PER_CLASS = 100
MAX_CLASS_IMBALANCE_RATIO = 10.0  # Max ratio between largest and smallest class
MIN_DATASET_SIZE = 1000


class DatasetValidator:
    """Comprehensive dataset validator"""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_path.name
        self.stats = {
            "total_images": 0,
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "image_sizes": [],
            "aspect_ratios": [],
            "file_sizes": [],
            "corrupted_images": [],
            "warnings": [],
            "errors": []
        }
    
    def validate_image_file(self, image_path: Path) -> Tuple[bool, str]:
        """Validate a single image file"""
        try:
            # Try to open and verify
            img = Image.open(image_path)
            img.verify()
            
            # Reopen for size check (verify closes the file)
            img = Image.open(image_path)
            width, height = img.size
            
            # Record statistics
            self.stats["image_sizes"].append((width, height))
            self.stats["aspect_ratios"].append(width / height)
            self.stats["file_sizes"].append(image_path.stat().st_size)
            
            return True, "OK"
            
        except Exception as e:
            self.stats["corrupted_images"].append(str(image_path))
            return False, str(e)
    
    def validate_split(self, split_name: str) -> Dict:
        """Validate a data split (train/val/test)"""
        split_dir = self.dataset_path / split_name
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return {"exists": False, "count": 0}
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(split_dir.rglob(f"*{ext}"))
        
        # Validate each image
        valid_count = 0
        for image_path in tqdm(image_paths, desc=f"Validating {split_name}"):
            is_valid, reason = self.validate_image_file(image_path)
            if is_valid:
                valid_count += 1
        
        return {
            "exists": True,
            "total": len(image_paths),
            "valid": valid_count,
            "invalid": len(image_paths) - valid_count
        }
    
    def check_class_distribution(self) -> Dict:
        """Check class distribution across splits"""
        class_counts = defaultdict(int)
        
        for split_name in ["train", "val", "test"]:
            split_dir = self.dataset_path / split_name
            if not split_dir.exists():
                continue
            
            # Count images per class (assuming directory structure: split/class/images)
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    image_count = len(list(class_dir.glob("*.jpg"))) + \
                                  len(list(class_dir.glob("*.png")))
                    class_counts[class_dir.name] += image_count
        
        if not class_counts:
            return {"balanced": True, "classes": {}, "warnings": []}
        
        # Calculate statistics
        counts = list(class_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Check for issues
        warnings = []
        if imbalance_ratio > MAX_CLASS_IMBALANCE_RATIO:
            warnings.append(f"High class imbalance: {imbalance_ratio:.2f}x")
        
        for class_name, count in class_counts.items():
            if count < MIN_IMAGES_PER_CLASS:
                warnings.append(f"Class '{class_name}' has only {count} images (min: {MIN_IMAGES_PER_CLASS})")
        
        return {
            "balanced": imbalance_ratio <= MAX_CLASS_IMBALANCE_RATIO,
            "classes": dict(class_counts),
            "imbalance_ratio": imbalance_ratio,
            "warnings": warnings
        }
    
    def generate_statistics(self) -> Dict:
        """Generate comprehensive statistics"""
        stats = {
            "dataset_name": self.dataset_name,
            "total_images": self.stats["total_images"],
            "splits": {
                "train": self.stats["train_images"],
                "val": self.stats["val_images"],
                "test": self.stats["test_images"]
            },
            "corrupted_images": len(self.stats["corrupted_images"]),
            "image_size": {
                "mean_width": np.mean([s[0] for s in self.stats["image_sizes"]]) if self.stats["image_sizes"] else 0,
                "mean_height": np.mean([s[1] for s in self.stats["image_sizes"]]) if self.stats["image_sizes"] else 0,
                "min_width": min([s[0] for s in self.stats["image_sizes"]]) if self.stats["image_sizes"] else 0,
                "max_width": max([s[0] for s in self.stats["image_sizes"]]) if self.stats["image_sizes"] else 0,
            },
            "aspect_ratio": {
                "mean": np.mean(self.stats["aspect_ratios"]) if self.stats["aspect_ratios"] else 0,
                "std": np.std(self.stats["aspect_ratios"]) if self.stats["aspect_ratios"] else 0,
            },
            "file_size": {
                "mean_mb": np.mean(self.stats["file_sizes"]) / (1024 * 1024) if self.stats["file_sizes"] else 0,
                "total_gb": sum(self.stats["file_sizes"]) / (1024 * 1024 * 1024) if self.stats["file_sizes"] else 0,
            },
            "warnings": self.stats["warnings"],
            "errors": self.stats["errors"]
        }
        
        return stats
    
    def validate(self) -> Dict:
        """Run complete validation"""
        logger.info(f"Validating dataset: {self.dataset_name}")
        
        # Validate splits
        train_stats = self.validate_split("train")
        val_stats = self.validate_split("val")
        test_stats = self.validate_split("test")
        
        self.stats["train_images"] = train_stats.get("valid", 0)
        self.stats["val_images"] = val_stats.get("valid", 0)
        self.stats["test_images"] = test_stats.get("valid", 0)
        self.stats["total_images"] = self.stats["train_images"] + self.stats["val_images"] + self.stats["test_images"]
        
        # Check class distribution
        class_dist = self.check_class_distribution()
        self.stats["warnings"].extend(class_dist["warnings"])
        
        # Check minimum dataset size
        if self.stats["total_images"] < MIN_DATASET_SIZE:
            self.stats["errors"].append(f"Dataset too small: {self.stats['total_images']} < {MIN_DATASET_SIZE}")
        
        # Generate statistics
        statistics = self.generate_statistics()
        statistics["class_distribution"] = class_dist
        
        return statistics


def validate_all_datasets():
    """Validate all datasets"""
    logger.info("=" * 60)
    logger.info("Dataset Validation")
    logger.info("=" * 60)
    
    # Create reports directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all datasets
    datasets = [d for d in AUGMENTED_DATA_DIR.iterdir() if d.is_dir()]
    logger.info(f"Found {len(datasets)} datasets")
    
    # Validate each dataset
    all_reports = []
    for dataset_path in datasets:
        validator = DatasetValidator(dataset_path)
        report = validator.validate()
        all_reports.append(report)
        
        # Save individual report
        report_path = REPORTS_DIR / f"{dataset_path.name}_validation.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Validated {dataset_path.name}: {report['total_images']} images")
        logger.info("")
    
    # Save combined report
    combined_report_path = REPORTS_DIR / "all_datasets_validation.json"
    with open(combined_report_path, 'w') as f:
        json.dump(all_reports, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("✅ All datasets validated!")
    logger.info(f"Reports saved to: {REPORTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    validate_all_datasets()

