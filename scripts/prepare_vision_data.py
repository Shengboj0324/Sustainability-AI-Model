#!/usr/bin/env python3
"""
Vision Data Preparation Pipeline
=================================
Downloads, merges, splits, and validates image data for waste classification.

Sources:
  1. Kaggle: alistairking/recyclable-and-household-waste-classification
     (15,000 images, 30 classes × 500)
  2. Kaggle: sumn2u/garbage-classification-v2
     (12,259 images, 10 categories → mapped to matching classes)

Produces:
  data/processed/vision_cls/train/<class>/*.jpg  (80%)
  data/processed/vision_cls/val/<class>/*.jpg    (10%)
  data/processed/vision_cls/test/<class>/*.jpg   (10%)
"""

import os, sys, shutil, random, logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10

# ── Primary: 30-class Recyclable & Household Waste ────────────────────
PRIMARY_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle_waste" / "images" / "images"

# ── Secondary: Garbage Classification v2 (standardized 384px) ─────────
SECONDARY_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle_garbage" / "standardized_384"

# Map secondary dataset classes → primary class names
SECONDARY_CLASS_MAP = {
    "cardboard":  "cardboard_boxes",
    "clothes":    "clothing",
    "glass":      "glass_beverage_bottles",
    "metal":      "steel_food_cans",
    "paper":      "office_paper",
    "plastic":    "plastic_water_bottles",
    "shoes":      "shoes",
    "biological": "food_waste",
    "trash":      "disposable_plastic_cutlery",
    # "battery" has no match → skip
}

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "vision_cls"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def collect_images(source_dir: Path, class_map: dict = None) -> dict:
    """Collect {class_name: [file_paths]} from a directory.

    Handles nested subdirectories (e.g. class/default/*.png, class/real_world/*.png)
    by recursing into each class folder.
    """
    result = defaultdict(list)
    if not source_dir.exists():
        logger.warning(f"Source dir not found: {source_dir}")
        return result

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_map is not None:
            class_name = class_map.get(class_dir.name)
            if class_name is None:
                continue  # Skip unmapped classes

        # Recurse into all subdirectories (handles default/, real_world/, etc.)
        for img in class_dir.rglob("*"):
            if img.is_file() and img.suffix.lower() in IMG_EXTENSIONS:
                result[class_name].append(img)

    return result


def split_and_copy(class_images: dict, output_dir: Path):
    """Split images into train/val/test and copy to output directory."""
    random.seed(SEED)
    stats = {"train": 0, "val": 0, "test": 0}

    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    for cls_name, images in sorted(class_images.items()):
        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split_name, split_imgs in splits.items():
            dst_dir = output_dir / split_name / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_imgs:
                dst = dst_dir / img_path.name
                # Avoid duplicates from secondary dataset
                if dst.exists():
                    dst = dst_dir / f"aug_{img_path.name}"
                shutil.copy2(str(img_path), str(dst))
                stats[split_name] += 1

        logger.info(
            f"  {cls_name:40s} "
            f"train={len(splits['train']):4d}  "
            f"val={len(splits['val']):4d}  "
            f"test={len(splits['test']):4d}  "
            f"total={n}"
        )

    return stats


def validate(output_dir: Path):
    """Validate the prepared dataset."""
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION")
    logger.info("=" * 70)
    all_ok = True
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        if not split_dir.exists():
            logger.error(f"  ❌ Missing split: {split}")
            all_ok = False
            continue
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        total = sum(len(list((split_dir / c).iterdir())) for c in classes)
        logger.info(f"  {split:5s}: {len(classes)} classes, {total:,} images")
        if len(classes) == 0:
            logger.error(f"  ❌ No classes in {split}")
            all_ok = False
    return all_ok


def main():
    logger.info("=" * 70)
    logger.info("VISION DATA PREPARATION PIPELINE")
    logger.info("=" * 70)

    # Clear old data
    if OUTPUT_DIR.exists():
        logger.info(f"Clearing old data: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # 1. Collect primary dataset (30 classes × 500)
    logger.info(f"\n📦 Primary dataset: {PRIMARY_DIR}")
    primary = collect_images(PRIMARY_DIR)
    logger.info(f"   {len(primary)} classes, {sum(len(v) for v in primary.values()):,} images")

    # 2. Collect secondary dataset (mapped to primary classes)
    logger.info(f"\n📦 Secondary dataset: {SECONDARY_DIR}")
    secondary = collect_images(SECONDARY_DIR, class_map=SECONDARY_CLASS_MAP)
    logger.info(f"   {len(secondary)} mapped classes, {sum(len(v) for v in secondary.values()):,} images")

    # 3. Merge
    merged = defaultdict(list)
    for d in (primary, secondary):
        for cls, imgs in d.items():
            merged[cls].extend(imgs)

    total = sum(len(v) for v in merged.values())
    logger.info(f"\n📊 Merged: {len(merged)} classes, {total:,} images")

    # 4. Split and copy
    logger.info(f"\n✂️  Splitting (train={TRAIN_RATIO}/val={VAL_RATIO}/test={TEST_RATIO}):")
    stats = split_and_copy(merged, OUTPUT_DIR)
    logger.info(f"\n📈 Totals: train={stats['train']:,}, val={stats['val']:,}, test={stats['test']:,}")

    # 5. Validate
    ok = validate(OUTPUT_DIR)

    if ok:
        logger.info("\n✅ VISION DATA PREPARATION COMPLETE")
    else:
        logger.error("\n❌ VALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
