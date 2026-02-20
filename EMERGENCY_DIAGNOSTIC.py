#!/usr/bin/env python3
"""
EMERGENCY DIAGNOSTIC - Check what's REALLY happening with the data
Run this to see if images are loading or if dummy tensors are being used
"""

import torch
from pathlib import Path
from PIL import Image
import sys

print("=" * 80)
print("EMERGENCY DIAGNOSTIC - CHECKING DATA LOADING")
print("=" * 80)

# Check if dataset exists
dataset_dir = Path("data/waste_classification")
if not dataset_dir.exists():
    print(f"❌ ERROR: Dataset directory not found: {dataset_dir}")
    sys.exit(1)

print(f"✅ Dataset directory exists: {dataset_dir}")

# Count total images
all_images = list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.png"))
print(f"✅ Found {len(all_images)} total images")

if len(all_images) == 0:
    print("❌ ERROR: NO IMAGES FOUND! Dataset is empty!")
    sys.exit(1)

# Test loading a few images
print("\n" + "=" * 80)
print("TESTING IMAGE LOADING")
print("=" * 80)

for i, img_path in enumerate(all_images[:5]):
    try:
        img = Image.open(img_path)
        img.load()
        print(f"✅ Image {i+1}: {img_path.name} - Size: {img.size}, Mode: {img.mode}")
    except Exception as e:
        print(f"❌ Image {i+1}: {img_path.name} - FAILED: {e}")

# Check label distribution
print("\n" + "=" * 80)
print("CHECKING LABEL DISTRIBUTION")
print("=" * 80)

from collections import Counter
labels = [p.parent.name for p in all_images]
label_counts = Counter(labels)

print(f"Found {len(label_counts)} unique classes:")
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {label}: {count} images")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

