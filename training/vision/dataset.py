"""
Vision Dataset Loaders

CRITICAL: Support multiple dataset formats
- COCO format for detection
- Multi-label classification
- Proper data augmentation
- Class balancing
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WasteClassificationDataset(Dataset):
    """Multi-label waste classification dataset"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        img_size: int = 224
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        
        # Load annotations
        ann_file = self.data_dir / f"{split}_annotations.json"
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = self.annotations['images']
        self.image_annotations = self._group_annotations()

        # CRITICAL FIX: Validate dataset is not empty
        if len(self.images) == 0:
            raise ValueError(f"Dataset is empty for split '{split}'")
        if len(self.images) < 10:
            logger.warning(f"Very small dataset: only {len(self.images)} samples for split '{split}'")

        # Default transform if none provided
        if transform is None:
            self.transform = self._get_default_transform(split == "train")
        else:
            self.transform = transform

        logger.info(f"Loaded {len(self.images)} images for {split}")
    
    def _group_annotations(self) -> Dict:
        """Group annotations by image ID"""
        grouped = defaultdict(list)
        for ann in self.annotations['annotations']:
            grouped[ann['image_id']].append(ann)
        return grouped
    
    def _get_default_transform(self, is_train: bool) -> A.Compose:
        """Get default augmentation pipeline"""
        if is_train:
            return A.Compose([
                A.RandomResizedCrop(self.img_size, self.img_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Get image info
        img_info = self.images[idx]
        img_path = self.data_dir / "images" / self.split / img_info['file_name']

        # CRITICAL FIX: Validate file exists before loading
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load image with error handling
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        # Get annotations
        anns = self.image_annotations.get(img_info['id'], [])

        # CRITICAL FIX: Validate annotations exist (no silent defaults!)
        if not anns:
            raise ValueError(f"Missing annotations for image {img_info['id']} ({img_info['file_name']})")

        if len(anns) > 1:
            logger.warning(f"Multiple annotations for image {img_info['id']}, using first")

        # Extract labels (multi-label)
        item_type = anns[0]['item_type']
        material_type = anns[0]['material_type']
        bin_type = anns[0]['bin_type']

        # CRITICAL FIX: Validate label ranges (prevent invalid labels)
        # Note: Adjust max values based on your actual number of classes
        if not (0 <= item_type < 100):  # Adjust max as needed
            raise ValueError(f"Invalid item_type {item_type} for image {img_info['id']}")
        if not (0 <= material_type < 100):
            raise ValueError(f"Invalid material_type {material_type} for image {img_info['id']}")
        if not (0 <= bin_type < 100):
            raise ValueError(f"Invalid bin_type {bin_type} for image {img_info['id']}")

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        labels = {
            'item_type': torch.tensor(item_type, dtype=torch.long),
            'material_type': torch.tensor(material_type, dtype=torch.long),
            'bin_type': torch.tensor(bin_type, dtype=torch.long)
        }

        return image, labels


class WasteDetectionDataset(Dataset):
    """COCO format waste detection dataset"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        img_size: int = 640
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        
        # Load COCO annotations
        ann_file = self.data_dir / "annotations" / f"{split}.json"
        self.coco = COCO(str(ann_file))
        self.image_ids = list(self.coco.imgs.keys())

        # CRITICAL FIX: Validate dataset is not empty
        if len(self.image_ids) == 0:
            raise ValueError(f"Dataset is empty for split '{split}'")
        if len(self.image_ids) < 10:
            logger.warning(f"Very small dataset: only {len(self.image_ids)} samples for split '{split}'")

        # Default transform if none provided
        if transform is None:
            self.transform = self._get_default_transform(split == "train")
        else:
            self.transform = transform

        logger.info(f"Loaded {len(self.image_ids)} images for {split}")
    
    def _get_default_transform(self, is_train: bool) -> A.Compose:
        """Get default augmentation pipeline for detection"""
        if is_train:
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        else:
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Get image
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.data_dir / "images" / self.split / img_info['file_name']

        # CRITICAL FIX: Validate file exists
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load image with error handling
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Extract bounding boxes and labels
        bboxes = []
        class_labels = []
        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            bboxes.append(bbox)
            class_labels.append(ann['category_id'])

        # Apply transforms
        # CRITICAL FIX: Handle empty bboxes and augmentation that removes all boxes
        if self.transform:
            if len(bboxes) > 0:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                image = transformed['image']
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']

                # Handle case where augmentation removed all boxes
                if len(bboxes) == 0:
                    logger.warning(f"Augmentation removed all boxes for image {img_id}")
            else:
                # No boxes to transform, just transform image
                transformed = self.transform(image=image)
                image = transformed['image']
        
        # Convert to tensors
        target = {
            'boxes': torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4)),
            'labels': torch.tensor(class_labels, dtype=torch.long) if class_labels else torch.zeros((0,), dtype=torch.long),
            'image_id': torch.tensor(img_id)
        }
        
        return image, target


def get_balanced_sampler(dataset: WasteClassificationDataset) -> WeightedRandomSampler:
    """Create weighted sampler for class balancing"""
    # Count samples per class
    class_counts = defaultdict(int)
    for idx in range(len(dataset)):
        _, labels = dataset[idx]
        item_type = labels['item_type'].item()
        class_counts[item_type] += 1
    
    # Calculate weights
    total_samples = len(dataset)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    # Assign weight to each sample
    sample_weights = []
    for idx in range(len(dataset)):
        _, labels = dataset[idx]
        item_type = labels['item_type'].item()
        sample_weights.append(class_weights[item_type])
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

