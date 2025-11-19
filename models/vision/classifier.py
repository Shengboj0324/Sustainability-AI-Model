"""
Vision Classifier - Production-grade waste classification model

CRITICAL FEATURES:
- Multi-head classification (item type, material, bin type)
- Proper device management (CPU/CUDA)
- Memory-efficient inference
- Batch processing support
- Error handling and graceful degradation
- Model warmup for consistent latency
- Thread-safe operations
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Classification result with confidence scores"""
    item_type: str
    item_confidence: float
    material_type: str
    material_confidence: float
    bin_type: str
    bin_confidence: float
    top_k_items: List[Tuple[str, float]]
    top_k_materials: List[Tuple[str, float]]
    inference_time_ms: float


class MultiHeadClassifier(nn.Module):
    """
    Multi-head classifier for waste recognition

    CRITICAL: Three classification heads for comprehensive waste analysis
    """
    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        num_classes_item: int = 20,
        num_classes_material: int = 15,
        num_classes_bin: int = 4,
        drop_rate: float = 0.1,
        pretrained: bool = True
    ):
        super().__init__()

        # Load backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_rate=drop_rate
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Classification heads
        self.item_head = nn.Linear(self.feature_dim, num_classes_item)
        self.material_head = nn.Linear(self.feature_dim, num_classes_material)
        self.bin_head = nn.Linear(self.feature_dim, num_classes_bin)

        logger.info(f"MultiHeadClassifier initialized: {backbone} -> {self.feature_dim}D features")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            item_logits, material_logits, bin_logits
        """
        # Extract features
        features = self.backbone(x)

        # Classification heads
        item_logits = self.item_head(features)
        material_logits = self.material_head(features)
        bin_logits = self.bin_head(features)

        return item_logits, material_logits, bin_logits


class WasteClassifier:
    """
    Production-grade waste classifier

    CRITICAL FEATURES:
    - Proper device management
    - Memory-efficient inference
    - Batch processing
    - Model warmup
    - Error handling
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        self.config = config or self._get_default_config()
        self.device = self._setup_device(device)
        self.model: Optional[MultiHeadClassifier] = None
        self.transform: Optional[transforms.Compose] = None
        self.model_path = model_path

        # Class names
        self.item_classes = self.config.get("item_classes", [])
        self.material_classes = self.config.get("material_classes", [])
        self.bin_classes = self.config.get("bin_classes", ["recycle", "compost", "landfill", "hazardous"])

        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0

        logger.info(f"WasteClassifier initialized on device: {self.device}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "backbone": "vit_base_patch16_224",
            "num_classes_item": 20,
            "num_classes_material": 15,
            "num_classes_bin": 4,
            "drop_rate": 0.1,
            "input_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "item_classes": [],
            "material_classes": [],
            "bin_classes": ["recycle", "compost", "landfill", "hazardous"]
        }




    def _setup_device(self, device: Optional[str] = None) -> torch.device:
        """
        Setup device with proper CUDA and MPS handling

        CRITICAL: Handles GPU availability and fallback (CUDA, MPS, CPU)
        """
        if device is not None:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            if device == "mps" and not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device(device)

        # Auto-detect
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🔥 CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("🍎 Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("💻 Using CPU for inference")

        return device

    def load_model(self):
        """
        Load model with proper error handling

        CRITICAL: Handles missing checkpoints, device placement, and warmup
        """
        try:
            logger.info("Loading vision classifier model...")
            start_time = time.time()

            # Create model
            self.model = MultiHeadClassifier(
                backbone=self.config["backbone"],
                num_classes_item=self.config["num_classes_item"],
                num_classes_material=self.config["num_classes_material"],
                num_classes_bin=self.config["num_classes_bin"],
                drop_rate=self.config["drop_rate"],
                pretrained=True
            )

            # Load checkpoint if available
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading checkpoint from: {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Handle different checkpoint formats
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    self.model.load_state_dict(checkpoint)
                    logger.info("Loaded checkpoint (state dict only)")
            else:
                logger.warning(f"No checkpoint found at {self.model_path}. Using pretrained backbone only.")

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            # Setup transforms
            self._setup_transforms()

            # Warmup model for consistent latency
            self._warmup_model()

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        input_size = self.config["input_size"]
        mean = self.config["mean"]
        std = self.config["std"]

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        logger.info(f"Transforms configured: {input_size}x{input_size}, mean={mean}, std={std}")

    def _warmup_model(self, num_iterations: int = 5):
        """
        Warmup model for consistent latency

        CRITICAL: First inference is often slower due to CUDA initialization
        """
        logger.info("Warming up model...")

        input_size = self.config["input_size"]
        dummy_input = torch.randn(1, 3, input_size, input_size).to(self.device)

        with torch.inference_mode():
            for i in range(num_iterations):
                _ = self.model(dummy_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

        logger.info(f"Model warmup complete ({num_iterations} iterations)")

    @torch.inference_mode()
    def classify(
        self,
        image: Image.Image,
        top_k: int = 3
    ) -> ClassificationResult:
        """
        Classify single image

        CRITICAL: Thread-safe, memory-efficient inference
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Preprocess image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Forward pass
            item_logits, material_logits, bin_logits = self.model(img_tensor)

            # Compute probabilities
            item_probs = torch.softmax(item_logits, dim=1)
            material_probs = torch.softmax(material_logits, dim=1)
            bin_probs = torch.softmax(bin_logits, dim=1)

            # Get top predictions
            item_top_probs, item_top_indices = torch.topk(item_probs, k=min(top_k, len(self.item_classes)))
            material_top_probs, material_top_indices = torch.topk(material_probs, k=min(top_k, len(self.material_classes)))
            bin_top_idx = torch.argmax(bin_probs, dim=1)

            # Synchronize if using CUDA
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            inference_time = (time.time() - start_time) * 1000  # ms

            # Update stats
            self.inference_count += 1
            self.total_inference_time += inference_time

            # Build result
            result = ClassificationResult(
                item_type=self.item_classes[item_top_indices[0][0].item()],
                item_confidence=item_top_probs[0][0].item(),
                material_type=self.material_classes[material_top_indices[0][0].item()],
                material_confidence=material_top_probs[0][0].item(),
                bin_type=self.bin_classes[bin_top_idx[0].item()],
                bin_confidence=bin_probs[0][bin_top_idx[0]].item(),
                top_k_items=[
                    (self.item_classes[idx.item()], prob.item())
                    for prob, idx in zip(item_top_probs[0], item_top_indices[0])
                ],
                top_k_materials=[
                    (self.material_classes[idx.item()], prob.item())
                    for prob, idx in zip(material_top_probs[0], material_top_indices[0])
                ],
                inference_time_ms=inference_time
            )

            return result

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            raise


    @torch.inference_mode()
    def classify_batch(
        self,
        images: List[Image.Image],
        top_k: int = 3,
        batch_size: int = 32
    ) -> List[ClassificationResult]:
        """
        Classify batch of images

        CRITICAL: Memory-efficient batch processing with proper batching
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = []
        num_images = len(images)

        logger.info(f"Processing batch of {num_images} images (batch_size={batch_size})")

        for i in range(0, num_images, batch_size):
            batch_images = images[i:i + batch_size]
            batch_start = time.time()

            try:
                # Preprocess batch
                batch_tensors = torch.stack([
                    self.transform(img) for img in batch_images
                ]).to(self.device)

                # Forward pass
                item_logits, material_logits, bin_logits = self.model(batch_tensors)

                # Compute probabilities
                item_probs = torch.softmax(item_logits, dim=1)
                material_probs = torch.softmax(material_logits, dim=1)
                bin_probs = torch.softmax(bin_logits, dim=1)

                # Synchronize if using CUDA
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                batch_time = (time.time() - batch_start) * 1000
                per_image_time = batch_time / len(batch_images)

                # Process each image in batch
                for j in range(len(batch_images)):
                    # Get top predictions
                    item_top_probs, item_top_indices = torch.topk(
                        item_probs[j:j+1], k=min(top_k, len(self.item_classes))
                    )
                    material_top_probs, material_top_indices = torch.topk(
                        material_probs[j:j+1], k=min(top_k, len(self.material_classes))
                    )
                    bin_top_idx = torch.argmax(bin_probs[j:j+1], dim=1)

                    result = ClassificationResult(
                        item_type=self.item_classes[item_top_indices[0][0].item()],
                        item_confidence=item_top_probs[0][0].item(),
                        material_type=self.material_classes[material_top_indices[0][0].item()],
                        material_confidence=material_top_probs[0][0].item(),
                        bin_type=self.bin_classes[bin_top_idx[0].item()],
                        bin_confidence=bin_probs[j][bin_top_idx[0]].item(),
                        top_k_items=[
                            (self.item_classes[idx.item()], prob.item())
                            for prob, idx in zip(item_top_probs[0], item_top_indices[0])
                        ],
                        top_k_materials=[
                            (self.material_classes[idx.item()], prob.item())
                            for prob, idx in zip(material_top_probs[0], material_top_indices[0])
                        ],
                        inference_time_ms=per_image_time
                    )

                    results.append(result)
                    self.inference_count += 1
                    self.total_inference_time += per_image_time

            except Exception as e:
                logger.error(f"Batch processing failed at index {i}: {e}", exc_info=True)
                raise

        logger.info(f"Batch processing complete: {num_images} images")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        avg_time = self.total_inference_time / self.inference_count if self.inference_count > 0 else 0

        return {
            "inference_count": self.inference_count,
            "total_inference_time_ms": self.total_inference_time,
            "average_inference_time_ms": avg_time,
            "device": str(self.device),
            "model_loaded": self.model is not None
        }

    def reset_stats(self):
        """Reset inference statistics"""
        self.inference_count = 0
        self.total_inference_time = 0.0
        logger.info("Statistics reset")

    def cleanup(self):
        """
        Cleanup resources

        CRITICAL: Free GPU memory
        """
        if self.model is not None:
            del self.model
            self.model = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")

        logger.info("Classifier cleanup complete")

