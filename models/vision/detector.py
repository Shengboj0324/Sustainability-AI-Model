"""
YOLOv8 Object Detector
"""

import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from dataclasses import dataclass
import cv2

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection result"""
    bbox: List[float]  # [x, y, w, h] in pixels
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] for visualization
    class_name: str
    class_id: int
    confidence: float
    area: float


@dataclass
class DetectionResult:
    """Detection result for an image"""
    detections: List[Detection]
    num_detections: int
    image_size: Tuple[int, int]  # (width, height)
    inference_time_ms: float
    preprocessing_time_ms: float


class WasteDetector:
    """
    YOLOv8 waste detector
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        self.config = config or self._get_default_config()
        self.device = self._setup_device(device)
        self.model: Optional[YOLO] = None
        self.model_path = model_path

        # Class names from config
        self.class_names = self.config.get("classes", [])

        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0

        logger.info(f"WasteDetector initialized on device: {self.device}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "model_type": "yolov8m",
            "img_size": 640,
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "max_det": 100,
            "agnostic_nms": False,
            "classes": [
                "plastic_bottle", "glass_bottle", "aluminum_can", "steel_can",
                "cardboard_box", "paper", "plastic_bag", "food_container",
                "cup", "textile", "e_waste", "battery", "light_bulb",
                "organic_waste", "styrofoam", "tetra_pak", "mixed_plastic",
                "metal_scrap", "wood", "cigarette_butt", "straw",
                "bottle_cap", "wrapper", "mask", "other"
            ]
        }

    def _setup_device(self, device: Optional[str] = None) -> str:
        """
        Setup device with proper CUDA and MPS handling

        CRITICAL: YOLOv8 uses string device names ("cuda", "mps", "cpu")
        """
        if device is not None:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return "cpu"
            if device == "mps" and not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available. Falling back to CPU.")
                return "cpu"
            return device

        # Auto-detect
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"🔥 CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("🍎 Using Apple Silicon GPU (MPS)")
        else:
            device = "cpu"
            logger.info("💻 Using CPU for inference")

        return device

    def load_model(self):
        """
        Load YOLOv8 model with proper error handling

        CRITICAL: Handles missing checkpoints, device placement, and warmup
        """
        try:
            logger.info("Loading YOLOv8 detector model...")
            start_time = time.time()

            # Try to load custom checkpoint first
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading checkpoint from: {self.model_path}")
                self.model = YOLO(self.model_path)
                logger.info("Loaded custom checkpoint")
            else:
                # Fall back to pretrained model
                model_type = self.config["model_type"]
                logger.warning(f"No checkpoint found at {self.model_path}. Using pretrained {model_type}")
                self.model = YOLO(f"{model_type}.pt")

            # Move to device
            self.model.to(self.device)

            # Warmup model
            self._warmup_model()

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def _warmup_model(self, num_iterations: int = 3):
        """
        Warmup model for consistent latency

        CRITICAL: First inference is often slower
        """
        logger.info("Warming up detector model...")

        img_size = self.config["img_size"]
        dummy_image = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        for i in range(num_iterations):
            _ = self.model(dummy_image, verbose=False)

        logger.info(f"Model warmup complete ({num_iterations} iterations)")

    def _validate_and_preprocess_image(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        """
        Validate and preprocess image for detection

        CRITICAL: Handles ANY random image - any size, format, quality
        """
        start_time = time.time()

        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert("RGB")

            # Convert PIL to numpy array
            img_array = np.array(image)

            # Validate image dimensions
            if img_array.ndim != 3 or img_array.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {img_array.shape}. Expected (H, W, 3)")

            # Check for extremely small images
            h, w = img_array.shape[:2]
            if h < 32 or w < 32:
                logger.warning(f"Image too small ({w}x{h}). Resizing to minimum 32x32")
                img_array = cv2.resize(img_array, (max(32, w), max(32, h)))

            # Check for extremely large images (memory protection)
            max_size = 4096
            if h > max_size or w > max_size:
                logger.warning(f"Image too large ({w}x{h}). Resizing to max {max_size}x{max_size}")
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img_array = cv2.resize(img_array, (new_w, new_h))

            # Validate pixel values
            if img_array.max() == 0:
                raise ValueError("Image is completely black")

            preprocessing_time = (time.time() - start_time) * 1000

            return img_array, preprocessing_time

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}", exc_info=True)
            raise

    def detect(
        self,
        image: Image.Image,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None
    ) -> DetectionResult:
        """
        Detect objects in single image

        CRITICAL: Handles ANY random image with robust error handling
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Use config defaults if not specified
        conf_threshold = conf_threshold or self.config["conf_thres"]
        iou_threshold = iou_threshold or self.config["iou_thres"]
        max_detections = max_detections or self.config["max_det"]

        # Validate and preprocess image
        img_array, preprocess_time = self._validate_and_preprocess_image(image)

        start_time = time.time()

        try:
            # Run detection
            results = self.model(
                img_array,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                agnostic_nms=self.config["agnostic_nms"],
                verbose=False,
                device=self.device
            )

            inference_time = (time.time() - start_time) * 1000

            # Parse results
            detections = []
            for result in results:
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    logger.info("No objects detected in image")
                    continue

                for box in boxes:
                    # Get box coordinates
                    xywh = box.xywh[0].cpu().numpy()
                    xyxy = box.xyxy[0].cpu().numpy()

                    # Get class info
                    class_id = int(box.cls[0].cpu().item())
                    confidence = float(box.conf[0].cpu().item())

                    # Get class name
                    if hasattr(result, 'names') and class_id in result.names:
                        class_name = result.names[class_id]
                    elif class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"

                    # Calculate area
                    area = float(xywh[2] * xywh[3])

                    detection = Detection(
                        bbox=xywh.tolist(),
                        bbox_xyxy=xyxy.tolist(),
                        class_name=class_name,
                        class_id=class_id,
                        confidence=confidence,
                        area=area
                    )

                    detections.append(detection)

            # Update stats
            self.inference_count += 1
            self.total_inference_time += inference_time

            # Build result
            result = DetectionResult(
                detections=detections,
                num_detections=len(detections),
                image_size=(img_array.shape[1], img_array.shape[0]),
                inference_time_ms=inference_time,
                preprocessing_time_ms=preprocess_time
            )

            logger.info(f"Detected {len(detections)} objects in {inference_time:.2f}ms")

            return result

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            raise


    def detect_batch(
        self,
        images: List[Image.Image],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None
    ) -> List[DetectionResult]:
        """
        Detect objects in batch of images

        CRITICAL: Memory-efficient batch processing
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = []
        logger.info(f"Processing batch of {len(images)} images")

        for idx, image in enumerate(images):
            try:
                result = self.detect(image, conf_threshold, iou_threshold, max_detections)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {idx}: {e}")
                # Create empty result for failed image
                results.append(DetectionResult(
                    detections=[],
                    num_detections=0,
                    image_size=(0, 0),
                    inference_time_ms=0.0,
                    preprocessing_time_ms=0.0
                ))

        logger.info(f"Batch processing complete: {len(results)} images")
        return results

    def filter_detections(
        self,
        result: DetectionResult,
        min_confidence: float = 0.5,
        min_area: float = 100.0,
        class_filter: Optional[List[str]] = None
    ) -> DetectionResult:
        """
        Filter detections by confidence, area, and class

        CRITICAL: Post-processing for quality control
        """
        filtered_detections = []

        for detection in result.detections:
            # Filter by confidence
            if detection.confidence < min_confidence:
                continue

            # Filter by area
            if detection.area < min_area:
                continue

            # Filter by class
            if class_filter and detection.class_name not in class_filter:
                continue

            filtered_detections.append(detection)

        return DetectionResult(
            detections=filtered_detections,
            num_detections=len(filtered_detections),
            image_size=result.image_size,
            inference_time_ms=result.inference_time_ms,
            preprocessing_time_ms=result.preprocessing_time_ms
        )

    def get_dominant_object(self, result: DetectionResult) -> Optional[Detection]:
        """
        Get the most dominant object (largest area with high confidence)

        CRITICAL: For single-object classification scenarios
        """
        if result.num_detections == 0:
            return None

        # Score = area * confidence
        best_detection = max(
            result.detections,
            key=lambda d: d.area * d.confidence
        )

        return best_detection

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        avg_time = self.total_inference_time / self.inference_count if self.inference_count > 0 else 0

        return {
            "inference_count": self.inference_count,
            "total_inference_time_ms": self.total_inference_time,
            "average_inference_time_ms": avg_time,
            "device": self.device,
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

        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")

        logger.info("Detector cleanup complete")


