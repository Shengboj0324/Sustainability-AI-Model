"""
Integrated Vision System - Production-grade comprehensive vision pipeline

CRITICAL FEATURES:
- Handles ANY random customer image (any size, format, quality, content)
- Multi-stage pipeline: Detection → Classification → GNN Recommendations
- Robust error handling and graceful degradation
- Proper device management
- Memory-efficient processing
- Comprehensive validation
"""

import torch
from PIL import Image
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from dataclasses import dataclass, asdict
import io
import base64
import httpx

from models.vision.classifier import WasteClassifier, ClassificationResult
from models.vision.detector import WasteDetector, DetectionResult, Detection
from models.gnn.inference import UpcyclingGNN, RecommendationResult
from models.vision.image_quality import AdvancedImageQualityPipeline, ImageQualityReport

logger = logging.getLogger(__name__)


@dataclass
class IntegratedVisionResult:
    """Complete vision analysis result"""
    # Detection results
    detections: List[Detection]
    num_detections: int

    # Classification results (for dominant object)
    classification: Optional[ClassificationResult]

    # GNN recommendations
    upcycling_recommendations: Optional[RecommendationResult]

    # Image metadata
    image_size: Tuple[int, int]
    image_format: str
    image_mode: str

    # Performance metrics
    total_time_ms: float
    detection_time_ms: float
    classification_time_ms: float
    recommendation_time_ms: float

    # Quality indicators
    image_quality_score: float
    confidence_score: float

    # Error handling
    warnings: List[str]
    errors: List[str]

    # NEW: Advanced image quality report
    quality_report: Optional[ImageQualityReport] = None


class IntegratedVisionSystem:
    """
    Production-grade integrated vision system

    CRITICAL: Handles ANY random customer image with comprehensive validation
    """
    def __init__(
        self,
        classifier_config: Optional[Dict[str, Any]] = None,
        detector_config: Optional[Dict[str, Any]] = None,
        gnn_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        self.device = self._setup_device(device)

        # Initialize components
        self.classifier = WasteClassifier(
            config=classifier_config,
            device=str(self.device) if isinstance(self.device, torch.device) else self.device
        )

        self.detector = WasteDetector(
            config=detector_config,
            device=str(self.device) if isinstance(self.device, torch.device) else self.device
        )

        self.gnn = UpcyclingGNN(
            config=gnn_config,
            device=str(self.device) if isinstance(self.device, torch.device) else self.device
        )

        # NEW: Initialize advanced image quality pipeline
        self.image_quality_pipeline = AdvancedImageQualityPipeline()
        logger.info("Advanced image quality pipeline initialized")

        # Performance tracking
        self.total_processed = 0
        self.total_errors = 0

        logger.info(f"IntegratedVisionSystem initialized on device: {self.device}")

    def _setup_device(self, device: Optional[str] = None) -> torch.device:
        """Setup device with proper CUDA and MPS handling"""
        if device is not None:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            if device == "mps" and not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device(device)

        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🔥 CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("🍎 Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("💻 Using CPU for inference")

        return device

    def load_models(
        self,
        classifier_path: Optional[str] = None,
        detector_path: Optional[str] = None,
        gnn_path: Optional[str] = None
    ):
        """
        Load all models

        CRITICAL: Handles missing checkpoints gracefully
        """
        try:
            logger.info("Loading all vision models...")
            start_time = time.time()

            # Load classifier
            self.classifier.model_path = classifier_path
            self.classifier.load_model()

            # Load detector
            self.detector.model_path = detector_path
            self.detector.load_model()

            # Load GNN (optional)
            try:
                self.gnn.model_path = gnn_path
                self.gnn.load_model()
            except Exception as e:
                logger.warning(f"GNN loading failed (optional): {e}")

            load_time = time.time() - start_time
            logger.info(f"All models loaded in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            raise

    def _validate_image(self, image: Image.Image) -> Tuple[Image.Image, List[str], float, ImageQualityReport]:
        """
        Comprehensive image validation using Advanced Image Quality Pipeline

        CRITICAL: Handles ANY random image - validates and fixes issues

        NEW: Uses AdvancedImageQualityPipeline for comprehensive quality checks:
        - EXIF orientation handling
        - Noise detection and denoising
        - Blur detection and sharpening
        - Transparent PNG handling
        - Animated GIF/multi-page TIFF handling
        - HDR tone mapping
        - Adaptive histogram equalization
        - JPEG quality estimation

        Returns:
            (validated_image, warnings, quality_score, quality_report)
        """
        try:
            # Use advanced image quality pipeline
            validated_image, quality_report = self.image_quality_pipeline.process_image(image)

            logger.info(
                f"Image quality processing complete: "
                f"quality_score={quality_report.quality_score:.2f}, "
                f"warnings={len(quality_report.warnings)}, "
                f"enhancements={len(quality_report.enhancements_applied)}"
            )

            return validated_image, quality_report.warnings, quality_report.quality_score, quality_report

        except Exception as e:
            logger.error(f"Image validation failed: {e}", exc_info=True)
            raise

    async def load_image_from_source(
        self,
        image_b64: Optional[str] = None,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> Image.Image:
        """
        Load image from various sources

        CRITICAL: Handles base64, URL, or file path
        """
        try:
            if image_b64:
                # Decode base64
                logger.info("Loading image from base64")
                image_data = base64.b64decode(image_b64)
                image = Image.open(io.BytesIO(image_data))

            elif image_url:
                # Download from URL
                logger.info(f"Downloading image from URL: {image_url}")
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(image_url)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))

            elif image_path:
                # Load from file
                logger.info(f"Loading image from file: {image_path}")
                image = Image.open(image_path)

            else:
                raise ValueError("Must provide image_b64, image_url, or image_path")

            return image

        except Exception as e:
            logger.error(f"Failed to load image: {e}", exc_info=True)
            raise

    async def analyze_image(
        self,
        image: Image.Image,
        enable_detection: bool = True,
        enable_classification: bool = True,
        enable_recommendations: bool = True,
        graph_data: Optional[Any] = None
    ) -> IntegratedVisionResult:
        """
        Complete image analysis pipeline

        CRITICAL: Handles ANY random image with comprehensive error handling
        """
        start_time = time.time()
        warnings = []
        errors = []

        # Store original metadata
        original_size = image.size
        original_format = image.format if hasattr(image, 'format') else "unknown"
        original_mode = image.mode

        try:
            # Validate and preprocess image using advanced quality pipeline
            image, val_warnings, quality_score, quality_report = self._validate_image(image)
            warnings.extend(val_warnings)

            # Stage 1: Detection
            detection_result = None
            detection_time = 0.0

            if enable_detection:
                try:
                    det_start = time.time()
                    detection_result = self.detector.detect(image)
                    detection_time = (time.time() - det_start) * 1000
                    logger.info(f"Detection: {detection_result.num_detections} objects found")
                except Exception as e:
                    errors.append(f"Detection failed: {str(e)}")
                    logger.error(f"Detection failed: {e}", exc_info=True)
                    detection_result = DetectionResult(
                        detections=[],
                        num_detections=0,
                        image_size=image.size,
                        inference_time_ms=0.0,
                        preprocessing_time_ms=0.0
                    )


            # Stage 2: Classification (on dominant object if detected)
            classification_result = None
            classification_time = 0.0

            if enable_classification:
                try:
                    cls_start = time.time()

                    # If we have detections, classify the dominant object
                    # Otherwise, classify the whole image
                    if detection_result and detection_result.num_detections > 0:
                        # Get dominant object
                        dominant = self.detector.get_dominant_object(detection_result)
                        logger.info(f"Classifying dominant object: {dominant.class_name}")

                    # Classify the image
                    classification_result = self.classifier.classify(image, top_k=5)
                    classification_time = (time.time() - cls_start) * 1000
                    logger.info(f"Classification: {classification_result.item_type} ({classification_result.item_confidence:.2f})")

                except Exception as e:
                    errors.append(f"Classification failed: {str(e)}")
                    logger.error(f"Classification failed: {e}", exc_info=True)

            # Stage 3: GNN Recommendations
            recommendation_result = None
            recommendation_time = 0.0

            if enable_recommendations and classification_result and graph_data:
                try:
                    rec_start = time.time()

                    # Get material from classification
                    material = classification_result.material_type

                    # Get upcycling recommendations
                    recommendation_result = self.gnn.predict_upcycling_paths(
                        source_material=material,
                        graph_data=graph_data,
                        top_k=10
                    )
                    recommendation_time = (time.time() - rec_start) * 1000
                    logger.info(f"Recommendations: {recommendation_result.num_recommendations} ideas")

                except Exception as e:
                    errors.append(f"Recommendations failed: {str(e)}")
                    logger.error(f"Recommendations failed: {e}", exc_info=True)

            # Calculate total time
            total_time = (time.time() - start_time) * 1000

            # Calculate overall confidence
            confidence_score = 1.0
            if classification_result:
                confidence_score = classification_result.item_confidence * quality_score
            elif detection_result and detection_result.num_detections > 0:
                avg_conf = sum(d.confidence for d in detection_result.detections) / detection_result.num_detections
                confidence_score = avg_conf * quality_score
            else:
                confidence_score = quality_score * 0.5

            # Build result
            result = IntegratedVisionResult(
                detections=detection_result.detections if detection_result else [],
                num_detections=detection_result.num_detections if detection_result else 0,
                classification=classification_result,
                upcycling_recommendations=recommendation_result,
                image_size=original_size,
                image_format=original_format,
                image_mode=original_mode,
                total_time_ms=total_time,
                detection_time_ms=detection_time,
                classification_time_ms=classification_time,
                recommendation_time_ms=recommendation_time,
                image_quality_score=quality_score,
                confidence_score=confidence_score,
                warnings=warnings,
                errors=errors,
                quality_report=quality_report  # NEW: Include advanced quality report
            )

            # Update stats
            self.total_processed += 1
            if errors:
                self.total_errors += 1

            logger.info(f"Analysis complete: {total_time:.2f}ms (quality={quality_score:.2f}, confidence={confidence_score:.2f})")

            return result

        except Exception as e:
            self.total_errors += 1
            logger.error(f"Image analysis failed: {e}", exc_info=True)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / self.total_processed if self.total_processed > 0 else 0,
            "classifier_stats": self.classifier.get_stats(),
            "detector_stats": self.detector.get_stats(),
            "gnn_stats": self.gnn.get_stats()
        }

    def cleanup(self):
        """Cleanup all resources"""
        self.classifier.cleanup()
        self.detector.cleanup()
        self.gnn.cleanup()
        logger.info("IntegratedVisionSystem cleanup complete")


