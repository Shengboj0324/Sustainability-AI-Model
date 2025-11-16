"""
Vision Router - Handles image analysis endpoints

Routes requests to vision service for waste recognition and classification
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import httpx
import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Service URLs
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://localhost:8001")


class VisionRequest(BaseModel):
    """Vision analysis request"""
    image_b64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="Image URL")
    enable_detection: bool = Field(True, description="Enable object detection")
    enable_classification: bool = Field(True, description="Enable classification")
    enable_recommendations: bool = Field(False, description="Enable GNN recommendations")
    top_k: int = Field(5, ge=1, le=20, description="Top-K classification results")


class DetectionResult(BaseModel):
    """Detection result"""
    bbox: List[float]
    class_name: str
    confidence: float
    area: float


class ClassificationResult(BaseModel):
    """Classification result"""
    item_type: str
    item_confidence: float
    material_type: str
    material_confidence: float
    bin_type: str
    bin_confidence: float
    top_k_items: List[Tuple[str, float]]
    top_k_materials: List[Tuple[str, float]]


class RecommendationResult(BaseModel):
    """Upcycling recommendation"""
    target_material: str
    score: float
    difficulty: int
    time_required_minutes: int
    tools_required: List[str]
    skills_required: List[str]


class VisionResponse(BaseModel):
    """Complete vision analysis response"""
    detections: List[DetectionResult]
    num_detections: int
    classification: Optional[ClassificationResult]
    recommendations: Optional[List[RecommendationResult]]
    image_size: Tuple[int, int]
    image_format: str
    image_quality_score: float
    confidence_score: float
    total_time_ms: float
    detection_time_ms: float
    classification_time_ms: float
    recommendation_time_ms: float
    warnings: List[str]
    errors: List[str]


@router.post("/analyze", response_model=VisionResponse)
async def analyze_image(request: VisionRequest, http_request: Request):
    """
    Complete image analysis endpoint
    
    CRITICAL: Handles ANY random customer image
    - Object detection (YOLOv8)
    - Classification (ViT multi-head)
    - Upcycling recommendations (GNN)
    """
    try:
        # Validate request
        if not request.image_b64 and not request.image_url:
            raise HTTPException(status_code=400, detail="Must provide image_b64 or image_url")
        
        # Call vision service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{VISION_SERVICE_URL}/analyze",
                json=request.dict()
            )
            response.raise_for_status()
            result = response.json()
        
        return VisionResponse(**result)
        
    except httpx.TimeoutException:
        logger.error("Vision service timeout")
        raise HTTPException(status_code=504, detail="Vision analysis timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"Vision service error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Vision analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect")
async def detect_objects(request: VisionRequest):
    """
    Object detection only endpoint
    
    Faster endpoint for just detecting objects without classification
    """
    try:
        # Force detection only
        request.enable_detection = True
        request.enable_classification = False
        request.enable_recommendations = False
        
        return await analyze_image(request, None)
        
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify")
async def classify_image(request: VisionRequest):
    """
    Classification only endpoint
    
    For classifying the whole image without object detection
    """
    try:
        # Force classification only
        request.enable_detection = False
        request.enable_classification = True
        request.enable_recommendations = False
        
        return await analyze_image(request, None)
        
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """Health check"""
    # Check vision service health
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{VISION_SERVICE_URL}/health")
            vision_health = response.json()
        
        return {
            "status": "healthy" if vision_health.get("status") == "healthy" else "unhealthy",
            "router": "vision",
            "vision_service": vision_health
        }
    except Exception as e:
        logger.error(f"Vision service health check failed: {e}")
        return {"status": "unhealthy", "router": "vision", "error": str(e)}

