"""
Pydantic schemas for API Gateway
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime
    services: Dict[str, Dict[str, Any]]


class Location(BaseModel):
    """Geographic location"""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class ChatMessage(BaseModel):
    """Chat message"""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request"""
    messages: List[ChatMessage]
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    location: Optional[Location] = None
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0, le=2)
    stream: bool = False

    @validator('image', 'image_url')
    def validate_image_input(cls, v, values):
        """Ensure only one image input method is used"""
        if values.get('image') and values.get('image_url'):
            raise ValueError("Provide either 'image' or 'image_url', not both")
        return v


class ChatResponse(BaseModel):
    """Chat response"""
    response: str
    sources: Optional[List[Dict[str, str]]] = None
    suggestions: Optional[List[str]] = None
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


class VisionClassifyRequest(BaseModel):
    """Vision classification request"""
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    return_probabilities: bool = True
    top_k: int = Field(3, ge=1, le=10)


class ClassificationResult(BaseModel):
    """Single classification result"""
    class_name: str = Field(..., alias="class")
    confidence: float = Field(..., ge=0, le=1)

    class Config:
        populate_by_name = True


class VisionClassifyResponse(BaseModel):
    """Vision classification response"""
    predictions: Dict[str, Any]
    processing_time_ms: float


class VisionDetectRequest(BaseModel):
    """Vision detection request"""
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    conf_threshold: float = Field(0.25, ge=0, le=1)
    iou_threshold: float = Field(0.45, ge=0, le=1)
    max_detections: int = Field(100, ge=1, le=300)


class Detection(BaseModel):
    """Single object detection"""
    bbox: List[float] = Field(..., description="[x, y, width, height]")
    class_name: str = Field(..., alias="class")
    confidence: float = Field(..., ge=0, le=1)
    material: Optional[str] = None
    material_confidence: Optional[float] = None

    class Config:
        populate_by_name = True


class VisionDetectResponse(BaseModel):
    """Vision detection response"""
    detections: List[Detection]
    num_detections: int
    processing_time_ms: float


class OrgSearchRequest(BaseModel):
    """Organization search request"""
    query: str = Field(..., min_length=1, max_length=200)
    location: Location
    radius_km: float = Field(10, ge=0.1, le=100)
    org_type: Optional[Literal["charity", "recycling_facility", "club", "all"]] = "all"
    limit: int = Field(10, ge=1, le=50)


class Organization(BaseModel):
    """Organization information"""
    org_id: str
    name: str
    type: str
    description: Optional[str] = None
    url: Optional[str] = None
    location: Location
    address: Dict[str, str]
    accepted_materials: List[str]
    services: List[str]
    distance_km: float
    rating: Optional[float] = None


class OrgSearchResponse(BaseModel):
    """Organization search response"""
    organizations: List[Organization]
    total_results: int
    processing_time_ms: float


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    status_code: int
    timestamp: str
    details: Optional[Dict[str, Any]] = None




# Vision Analysis Schemas (consolidated from routers/vision.py)
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


class VisionClassificationResult(BaseModel):
    """Complete classification result with multi-head outputs"""
    item_type: str
    item_confidence: float
    material_type: str
    material_confidence: float
    bin_type: str
    bin_confidence: float
    top_k_items: List[tuple]  # List of (str, float) tuples
    top_k_materials: List[tuple]  # List of (str, float) tuples


class RecommendationResult(BaseModel):
    """Upcycling recommendation from GNN"""
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
    classification: Optional[VisionClassificationResult]
    recommendations: Optional[List[RecommendationResult]]
    image_size: tuple  # (width, height)
    image_format: str
    image_quality_score: float
    confidence_score: float
    total_time_ms: float
    detection_time_ms: float
    classification_time_ms: float
    recommendation_time_ms: float
    warnings: List[str]
    errors: List[str]
