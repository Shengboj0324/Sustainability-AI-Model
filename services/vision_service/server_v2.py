"""
Vision Service V2 - Production-grade waste recognition with integrated vision system

CRITICAL FEATURES:
- Handles ANY random customer image (any size, format, quality)
- Complete 3-stage pipeline: Detection → Classification → GNN Recommendations
- Rate limiting (100 req/min per IP)
- Request caching (LRU + TTL)
- Prometheus metrics
- Timeouts on all operations
- Graceful shutdown
- CORS for web + iOS
- Comprehensive error handling
"""

import asyncio
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import base64
import io

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import torch
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

# Import shared utilities - CRITICAL: Single source of truth
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.utils import RateLimiter, RequestCache

# Import our production-grade vision system
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.vision.integrated_vision import IntegratedVisionSystem, IntegratedVisionResult
from models.vision.classifier import ClassificationResult
from models.vision.detector import Detection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUESTS_TOTAL = Counter('vision_requests_total', 'Total vision requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('vision_request_duration_seconds', 'Request duration', ['endpoint'])
ACTIVE_REQUESTS = Gauge('vision_active_requests', 'Active requests')
DETECTION_TIME = Histogram('vision_detection_time_ms', 'Detection time in ms')
CLASSIFICATION_TIME = Histogram('vision_classification_time_ms', 'Classification time in ms')
RECOMMENDATION_TIME = Histogram('vision_recommendation_time_ms', 'Recommendation time in ms')
IMAGE_QUALITY_SCORE = Histogram('vision_image_quality_score', 'Image quality score')
CONFIDENCE_SCORE = Histogram('vision_confidence_score', 'Overall confidence score')

# Initialize FastAPI app
app = FastAPI(
    title="ReleAF AI Vision Service V2",
    description="Production-grade waste recognition service (handles ANY random image)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for web and iOS clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
# NOTE: These schemas are intentionally defined here for microservice independence
# The API Gateway has its own schemas for external API contracts
# This allows the vision service to evolve independently
class VisionRequest(BaseModel):
    """Vision analysis request"""
    image_b64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="Image URL")
    enable_detection: bool = Field(True, description="Enable object detection")
    enable_classification: bool = Field(True, description="Enable classification")
    enable_recommendations: bool = Field(False, description="Enable GNN recommendations")
    top_k: int = Field(5, ge=1, le=20, description="Top-K classification results")


class DetectionResponse(BaseModel):
    """Detection result"""
    bbox: List[float]
    class_name: str
    confidence: float
    area: float


class ClassificationResponse(BaseModel):
    """Classification result"""
    item_type: str
    item_confidence: float
    material_type: str
    material_confidence: float
    bin_type: str
    bin_confidence: float
    top_k_items: List[Tuple[str, float]]
    top_k_materials: List[Tuple[str, float]]


class RecommendationResponse(BaseModel):
    """Upcycling recommendation"""
    target_material: str
    score: float
    difficulty: int
    time_required_minutes: int
    tools_required: List[str]
    skills_required: List[str]


class VisionResponse(BaseModel):
    """Complete vision analysis response"""
    # Detection results
    detections: List[DetectionResponse]
    num_detections: int

    # Classification results
    classification: Optional[ClassificationResponse]

    # Recommendations
    recommendations: Optional[List[RecommendationResponse]]

    # Image metadata
    image_size: Tuple[int, int]
    image_format: str
    image_quality_score: float
    confidence_score: float

    # Performance metrics
    total_time_ms: float
    detection_time_ms: float
    classification_time_ms: float
    recommendation_time_ms: float

    # Quality indicators
    warnings: List[str]
    errors: List[str]


# REMOVED: RateLimiter and RequestCache now imported from shared.utils
# This eliminates code duplication and ensures single source of truth


# Initialize components
rate_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60"))
)

request_cache = RequestCache(
    max_size=int(os.getenv("CACHE_MAX_SIZE", "1000")),
    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300"))
)


class VisionServiceV2:
    """
    Production-grade vision service

    CRITICAL: Handles ANY random customer image with comprehensive error handling
    """
    def __init__(self):
        self.vision_system: Optional[IntegratedVisionSystem] = None
        self._shutdown = False

        # Load configuration
        self.classifier_path = os.getenv("CLASSIFIER_CHECKPOINT_PATH")
        self.detector_path = os.getenv("DETECTOR_CHECKPOINT_PATH")
        self.gnn_path = os.getenv("GNN_CHECKPOINT_PATH")

        logger.info("VisionServiceV2 initialized")

    async def initialize(self):
        """Initialize vision system"""
        try:
            logger.info("Initializing integrated vision system...")
            start_time = time.time()

            # Initialize vision system
            self.vision_system = IntegratedVisionSystem()

            # Load models
            self.vision_system.load_models(
                classifier_path=self.classifier_path,
                detector_path=self.detector_path,
                gnn_path=self.gnn_path
            )

            init_time = time.time() - start_time
            logger.info(f"Vision system initialized in {init_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to initialize vision system: {e}", exc_info=True)
            raise

    async def _load_graph_data(self) -> Optional[Any]:
        """
        Load graph data for GNN recommendations

        Returns None if graph data not available (graceful degradation)
        """
        try:
            graph_data_path = os.getenv("GRAPH_DATA_PATH")
            if not graph_data_path or not os.path.exists(graph_data_path):
                logger.warning("Graph data not found, GNN recommendations will be limited")
                return None

            # Load graph data asynchronously
            import torch
            graph_data = await asyncio.to_thread(torch.load, graph_data_path)
            logger.info(f"Loaded graph data from {graph_data_path}")
            return graph_data

        except Exception as e:
            logger.warning(f"Failed to load graph data: {e}")
            return None

    async def analyze(
        self,
        request: VisionRequest,
        timeout: float = 30.0
    ) -> IntegratedVisionResult:
        """
        Analyze image with timeout

        CRITICAL: Handles ANY random image with comprehensive validation
        """
        try:
            # Load image from source
            image = await asyncio.wait_for(
                self.vision_system.load_image_from_source(
                    image_b64=request.image_b64,
                    image_url=request.image_url
                ),
                timeout=10.0  # 10s timeout for image loading
            )

            # Load graph data if recommendations enabled
            graph_data = None
            if request.enable_recommendations:
                graph_data = await self._load_graph_data()

            # Analyze image
            result = await asyncio.wait_for(
                self.vision_system.analyze_image(
                    image=image,
                    enable_detection=request.enable_detection,
                    enable_classification=request.enable_classification,
                    enable_recommendations=request.enable_recommendations,
                    graph_data=graph_data
                ),
                timeout=timeout
            )

            return result

        except asyncio.TimeoutError:
            logger.error(f"Analysis timeout after {timeout}s")
            raise HTTPException(status_code=504, detail="Analysis timeout")
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise

    async def close(self):
        """Graceful shutdown"""
        self._shutdown = True
        if self.vision_system:
            self.vision_system.cleanup()
        logger.info("Vision service shutdown complete")


# Initialize service
vision_service = VisionServiceV2()



# Lifecycle hooks
@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    await vision_service.initialize()


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown"""
    await vision_service.close()


# API Endpoints
@app.post("/analyze", response_model=VisionResponse)
async def analyze_image(request: VisionRequest, http_request: Request):
    """
    Complete image analysis endpoint

    CRITICAL: Handles ANY random customer image with:
    - Rate limiting (100 req/min per IP)
    - Request caching (5min TTL)
    - Comprehensive validation
    - Graceful error handling
    - Prometheus metrics
    """
    ACTIVE_REQUESTS.inc()
    endpoint = "analyze"
    start_time = time.time()

    try:
        # Get client IP
        client_ip = http_request.client.host

        # Check rate limit
        if not await rate_limiter.check_rate_limit(client_ip):
            REQUESTS_TOTAL.labels(endpoint=endpoint, status="rate_limited").inc()
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Validate request
        if not request.image_b64 and not request.image_url:
            REQUESTS_TOTAL.labels(endpoint=endpoint, status="invalid_request").inc()
            raise HTTPException(status_code=400, detail="Must provide image_b64 or image_url")

        # Check cache (use image hash as key)
        cache_key = None
        if request.image_b64:
            cache_key = f"b64:{hash(request.image_b64)}"
        elif request.image_url:
            cache_key = f"url:{request.image_url}"

        if cache_key:
            cached_result = await request_cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {cache_key[:50]}...")
                REQUESTS_TOTAL.labels(endpoint=endpoint, status="cache_hit").inc()
                return cached_result

        # Analyze image
        result = await vision_service.analyze(request, timeout=30.0)

        # Convert to response format
        detections = [
            DetectionResponse(
                bbox=[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]],
                class_name=d.class_name,
                confidence=d.confidence,
                area=d.area
            )
            for d in result.detections
        ]

        classification = None
        if result.classification:
            c = result.classification
            classification = ClassificationResponse(
                item_type=c.item_type,
                item_confidence=c.item_confidence,
                material_type=c.material_type,
                material_confidence=c.material_confidence,
                bin_type=c.bin_type,
                bin_confidence=c.bin_confidence,
                top_k_items=c.top_k_items,
                top_k_materials=c.top_k_materials
            )

        recommendations = None
        if result.upcycling_recommendations:
            r = result.upcycling_recommendations
            recommendations = [
                RecommendationResponse(
                    target_material=rec["target_material"],
                    score=rec["score"],
                    difficulty=rec.get("difficulty", 3),
                    time_required_minutes=rec.get("time_required_minutes", 60),
                    tools_required=rec.get("tools_required", []),
                    skills_required=rec.get("skills_required", [])
                )
                for rec in r.recommendations
            ]

        response = VisionResponse(
            detections=detections,
            num_detections=result.num_detections,
            classification=classification,
            recommendations=recommendations,
            image_size=result.image_size,
            image_format=result.image_format,
            image_quality_score=result.image_quality_score,
            confidence_score=result.confidence_score,
            total_time_ms=result.total_time_ms,
            detection_time_ms=result.detection_time_ms,
            classification_time_ms=result.classification_time_ms,
            recommendation_time_ms=result.recommendation_time_ms,
            warnings=result.warnings,
            errors=result.errors
        )

        # Cache result
        if cache_key:
            await request_cache.set(cache_key, response)

        # Update metrics
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="success").inc()
        REQUEST_DURATION.labels(endpoint=endpoint).observe(time.time() - start_time)
        DETECTION_TIME.observe(result.detection_time_ms)
        CLASSIFICATION_TIME.observe(result.classification_time_ms)
        RECOMMENDATION_TIME.observe(result.recommendation_time_ms)
        IMAGE_QUALITY_SCORE.observe(result.image_quality_score)
        CONFIDENCE_SCORE.observe(result.confidence_score)

        return response

    except HTTPException:
        raise
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/health")
async def health():
    """
    Health check endpoint for load balancer

    Returns detailed health status for monitoring
    """
    is_healthy = (
        vision_service.vision_system is not None and
        not vision_service._shutdown
    )

    stats = {}
    if vision_service.vision_system:
        stats = vision_service.vision_system.get_stats()

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "vision_v2",
        "version": "2.0.0",
        "vision_system_loaded": vision_service.vision_system is not None,
        "shutdown": vision_service._shutdown,
        "cache_size": len(request_cache.cache),
        "stats": stats
    }


@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    stats = {}
    if vision_service.vision_system:
        stats = vision_service.vision_system.get_stats()

    return {
        "service": "vision_v2",
        "cache_size": len(request_cache.cache),
        "cache_max_size": request_cache.max_size,
        "cache_ttl_seconds": request_cache.ttl_seconds,
        "rate_limit_requests": rate_limiter.max_requests,
        "rate_limit_window": rate_limiter.window_seconds,
        "vision_stats": stats
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/cache/clear")
async def clear_cache():
    """Clear request cache"""
    request_cache.clear()
    return {"status": "cache_cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_v2:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        reload=False,
        log_level="info"
    )

