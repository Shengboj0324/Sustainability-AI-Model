"""
RAG Service - Retrieval-Augmented Generation for sustainability knowledge

- Async Qdrant client with connection pooling
- Request caching for mobile clients
- Rate limiting and timeouts
- Prometheus metrics
- Graceful shutdown
- Memory-efficient model loading
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import sys
import yaml
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import os
import hashlib
from functools import lru_cache
import time
import uuid

# Import shared utilities - CRITICAL: Single source of truth
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.utils import QueryCache

# Import monitoring components
from common.structured_logging import get_logger, log_context, set_correlation_id
from common.health_checks import HealthChecker, check_qdrant_health, HealthStatus
from common.alerting import init_alerting, send_alert, AlertSeverity
from common.circuit_breaker import CircuitBreaker

# Try to import optional monitoring components
try:
    from common.tracing import init_tracing, trace_operation
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    def trace_operation(name, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from common.error_tracking import init_sentry, capture_exception, add_breadcrumb
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    def capture_exception(exc, **kwargs):
        pass
    def add_breadcrumb(msg, **kwargs):
        pass

# Import provenance system - NEW: Enhanced embedding provenance
from provenance import (
    EmbeddingMetadata,
    DataLineage,
    TrustIndicators,
    ProvenanceValidator,
    generate_checksum,
    get_utc_timestamp,
    PROVENANCE_SCHEMA_VERSION
)
from version_tracker import EmbeddingVersionTracker

# Import audit trail and transparency API - Phase 2 integration
from audit_trail import AuditTrailManager, EventType, EntityType, ActorType, Action
from transparency_api import create_transparency_router

# Third-party imports
try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from starlette.responses import Response
except ImportError as e:
    logging.error(f"Missing dependencies: {e}. Install with: pip install qdrant-client sentence-transformers prometheus-client")
    raise

# Configure structured logging
logger = get_logger(__name__)

# Prometheus metrics
REQUESTS_TOTAL = Counter('rag_requests_total', 'Total RAG requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'Request duration', ['endpoint'])
EMBEDDING_DURATION = Histogram('rag_embedding_duration_seconds', 'Embedding generation duration')
RETRIEVAL_DURATION = Histogram('rag_retrieval_duration_seconds', 'Retrieval duration')
RERANK_DURATION = Histogram('rag_rerank_duration_seconds', 'Re-ranking duration')
CACHE_HITS = Counter('rag_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('rag_cache_misses_total', 'Cache misses')
ACTIVE_REQUESTS = Gauge('rag_active_requests', 'Active requests')

# Initialize FastAPI app
app = FastAPI(
    title="ReleAF AI RAG Service",
    description="Retrieval-Augmented Generation service for sustainability knowledge (Production)",
    version="0.1.0",
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

# Initialize monitoring components
health_checker = HealthChecker(service_name="rag_service", check_timeout=5.0)
alert_manager = None  # Initialized in startup
qdrant_circuit_breaker = CircuitBreaker(
    name="qdrant",
    failure_threshold=5,
    recovery_timeout=30.0,
    expected_exception=Exception
)

# REMOVED: RateLimiter now imported from shared.utils
# This eliminates code duplication and ensures single source of truth

# Import RateLimiter from shared utilities
from shared.utils import RateLimiter

# Initialize rate limiter
rate_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60"))
)

# REMOVED: QueryCache now imported from shared.utils
# This eliminates code duplication and ensures single source of truth
# Note: Metrics tracking (CACHE_HITS/CACHE_MISSES) moved to endpoint level

# Global cache instance
query_cache = QueryCache(
    max_size=int(os.getenv("CACHE_SIZE", "1000")),
    ttl_seconds=int(os.getenv("CACHE_TTL", "300"))
)


class RetrievalMode(str, Enum):
    """Retrieval modes"""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class DocumentType(str, Enum):
    """Document types in the knowledge base"""
    RECYCLING_GUIDELINE = "recycling_guideline"
    UPCYCLING_PROJECT = "upcycling_project"
    MATERIAL_PROPERTY = "material_property"
    SAFETY_INFO = "safety_info"
    GENERAL_KNOWLEDGE = "general_knowledge"


@dataclass
class RetrievedDocument:
    """
    Retrieved document with comprehensive provenance metadata

    Enhanced with full embedding provenance tracking including:
    - Embedding metadata (model version, generation time, checksums)
    - Data lineage (source tracking, processing history, updates)
    - Trust indicators (quality scores, verification status, usage stats)

    Backward compatible: All new fields have defaults.
    """
    # Core fields (existing)
    content: str
    score: float
    doc_id: str
    doc_type: str
    metadata: Dict[str, Any]
    source: Optional[str] = None

    # NEW: Enhanced provenance fields
    embedding_metadata: Optional[EmbeddingMetadata] = None
    lineage: Optional[DataLineage] = None
    trust_indicators: Optional[TrustIndicators] = None

    def __post_init__(self):
        """Initialize default provenance if not provided"""
        if self.embedding_metadata is None:
            self.embedding_metadata = EmbeddingMetadata()
        if self.lineage is None:
            self.lineage = DataLineage(original_source=self.source or "unknown")
        if self.trust_indicators is None:
            self.trust_indicators = TrustIndicators()

    @property
    def freshness_score(self) -> float:
        """
        Calculate freshness score based on age

        Returns:
            Float between 0.0 and 1.0 (1.0 = very fresh, 0.0 = very old)
        """
        if not self.lineage or not self.lineage.last_updated:
            return 0.5  # Default for unknown age

        try:
            updated_time = datetime.fromisoformat(self.lineage.last_updated.replace('Z', '+00:00'))
            now = datetime.now(updated_time.tzinfo)
            age_days = (now - updated_time).days

            # Exponential decay: 1.0 at 0 days, 0.5 at 180 days, 0.1 at 365 days
            freshness = max(0.0, min(1.0, 1.0 - (age_days / 365.0)))
            return round(freshness, 3)

        except Exception as e:
            logger.warning(f"Failed to calculate freshness score: {e}")
            return 0.5

    @property
    def overall_trust_score(self) -> float:
        """
        Get overall trust score

        Returns:
            Float between 0.0 and 1.0
        """
        if self.trust_indicators:
            return self.trust_indicators.trust_score
        return 0.5

    def to_dict(self, include_provenance: bool = True) -> Dict[str, Any]:
        """
        Convert to dictionary for API response

        Args:
            include_provenance: Whether to include full provenance metadata

        Returns:
            Dictionary representation
        """
        result = {
            "content": self.content,
            "score": self.score,
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
            "source": self.source
        }

        if include_provenance:
            result["embedding_metadata"] = self.embedding_metadata.to_dict() if self.embedding_metadata else {}
            result["lineage"] = self.lineage.to_dict() if self.lineage else {}
            result["trust_indicators"] = self.trust_indicators.to_dict() if self.trust_indicators else {}
            result["freshness_score"] = self.freshness_score
            result["overall_trust_score"] = self.overall_trust_score

        return result


class RetrievalRequest(BaseModel):
    """RAG retrieval request with optional provenance"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of documents to retrieve")
    mode: RetrievalMode = Field(default=RetrievalMode.HYBRID, description="Retrieval mode")
    doc_types: Optional[List[DocumentType]] = Field(default=None, description="Filter by document types")
    location: Optional[Dict[str, float]] = Field(default=None, description="User location for local rules")
    rerank: bool = Field(default=True, description="Apply re-ranking")
    include_provenance: bool = Field(default=True, description="Include embedding provenance metadata")

    @validator('location')
    def validate_location(cls, v):
        """Validate location coordinates"""
        if v is not None:
            if 'lat' not in v or 'lon' not in v:
                raise ValueError("Location must contain 'lat' and 'lon' keys")
            if not (-90 <= v['lat'] <= 90):
                raise ValueError("Latitude must be between -90 and 90")
            if not (-180 <= v['lon'] <= 180):
                raise ValueError("Longitude must be between -180 and 180")
        return v


class RetrievalResponse(BaseModel):
    """RAG retrieval response"""
    documents: List[Dict[str, Any]]
    query: str
    num_results: int
    retrieval_time_ms: float
    metadata: Dict[str, Any]


class RAGService:
    """
    Production-grade RAG service optimized for Digital Ocean deployment

    Features:
    - Async Qdrant client with connection pooling
    - Request timeouts
    - Memory-efficient model loading
    - Graceful shutdown
    """

    def __init__(self, config_path: str = None):
        """Initialize RAG service with enhanced provenance tracking"""
        if config_path is None:
            config_path = os.getenv("RAG_CONFIG_PATH", "configs/rag.yaml")

        self.config = self._load_config(config_path)
        self.embedding_model: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None
        self.qdrant_client: Optional[AsyncQdrantClient] = None
        self.collection_name = self.config.get("qdrant", {}).get("collection_name", "sustainability_docs")
        self.embedding_dim = self.config.get("embedding", {}).get("dimension", 1024)
        self._shutdown = False

        # NEW: Initialize version tracker for embedding provenance
        self.version_tracker = EmbeddingVersionTracker()

        # NEW: Track current model information
        self.model_name = self.config.get("embedding", {}).get("model_name", "BAAI/bge-large-en-v1.5")
        self.model_version = self._extract_model_version(self.model_name)
        self.pooling_strategy = self.config.get("embedding", {}).get("pooling", "mean")
        self.normalize_embeddings = self.config.get("embedding", {}).get("normalize_embeddings", True)

        # PHASE 2: Initialize audit trail manager
        self.audit_manager = AuditTrailManager(
            storage_type=os.getenv("AUDIT_STORAGE_TYPE", "json"),
            json_dir=os.getenv("AUDIT_JSON_DIR", "data/audit_trail"),
            pg_connection_string=os.getenv("AUDIT_PG_CONNECTION"),
            batch_size=int(os.getenv("AUDIT_BATCH_SIZE", "100")),
            flush_interval_seconds=float(os.getenv("AUDIT_FLUSH_INTERVAL", "5.0"))
        )
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with validation"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return self._get_default_config()
            
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with environment variable overrides"""
        return {
            "embedding": {
                "model_name": os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
                "dimension": int(os.getenv("EMBEDDING_DIM", "1024"))
            },
            "qdrant": {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333")),
                "collection_name": os.getenv("QDRANT_COLLECTION", "sustainability_docs"),
                "timeout": int(os.getenv("QDRANT_TIMEOUT", "30")),
                "grpc_port": int(os.getenv("QDRANT_GRPC_PORT", "6334")),
                "prefer_grpc": os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"
            },
            "retrieval": {
                "dense_top_k": int(os.getenv("RETRIEVAL_TOP_K", "10")),
                "sparse_top_k": 10,
                "fusion_weights": {"dense": 0.6, "sparse": 0.4},
                "timeout": int(os.getenv("RETRIEVAL_TIMEOUT", "10"))
            },
            "reranking": {
                "model_name": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                "enabled": os.getenv("RERANKING_ENABLED", "true").lower() == "true"
            }
        }

    def _extract_model_version(self, model_name: str) -> str:
        """
        Extract model version from model name or metadata

        Args:
            model_name: Model identifier (e.g., "BAAI/bge-large-en-v1.5")

        Returns:
            Version string (e.g., "1.5.0")
        """
        try:
            # Try to extract version from model name
            # Common patterns: v1.5, v2.0, -1.5, etc.
            import re

            # Pattern 1: v1.5 or v2.0
            match = re.search(r'v(\d+)\.(\d+)', model_name)
            if match:
                major, minor = match.groups()
                return f"{major}.{minor}.0"

            # Pattern 2: -1.5 or -2.0
            match = re.search(r'-(\d+)\.(\d+)', model_name)
            if match:
                major, minor = match.groups()
                return f"{major}.{minor}.0"

            # Pattern 3: Check if model has config.json with version info
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
                if hasattr(config, 'version'):
                    return str(config.version)
                if hasattr(config, 'model_version'):
                    return str(config.model_version)
            except Exception as e:
                logger.debug(f"Could not load model config for version extraction: {e}")

            # Fallback: Use model name hash as version identifier
            import hashlib
            version_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
            logger.warning(f"Could not extract version from model name '{model_name}', using hash: {version_hash}")
            return f"unknown-{version_hash}"

        except Exception as e:
            logger.error(f"Error extracting model version: {e}")
            return "unknown"

    async def initialize(self):
        """Initialize models and connections"""
        try:
            logger.info("Initializing RAG service...")

            # Load embedding model
            await self._load_embedding_model()

            # Load reranker
            await self._load_reranker()

            # Connect to Qdrant
            await self._connect_qdrant()

            # PHASE 2: Initialize audit trail manager
            await self.audit_manager.async_init()
            logger.info("Audit trail manager initialized")

            logger.info("RAG service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}", exc_info=True)
            raise

    async def _load_embedding_model(self):
        """
        Load sentence transformer model with proper device placement

        CRITICAL: Handles GPU/CPU placement, memory management, and error recovery
        """
        try:
            model_name = self.config["embedding"]["model_name"]
            device = os.getenv("EMBEDDING_DEVICE", "cpu")  # cpu, cuda, or mps

            logger.info(f"Loading embedding model: {model_name} on device: {device}")

            # Check device availability
            if device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("CUDA requested but not available. Falling back to CPU.")
                        device = "cpu"
                    else:
                        logger.info(f"🔥 CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
                except ImportError:
                    logger.warning("PyTorch not available. Using CPU.")
                    device = "cpu"
            elif device == "mps":
                try:
                    import torch
                    if not torch.backends.mps.is_available():
                        logger.warning("MPS requested but not available. Falling back to CPU.")
                        device = "cpu"
                    else:
                        logger.info("🍎 MPS available. Using Apple Silicon GPU")
                except ImportError:
                    logger.warning("PyTorch not available. Using CPU.")
                    device = "cpu"

            # Run in thread pool to avoid blocking event loop
            # FIX: Use asyncio.to_thread() instead of deprecated get_event_loop()
            def load_model():
                """Load model in thread pool"""
                try:
                    model = SentenceTransformer(model_name, device=device)
                    # Set to eval mode for inference
                    model.eval()
                    return model
                except Exception as e:
                    logger.error(f"Model loading failed in thread: {e}")
                    raise

            self.embedding_model = await asyncio.wait_for(
                asyncio.to_thread(load_model),
                timeout=120.0  # 2 minute timeout for model download/loading
            )

            logger.info(f"Embedding model loaded successfully: {model_name} (device: {device})")

        except asyncio.TimeoutError:
            logger.error("Embedding model loading timeout (120s)")
            raise RuntimeError("Model loading timeout")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise

    async def _load_reranker(self):
        """
        Load cross-encoder reranker with proper error handling

        CRITICAL: Gracefully degrades if reranker fails to load
        """
        try:
            if not self.config["reranking"]["enabled"]:
                logger.info("Re-ranking disabled by configuration")
                return

            model_name = self.config["reranking"]["model_name"]
            device = os.getenv("RERANKER_DEVICE", "cpu")

            logger.info(f"Loading reranker: {model_name} on device: {device}")

            # Check CUDA availability
            if device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("CUDA requested for reranker but not available. Using CPU.")
                        device = "cpu"
                except ImportError:
                    device = "cpu"

            # FIX: Use asyncio.to_thread() instead of deprecated get_event_loop()
            def load_reranker():
                """Load reranker in thread pool"""
                try:
                    # CrossEncoder doesn't have device parameter in constructor
                    # It will use CUDA if available by default
                    reranker = CrossEncoder(model_name)
                    return reranker
                except Exception as e:
                    logger.error(f"Reranker loading failed in thread: {e}")
                    raise

            self.reranker = await asyncio.wait_for(
                asyncio.to_thread(load_reranker),
                timeout=120.0  # 2 minute timeout
            )

            logger.info(f"Reranker loaded successfully: {model_name}")

        except asyncio.TimeoutError:
            logger.warning("Reranker loading timeout. Continuing without re-ranking.")
            self.reranker = None
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Continuing without re-ranking.")
            self.reranker = None

    async def _connect_qdrant(self):
        """Connect to Qdrant vector database with async client"""
        try:
            qdrant_config = self.config["qdrant"]
            host = qdrant_config.get("host", "localhost")
            port = qdrant_config.get("port", 6333)
            timeout = qdrant_config.get("timeout", 30)
            grpc_port = qdrant_config.get("grpc_port", 6334)
            prefer_grpc = qdrant_config.get("prefer_grpc", True)

            logger.info(f"Connecting to Qdrant at {host}:{port} (gRPC: {prefer_grpc})")

            # Use async client with connection pooling
            self.qdrant_client = AsyncQdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
                # Connection pool settings for production
                limits={
                    "max_connections": 100,
                    "max_keepalive_connections": 20
                }
            )

            # Check if collection exists
            collections = await self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.warning(f"Collection '{self.collection_name}' not found. Creating...")
                await self._create_collection()
            else:
                logger.info(f"Connected to collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def _create_collection(self):
        """Create Qdrant collection"""
        try:
            await self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    async def close(self):
        """Graceful shutdown - close connections and cleanup resources"""
        try:
            self._shutdown = True
            logger.info("Shutting down RAG service...")

            # Close Qdrant connection
            if self.qdrant_client:
                await self.qdrant_client.close()
                logger.info("Qdrant connection closed")

            # Clear cache
            await query_cache.clear()
            logger.info("Cache cleared")

            # PHASE 2: Close audit trail manager
            if self.audit_manager:
                await self.audit_manager.close()
                logger.info("Audit trail manager closed")

            # Note: SentenceTransformer models don't need explicit cleanup
            # They will be garbage collected
            logger.info("RAG service shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for query with timeout

        NOTE: This method returns only the embedding vector for backward compatibility.
        For full provenance metadata, use embed_query_with_provenance().
        """
        try:
            if self.embedding_model is None:
                raise RuntimeError("Embedding model not initialized")

            start_time = time.time()
            # Run embedding in thread pool with timeout
            # FIX: Use asyncio.to_thread() instead of deprecated get_event_loop()
            embedding = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: self.embedding_model.encode(query, normalize_embeddings=True)
                ),
                timeout=5.0  # 5 second timeout for embedding
            )

            duration = time.time() - start_time
            EMBEDDING_DURATION.observe(duration)

            return embedding.tolist()

        except asyncio.TimeoutError:
            logger.error(f"Embedding timeout for query: {query[:50]}...")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Embedding generation timeout"
            )
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise

    async def embed_query_with_provenance(self, query: str) -> Tuple[List[float], EmbeddingMetadata]:
        """
        Generate embedding with full provenance metadata

        Args:
            query: Text to embed

        Returns:
            Tuple of (embedding vector, embedding metadata)
        """
        try:
            if self.embedding_model is None:
                raise RuntimeError("Embedding model not initialized")

            start_time = time.time()

            # Generate content checksum
            content_checksum = generate_checksum(query)

            # Run embedding in thread pool with timeout
            embedding = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: self.embedding_model.encode(query, normalize_embeddings=self.normalize_embeddings)
                ),
                timeout=5.0  # 5 second timeout for embedding
            )

            generation_time_ms = (time.time() - start_time) * 1000
            EMBEDDING_DURATION.observe(generation_time_ms / 1000)

            # Get current version info
            version_info = await self.version_tracker.get_current_version_info()

            # Create embedding metadata
            embedding_metadata = EmbeddingMetadata(
                model_name=self.model_name,
                model_version=self.model_version,
                model_checksum=version_info.get('model_checksum') if version_info else None,
                embedding_dim=self.embedding_dim,
                normalization=self.normalize_embeddings,
                pooling_strategy=self.pooling_strategy,
                embedding_created_at=get_utc_timestamp(),
                embedding_generation_time_ms=round(generation_time_ms, 2),
                content_checksum=content_checksum,
                schema_version=PROVENANCE_SCHEMA_VERSION,
                migration_history=[]
            )

            return embedding.tolist(), embedding_metadata

        except asyncio.TimeoutError:
            logger.error(f"Embedding timeout for query: {query[:50]}...")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Embedding generation timeout"
            )
        except Exception as e:
            logger.error(f"Failed to embed query with provenance: {e}")
            raise

    async def dense_retrieval(
        self,
        query_embedding: List[float],
        top_k: int,
        doc_types: Optional[List[str]] = None
    ) -> List[RetrievedDocument]:
        """Dense vector retrieval with timeout"""
        try:
            start_time = time.time()

            # Build filter if doc_types specified - FIX: Use 'should' for OR logic
            query_filter = None
            if doc_types:
                query_filter = Filter(
                    should=[  # Changed from 'must' to 'should' for OR logic
                        FieldCondition(
                            key="doc_type",
                            match=MatchValue(value=doc_type)
                        ) for doc_type in doc_types
                    ]
                )

            # Search with timeout
            timeout = self.config["retrieval"].get("timeout", 10)
            search_result = await asyncio.wait_for(
                self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    query_filter=query_filter
                ),
                timeout=timeout
            )

            # Convert to RetrievedDocument with enhanced provenance
            documents = []
            for hit in search_result:
                # Extract provenance metadata from payload
                embedding_metadata = None
                lineage = None
                trust_indicators = None

                # NEW: Extract embedding_metadata if present
                if "embedding_metadata" in hit.payload:
                    try:
                        embedding_metadata = EmbeddingMetadata.from_dict(hit.payload["embedding_metadata"])
                    except Exception as e:
                        logger.warning(f"Failed to parse embedding_metadata: {e}")

                # NEW: Extract lineage if present
                if "lineage" in hit.payload:
                    try:
                        lineage = DataLineage.from_dict(hit.payload["lineage"])
                    except Exception as e:
                        logger.warning(f"Failed to parse lineage: {e}")

                # NEW: Extract trust_indicators if present
                if "trust_indicators" in hit.payload:
                    try:
                        trust_indicators = TrustIndicators.from_dict(hit.payload["trust_indicators"])
                    except Exception as e:
                        logger.warning(f"Failed to parse trust_indicators: {e}")

                doc = RetrievedDocument(
                    content=hit.payload.get("content", ""),
                    score=hit.score,
                    doc_id=str(hit.id),
                    doc_type=hit.payload.get("doc_type", "unknown"),
                    metadata=hit.payload.get("metadata", {}),
                    source=hit.payload.get("source"),
                    # NEW: Add provenance fields
                    embedding_metadata=embedding_metadata,
                    lineage=lineage,
                    trust_indicators=trust_indicators
                )
                documents.append(doc)

            duration = time.time() - start_time
            RETRIEVAL_DURATION.observe(duration)

            return documents

        except asyncio.TimeoutError:
            logger.error(f"Retrieval timeout after {timeout}s")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Retrieval timeout"
            )
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            raise

    async def rerank_documents(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int
    ) -> List[RetrievedDocument]:
        """Re-rank documents using cross-encoder with timeout"""
        try:
            if self.reranker is None or not documents:
                return documents[:top_k]

            start_time = time.time()

            # Prepare pairs for re-ranking
            pairs = [[query, doc.content] for doc in documents]

            # Run re-ranking in thread pool with timeout
            # FIX: Use asyncio.to_thread() instead of deprecated get_event_loop()
            scores = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: self.reranker.predict(pairs)
                ),
                timeout=5.0  # 5 second timeout for re-ranking
            )

            # Update scores and sort
            for doc, score in zip(documents, scores):
                doc.score = float(score)

            # Sort by new scores and return top_k
            reranked = sorted(documents, key=lambda x: x.score, reverse=True)

            duration = time.time() - start_time
            RERANK_DURATION.observe(duration)

            return reranked[:top_k]

        except asyncio.TimeoutError:
            logger.warning(f"Re-ranking timeout. Returning original results.")
            return documents[:top_k]
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}. Returning original results.")
            return documents[:top_k]

    async def store_document(
        self,
        content: str,
        doc_type: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        lineage: Optional[DataLineage] = None,
        trust_indicators: Optional[TrustIndicators] = None
    ) -> str:
        """
        Store a document with full provenance metadata

        Args:
            content: Document text content
            doc_type: Document type (e.g., "recycling_guideline")
            source: Source identifier
            metadata: Additional metadata
            lineage: Data lineage information (optional, will create default)
            trust_indicators: Trust indicators (optional, will create default)

        Returns:
            Document ID (UUID)
        """
        try:
            if self.qdrant_client is None:
                raise RuntimeError("Qdrant client not initialized")

            # Generate document ID
            doc_id = str(uuid.uuid4())

            # Generate embedding with provenance
            embedding, embedding_metadata = await self.embed_query_with_provenance(content)

            # Create default lineage if not provided
            if lineage is None:
                lineage = DataLineage(
                    original_source=source,
                    collection_method="api",
                    collector_version="1.0.0",
                    processing_pipeline=["ingestion", "embedding"]
                )

            # Create default trust indicators if not provided
            if trust_indicators is None:
                trust_indicators = TrustIndicators(
                    source_reliability=0.8,  # Default reliability
                    content_quality=0.8,     # Default quality
                    freshness_score=1.0      # Fresh document
                )
                trust_indicators.calculate_trust_score()

            # Validate provenance
            if not ProvenanceValidator.validate_embedding_metadata(embedding_metadata):
                logger.warning(f"Invalid embedding metadata for doc {doc_id}")
            if not ProvenanceValidator.validate_lineage(lineage):
                logger.warning(f"Invalid lineage for doc {doc_id}")
            if not ProvenanceValidator.validate_trust_indicators(trust_indicators):
                logger.warning(f"Invalid trust indicators for doc {doc_id}")

            # Prepare payload with full provenance
            payload = {
                "content": content,
                "doc_type": doc_type,
                "source": source,
                "metadata": metadata or {},
                "embedding_metadata": embedding_metadata.to_dict(),
                "lineage": lineage.to_dict(),
                "trust_indicators": trust_indicators.to_dict()
            }

            # Store in Qdrant
            await asyncio.wait_for(
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[{
                        "id": doc_id,
                        "vector": embedding,
                        "payload": payload
                    }]
                ),
                timeout=10.0
            )

            # Increment document count in version tracker
            await self.version_tracker.increment_document_count()

            logger.info(f"Stored document {doc_id} with full provenance")
            return doc_id

        except asyncio.TimeoutError:
            logger.error(f"Timeout storing document")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Document storage timeout"
            )
        except Exception as e:
            logger.error(f"Failed to store document: {e}", exc_info=True)
            raise

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        doc_types: Optional[List[str]] = None,
        rerank: bool = True
    ) -> List[RetrievedDocument]:
        """
        Main retrieval method

        Args:
            query: Search query
            top_k: Number of documents to return
            mode: Retrieval mode (dense, sparse, hybrid)
            doc_types: Filter by document types
            rerank: Apply re-ranking

        Returns:
            List of retrieved documents
        """
        try:
            start_time = datetime.now()

            # Generate query embedding
            query_embedding = await self.embed_query(query)

            # Retrieve based on mode
            if mode == RetrievalMode.DENSE or mode == RetrievalMode.HYBRID:
                dense_top_k = self.config["retrieval"]["dense_top_k"]
                documents = await self.dense_retrieval(
                    query_embedding,
                    dense_top_k,
                    doc_types
                )
            else:
                documents = []

            # Apply re-ranking if enabled
            if rerank and documents:
                documents = await self.rerank_documents(query, documents, top_k)
            else:
                documents = documents[:top_k]

            retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Retrieved {len(documents)} documents in {retrieval_time:.2f}ms")

            return documents

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            raise


# Initialize service
rag_service = RAGService()

# PHASE 2: Mount transparency API router
transparency_router = create_transparency_router(
    rag_service,
    rag_service.audit_manager,
    rag_service.version_tracker
)
app.include_router(transparency_router)
logger.info("Transparency API router mounted at /provenance")


@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    global alert_manager

    logger.info("Starting RAG Service", service="rag_service", version="0.1.0")

    # Initialize distributed tracing
    if TRACING_AVAILABLE:
        try:
            jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
            if jaeger_endpoint:
                init_tracing(
                    service_name="rag_service",
                    service_version="0.1.0",
                    environment=os.getenv("ENVIRONMENT", "development"),
                    jaeger_endpoint=jaeger_endpoint,
                    sample_rate=float(os.getenv("TRACE_SAMPLE_RATE", "0.1"))
                )
                logger.info("Distributed tracing initialized", jaeger_endpoint=jaeger_endpoint)
        except Exception as e:
            logger.warning("Failed to initialize tracing", error=str(e))

    # Initialize error tracking
    if SENTRY_AVAILABLE:
        try:
            sentry_dsn = os.getenv("SENTRY_DSN")
            if sentry_dsn:
                init_sentry(
                    dsn=sentry_dsn,
                    service_name="rag_service",
                    environment=os.getenv("ENVIRONMENT", "development"),
                    release=os.getenv("RELEASE_VERSION", "0.1.0"),
                    traces_sample_rate=float(os.getenv("SENTRY_TRACE_SAMPLE_RATE", "0.1"))
                )
                logger.info("Error tracking initialized", sentry_enabled=True)
        except Exception as e:
            logger.warning("Failed to initialize Sentry", error=str(e))

    # Initialize alerting
    try:
        alert_manager = init_alerting(
            slack_webhook=os.getenv("SLACK_WEBHOOK"),
            pagerduty_key=os.getenv("PAGERDUTY_KEY")
        )
        logger.info("Alerting system initialized")
    except Exception as e:
        logger.warning("Failed to initialize alerting", error=str(e))

    # Initialize RAG service
    try:
        await rag_service.initialize()
        logger.info("RAG service initialized successfully")

        # Add health checks
        health_checker.add_check("qdrant", lambda: check_qdrant_health(rag_service.qdrant_client))
        health_checker.mark_ready()
        health_checker.mark_startup_complete()

        logger.info("Health checks configured")
    except Exception as e:
        logger.error("Failed to initialize RAG service", exc_info=True)
        capture_exception(e, extra={"component": "startup"})

        # Send critical alert
        if alert_manager:
            await send_alert(
                title="RAG Service Startup Failed",
                message=f"Failed to initialize RAG service: {str(e)}",
                severity=AlertSeverity.CRITICAL,
                service="rag_service",
                component="startup"
            )
        raise


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown"""
    logger.info("Shutting down RAG Service")
    health_checker.mark_not_ready()
    await rag_service.close()
    logger.info("RAG Service shutdown complete")


# Health check endpoints
@app.get("/health/live")
async def liveness():
    """Liveness probe - is service alive?"""
    return await health_checker.liveness()


@app.get("/health/ready")
async def readiness():
    """Readiness probe - is service ready for traffic?"""
    return await health_checker.readiness()


@app.get("/health/startup")
async def startup_probe():
    """Startup probe - has service finished initialization?"""
    return await health_checker.startup()


@app.get("/health")
async def health():
    """Detailed health check with all dependencies"""
    result = await health_checker.check_health()

    status_code = 200
    if result.status == HealthStatus.UNHEALTHY:
        status_code = 503
    elif result.status == HealthStatus.DEGRADED:
        status_code = 200

    from starlette.responses import Response as StarletteResponse
    return StarletteResponse(
        content=result.model_dump_json() if hasattr(result, 'model_dump_json') else str(result),
        status_code=status_code,
        media_type="application/json"
    )


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_knowledge(request: RetrievalRequest, http_request: Request):
    """
    Retrieve relevant documents from knowledge base

    Production-optimized with:
    - Rate limiting (100 req/min per IP)
    - Input sanitization
    - Request caching for mobile clients
    - Prometheus metrics
    - Timeout handling
    - Error tracking
    """
    ACTIVE_REQUESTS.inc()
    endpoint = "retrieve"

    try:
        start_time = time.time()

        # CRITICAL: Rate limiting check
        client_ip = http_request.client.host if http_request.client else "unknown"
        if not await rate_limiter.check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            REQUESTS_TOTAL.labels(endpoint=endpoint, status="rate_limited").inc()
            ACTIVE_REQUESTS.dec()
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )

        # CRITICAL: Input sanitization - strip dangerous characters
        sanitized_query = request.query.strip()
        if not sanitized_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )

        # Limit query length for safety
        if len(sanitized_query) > 1000:
            sanitized_query = sanitized_query[:1000]
            logger.warning(f"Query truncated to 1000 chars for IP: {client_ip}")

        # Convert doc_types enum to strings
        doc_types = [dt.value for dt in request.doc_types] if request.doc_types else None

        # Check cache first (important for mobile clients)
        cached_result = await query_cache.get(
            sanitized_query,
            request.top_k,
            request.mode.value,
            doc_types
        )

        if cached_result is not None:
            logger.info(f"Cache hit for query: {sanitized_query[:50]}...")
            REQUESTS_TOTAL.labels(endpoint=endpoint, status="success_cached").inc()
            REQUEST_DURATION.labels(endpoint=endpoint).observe(time.time() - start_time)
            ACTIVE_REQUESTS.dec()
            return cached_result

        # Retrieve documents (use sanitized query)
        documents = await rag_service.retrieve(
            query=sanitized_query,
            top_k=request.top_k,
            mode=request.mode,
            doc_types=doc_types,
            rerank=request.rerank
        )

        # Convert to response format with provenance
        # Check if client wants full provenance (default: True for transparency)
        include_provenance = request.dict().get("include_provenance", True)

        doc_dicts = [doc.to_dict(include_provenance=include_provenance) for doc in documents]

        retrieval_time = (time.time() - start_time) * 1000

        response = RetrievalResponse(
            documents=doc_dicts,
            query=sanitized_query,  # Return sanitized query
            num_results=len(doc_dicts),
            retrieval_time_ms=retrieval_time,
            metadata={
                "mode": request.mode.value,
                "rerank": request.rerank,
                "doc_types": doc_types,
                "cached": False
            }
        )

        # Cache the result (use sanitized query)
        await query_cache.set(
            sanitized_query,
            request.top_k,
            request.mode.value,
            doc_types,
            response
        )

        # PHASE 2: Record audit event for document access
        for doc in documents:
            await rag_service.audit_manager.record_event(
                event_type=EventType.DOCUMENT_ACCESSED.value,
                entity_type=EntityType.DOCUMENT.value,
                entity_id=doc.doc_id,
                action=Action.READ.value,
                actor_type=ActorType.API.value,
                actor_id=client_ip,
                success=True,
                duration_ms=retrieval_time,
                metadata={
                    "query": sanitized_query[:100],  # Truncate for privacy
                    "mode": request.mode.value,
                    "score": doc.score
                }
            )

        REQUESTS_TOTAL.labels(endpoint=endpoint, status="success").inc()
        REQUEST_DURATION.labels(endpoint=endpoint).observe(time.time() - start_time)
        ACTIVE_REQUESTS.dec()

        return response

    except HTTPException:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        ACTIVE_REQUESTS.dec()
        raise
    except Exception as e:
        logger.error(f"Retrieval request failed: {e}", exc_info=True)
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        ACTIVE_REQUESTS.dec()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )


@app.get("/health")
async def health():
    """
    Health check endpoint for load balancer

    Returns detailed health status for monitoring
    """
    is_healthy = (
        rag_service.embedding_model is not None and
        rag_service.qdrant_client is not None and
        not rag_service._shutdown
    )

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "rag",
        "version": "0.1.0",
        "embedding_model_loaded": rag_service.embedding_model is not None,
        "reranker_loaded": rag_service.reranker is not None,
        "qdrant_connected": rag_service.qdrant_client is not None,
        "collection": rag_service.collection_name,
        "cache_size": len(query_cache.cache),
        "shutdown": rag_service._shutdown
    }


@app.get("/stats")
async def get_stats():
    """Get collection and service statistics"""
    try:
        if rag_service.qdrant_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Qdrant not connected"
            )

        collection_info = await rag_service.qdrant_client.get_collection(
            collection_name=rag_service.collection_name
        )

        return {
            "collection_name": rag_service.collection_name,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
            "status": collection_info.status,
            "cache_size": len(query_cache.cache),
            "cache_max_size": query_cache.max_size,
            "cache_ttl_seconds": query_cache.ttl_seconds
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint

    Exposes metrics for monitoring and alerting
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/cache/clear")
async def clear_cache():
    """
    Clear query cache

    Admin endpoint to clear cache when needed
    """
    await query_cache.clear()
    return {"status": "success", "message": "Cache cleared"}


if __name__ == "__main__":
    import uvicorn

    # Production settings
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8003")),
        workers=int(os.getenv("WORKERS", "1")),  # Use 1 worker per service, scale with replicas
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
        # Production optimizations
        limit_concurrency=int(os.getenv("MAX_CONCURRENT", "100")),
        timeout_keep_alive=30,
    )

