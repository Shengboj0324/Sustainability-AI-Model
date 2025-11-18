"""
RAG Service - Retrieval-Augmented Generation for sustainability knowledge

Production-optimized for Digital Ocean deployment (Web + iOS backend):
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
from dataclasses import dataclass
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

# Import shared utilities - CRITICAL: Single source of truth
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.utils import QueryCache

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

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """Retrieved document with metadata"""
    content: str
    score: float
    doc_id: str
    doc_type: str
    metadata: Dict[str, Any]
    source: Optional[str] = None


class RetrievalRequest(BaseModel):
    """RAG retrieval request"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of documents to retrieve")
    mode: RetrievalMode = Field(default=RetrievalMode.HYBRID, description="Retrieval mode")
    doc_types: Optional[List[DocumentType]] = Field(default=None, description="Filter by document types")
    location: Optional[Dict[str, float]] = Field(default=None, description="User location for local rules")
    rerank: bool = Field(default=True, description="Apply re-ranking")
    
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
        """Initialize RAG service"""
        if config_path is None:
            config_path = os.getenv("RAG_CONFIG_PATH", "configs/rag.yaml")

        self.config = self._load_config(config_path)
        self.embedding_model: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None
        self.qdrant_client: Optional[AsyncQdrantClient] = None
        self.collection_name = self.config.get("qdrant", {}).get("collection_name", "sustainability_docs")
        self.embedding_dim = self.config.get("embedding", {}).get("dimension", 1024)
        self._shutdown = False
        
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
            device = os.getenv("EMBEDDING_DEVICE", "cpu")  # cpu or cuda

            logger.info(f"Loading embedding model: {model_name} on device: {device}")

            # Check if CUDA is available when requested
            if device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("CUDA requested but not available. Falling back to CPU.")
                        device = "cpu"
                    else:
                        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
                except ImportError:
                    logger.warning("PyTorch not available. Using CPU.")
                    device = "cpu"

            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()

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
                loop.run_in_executor(None, load_model),
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

            loop = asyncio.get_event_loop()

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
                loop.run_in_executor(None, load_reranker),
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

            # Note: SentenceTransformer models don't need explicit cleanup
            # They will be garbage collected

            logger.info("RAG service shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query with timeout"""
        try:
            if self.embedding_model is None:
                raise RuntimeError("Embedding model not initialized")

            start_time = time.time()

            # Run embedding in thread pool with timeout
            loop = asyncio.get_event_loop()
            embedding = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
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

            # Convert to RetrievedDocument
            documents = []
            for hit in search_result:
                doc = RetrievedDocument(
                    content=hit.payload.get("content", ""),
                    score=hit.score,
                    doc_id=str(hit.id),
                    doc_type=hit.payload.get("doc_type", "unknown"),
                    metadata=hit.payload.get("metadata", {}),
                    source=hit.payload.get("source")
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
            loop = asyncio.get_event_loop()
            scores = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
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


@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    await rag_service.initialize()


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown"""
    await rag_service.close()


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

        # Convert to response format
        doc_dicts = [
            {
                "content": doc.content,
                "score": doc.score,
                "doc_id": doc.doc_id,
                "doc_type": doc.doc_type,
                "metadata": doc.metadata,
                "source": doc.source
            }
            for doc in documents
        ]

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

