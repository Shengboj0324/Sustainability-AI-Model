"""
Organization Search Service - Find charities, recycling centers, and sustainability orgs

This service provides:
- Geospatial search for nearby organizations
- Filtering by organization type, services, materials
- PostgreSQL + PostGIS for location-based queries
- Production-ready with caching, metrics, and async I/O
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import logging
import yaml
from pathlib import Path
from datetime import datetime
import asyncio
import os
import time
import hashlib
import json

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    logging.warning("prometheus_client not installed. Metrics disabled.")
    class DummyMetric:
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    Counter = Histogram = Gauge = lambda *args, **kwargs: DummyMetric()
    generate_latest = lambda: b""
    CONTENT_TYPE_LATEST = "text/plain"

# PostgreSQL async driver
try:
    import asyncpg
except ImportError as e:
    logging.error(f"Missing asyncpg dependency: {e}. Install with: pip install asyncpg")
    raise

# Import monitoring components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.structured_logging import get_logger, log_context, set_correlation_id
from common.health_checks import HealthChecker, check_postgres_health, HealthStatus
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

# Configure structured logging
logger = get_logger(__name__)

# Prometheus metrics
REQUESTS_TOTAL = Counter('org_requests_total', 'Total org search requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('org_request_duration_seconds', 'Request duration', ['endpoint'])
SEARCH_DURATION = Histogram('org_search_duration_seconds', 'Search duration', ['search_type'])
CACHE_HITS = Counter('org_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('org_cache_misses_total', 'Cache misses')
ACTIVE_REQUESTS = Gauge('org_active_requests', 'Active requests')
DB_ERRORS = Counter('org_db_errors_total', 'Database errors', ['error_type'])

# Initialize FastAPI app
app = FastAPI(
    title="ReleAF AI Organization Search Service",
    description="Find charities, recycling centers, and sustainability organizations",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize monitoring components
health_checker = HealthChecker(service_name="org_search_service", check_timeout=5.0)
alert_manager = None  # Initialized in startup
postgres_circuit_breaker = CircuitBreaker(
    name="postgres",
    failure_threshold=5,
    recovery_timeout=30.0,
    expected_exception=Exception
)


class OrgType(str, Enum):
    """Organization types"""
    CHARITY = "charity"
    RECYCLING_CENTER = "recycling_center"
    DONATION_CENTER = "donation_center"
    REPAIR_CAFE = "repair_cafe"
    COMMUNITY_GARDEN = "community_garden"
    EDUCATION = "education"
    ADVOCACY = "advocacy"


class SearchRequest(BaseModel):
    """Organization search request"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    radius_km: float = Field(default=10.0, ge=0.1, le=100, description="Search radius in km")
    org_types: Optional[List[OrgType]] = Field(default=None, description="Filter by org types")
    materials: Optional[List[str]] = Field(default=None, description="Filter by accepted materials")
    limit: int = Field(default=20, ge=1, le=100, description="Max results")


class Organization(BaseModel):
    """Organization model"""
    id: int
    name: str
    org_type: str
    description: Optional[str]
    address: str
    city: str
    state: Optional[str]
    country: str
    postal_code: Optional[str]
    latitude: float
    longitude: float
    distance_km: float
    phone: Optional[str]
    email: Optional[str]
    website: Optional[str]
    accepted_materials: List[str]
    services: List[str]
    hours: Optional[Dict[str, str]]
    verified: bool
    rating: Optional[float]


class SearchResponse(BaseModel):
    """Search response"""
    organizations: List[Organization]
    num_results: int
    search_params: Dict[str, Any]
    query_time_ms: float


class QueryCache:
    """Thread-safe query cache with TTL"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()

    def _make_key(self, params: Dict[str, Any]) -> str:
        """Generate cache key"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()



    async def get(self, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result"""
        async with self._lock:
            key = self._make_key(params)
            if key in self.cache:
                result, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    CACHE_HITS.inc()
                    return result
                else:
                    del self.cache[key]
            CACHE_MISSES.inc()
            return None

    async def set(self, params: Dict[str, Any], result: Any):
        """Cache result"""
        async with self._lock:
            key = self._make_key(params)
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            self.cache[key] = (result, time.time())

    async def clear(self):
        """Clear cache"""
        async with self._lock:
            self.cache.clear()


# Global cache
query_cache = QueryCache(
    max_size=int(os.getenv("ORG_CACHE_SIZE", "1000")),
    ttl_seconds=int(os.getenv("ORG_CACHE_TTL", "300"))
)


class OrgSearchService:
    """
    Production-grade Organization Search Service

    Uses PostgreSQL + PostGIS for geospatial queries
    """

    def __init__(self, config_path: str = "configs/org_search.yaml"):
        """Initialize service"""
        self.config = self._load_config(config_path)
        self.pool: Optional[asyncpg.Pool] = None
        self._shutdown = False

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config not found: {config_path}, using defaults")
                return self._get_default_config()

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded config from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration with environment variables"""
        return {
            "postgres": {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "database": os.getenv("POSTGRES_DB", "releaf"),
                "user": os.getenv("POSTGRES_USER", "releaf_user"),
                "password": os.getenv("POSTGRES_PASSWORD", "releaf_password"),
                "min_pool_size": int(os.getenv("POSTGRES_MIN_POOL", "10")),
                "max_pool_size": int(os.getenv("POSTGRES_MAX_POOL", "20")),
                "command_timeout": int(os.getenv("POSTGRES_TIMEOUT", "30"))
            },
            "search": {
                "default_radius_km": float(os.getenv("DEFAULT_RADIUS_KM", "10.0")),
                "max_radius_km": float(os.getenv("MAX_RADIUS_KM", "100.0")),
                "default_limit": int(os.getenv("DEFAULT_LIMIT", "20"))
            }
        }

    async def initialize(self):
        """Initialize PostgreSQL connection pool"""
        try:
            logger.info("Initializing Organization Search service...")

            pg_config = self.config["postgres"]

            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=pg_config["host"],
                port=pg_config["port"],
                database=pg_config["database"],
                user=pg_config["user"],
                password=pg_config["password"],
                min_size=pg_config["min_pool_size"],
                max_size=pg_config["max_pool_size"],
                command_timeout=pg_config["command_timeout"],
                timeout=30
            )

            # Verify connectivity
            await self.verify_connectivity()

            logger.info("Organization Search service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize service: {e}", exc_info=True)
            DB_ERRORS.labels(error_type="init").inc()
            raise

    async def verify_connectivity(self):
        """Verify database connection"""
        try:
            async with self.pool.acquire() as conn:
                result = await asyncio.wait_for(
                    conn.fetchval("SELECT 1"),
                    timeout=5.0
                )
                if result != 1:
                    raise RuntimeError("Connectivity test failed")
                logger.info("PostgreSQL connectivity verified")
        except Exception as e:
            logger.error(f"Connectivity verification failed: {e}")
            raise

    async def close(self):
        """Graceful shutdown"""
        try:
            self._shutdown = True
            logger.info("Shutting down Organization Search service...")

            if self.pool:
                await self.pool.close()
                logger.info("PostgreSQL pool closed")

            await query_cache.clear()
            logger.info("Cache cleared")

            logger.info("Organization Search service shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    async def search_organizations(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 10.0,
        org_types: Optional[List[str]] = None,
        materials: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for organizations near a location

        Uses PostGIS for geospatial queries
        """
        start_time = time.time()

        try:
            async with self.pool.acquire() as conn:
                # Build query with PostGIS distance calculation
                query = """
                SELECT
                    id, name, org_type, description,
                    address, city, state, country, postal_code,
                    latitude, longitude,
                    ST_Distance(
                        ST_MakePoint($1, $2)::geography,
                        ST_MakePoint(longitude, latitude)::geography
                    ) / 1000.0 AS distance_km,
                    phone, email, website,
                    accepted_materials, services, hours,
                    verified, rating
                FROM organizations
                WHERE ST_DWithin(
                    ST_MakePoint($1, $2)::geography,
                    ST_MakePoint(longitude, latitude)::geography,
                    $3 * 1000
                )
                """

                params = [longitude, latitude, radius_km]
                param_idx = 4

                # Filter by org types
                if org_types:
                    placeholders = ", ".join([f"${i}" for i in range(param_idx, param_idx + len(org_types))])
                    query += f" AND org_type = ANY(ARRAY[{placeholders}])"
                    params.extend(org_types)
                    param_idx += len(org_types)

                # Filter by materials
                if materials:
                    query += f" AND accepted_materials && ${param_idx}"
                    params.append(materials)
                    param_idx += 1

                query += f" ORDER BY distance_km LIMIT ${param_idx}"
                params.append(limit)

                # Execute with timeout
                rows = await asyncio.wait_for(
                    conn.fetch(query, *params),
                    timeout=10.0
                )

                # Convert to dict
                results = []
                for row in rows:
                    results.append({
                        "id": row["id"],
                        "name": row["name"],
                        "org_type": row["org_type"],
                        "description": row["description"],
                        "address": row["address"],
                        "city": row["city"],
                        "state": row["state"],
                        "country": row["country"],
                        "postal_code": row["postal_code"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "distance_km": float(row["distance_km"]),
                        "phone": row["phone"],
                        "email": row["email"],
                        "website": row["website"],
                        "accepted_materials": row["accepted_materials"] or [],
                        "services": row["services"] or [],
                        "hours": row["hours"],
                        "verified": row["verified"],
                        "rating": float(row["rating"]) if row["rating"] else None
                    })

                duration = time.time() - start_time
                SEARCH_DURATION.labels(search_type="geospatial").observe(duration)

                return results

        except asyncio.TimeoutError:
            logger.error("Search query timeout")
            DB_ERRORS.labels(error_type="timeout").inc()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Search timeout"
            )
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            DB_ERRORS.labels(error_type="query_error").inc()
            raise


# Initialize service
org_service = OrgSearchService()


@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    global alert_manager

    logger.info("Starting Org Search Service", service="org_search_service", version="0.1.0")

    # Initialize distributed tracing
    if TRACING_AVAILABLE:
        try:
            jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
            if jaeger_endpoint:
                init_tracing(
                    service_name="org_search_service",
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
                    service_name="org_search_service",
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

    # Initialize Org Search service
    try:
        await org_service.initialize()
        logger.info("Org Search service initialized successfully")

        # Add health checks
        health_checker.add_check("postgres", lambda: check_postgres_health(org_service.pool))
        health_checker.mark_ready()
        health_checker.mark_startup_complete()

        logger.info("Health checks configured")
    except Exception as e:
        logger.error("Failed to initialize Org Search service", exc_info=True)
        capture_exception(e, extra={"component": "startup"})

        # Send critical alert
        if alert_manager:
            await send_alert(
                title="Org Search Service Startup Failed",
                message=f"Failed to initialize Org Search service: {str(e)}",
                severity=AlertSeverity.CRITICAL,
                service="org_search_service",
                component="startup"
            )
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down Org Search Service")
    health_checker.mark_not_ready()
    await org_service.close()
    logger.info("Org Search Service shutdown complete")


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

    return Response(
        content=result.model_dump_json() if hasattr(result, 'model_dump_json') else str(result),
        status_code=status_code,
        media_type="application/json"
    )


@app.post("/search", response_model=SearchResponse)
async def search_organizations_endpoint(request: SearchRequest, http_request: Request):
    """
    Search for organizations near a location

    Uses geospatial queries to find charities, recycling centers, etc.
    """
    endpoint = "search"
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        # Check cache
        cache_params = {
            "lat": request.latitude,
            "lon": request.longitude,
            "radius": request.radius_km,
            "types": request.org_types,
            "materials": request.materials,
            "limit": request.limit
        }

        cached_result = await query_cache.get(cache_params)
        if cached_result is not None:
            logger.info(f"Cache hit for location: ({request.latitude}, {request.longitude})")
            REQUESTS_TOTAL.labels(endpoint=endpoint, status="success_cached").inc()
            ACTIVE_REQUESTS.dec()
            return cached_result

        # Search
        org_types_list = [t.value for t in request.org_types] if request.org_types else None

        organizations = await org_service.search_organizations(
            latitude=request.latitude,
            longitude=request.longitude,
            radius_km=request.radius_km,
            org_types=org_types_list,
            materials=request.materials,
            limit=request.limit
        )

        query_time = (time.time() - start_time) * 1000

        response = SearchResponse(
            organizations=[Organization(**org) for org in organizations],
            num_results=len(organizations),
            search_params={
                "latitude": request.latitude,
                "longitude": request.longitude,
                "radius_km": request.radius_km,
                "org_types": org_types_list,
                "materials": request.materials
            },
            query_time_ms=query_time
        )

        # Cache result
        await query_cache.set(cache_params, response)

        # Metrics
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="success").inc()
        REQUEST_DURATION.labels(endpoint=endpoint).observe(time.time() - start_time)

        return response

    except HTTPException:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        raise
    except Exception as e:
        logger.error(f"Search request failed: {e}", exc_info=True)
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
    finally:
        ACTIVE_REQUESTS.dec()



@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    try:
        if org_service.pool is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not connected"
            )

        async with org_service.pool.acquire() as conn:
            # Count organizations by type
            org_counts = await asyncio.wait_for(
                conn.fetch("""
                    SELECT org_type, COUNT(*) as count
                    FROM organizations
                    GROUP BY org_type
                """),
                timeout=10.0
            )

            # Total count
            total = await asyncio.wait_for(
                conn.fetchval("SELECT COUNT(*) FROM organizations"),
                timeout=10.0
            )

            return {
                "total_organizations": total,
                "by_type": {row["org_type"]: row["count"] for row in org_counts},
                "cache_size": len(query_cache.cache),
                "cache_max_size": query_cache.max_size,
                "cache_ttl_seconds": query_cache.ttl_seconds
            }

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Stats query timeout"
        )
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

    Exposes metrics for monitoring
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/cache/clear")
async def clear_cache():
    """Clear query cache"""
    await query_cache.clear()
    return {"status": "success", "message": "Cache cleared"}


if __name__ == "__main__":
    import uvicorn

    # Production settings
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8005")),
        workers=int(os.getenv("WORKERS", "1")),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
        limit_concurrency=int(os.getenv("MAX_CONCURRENT", "100")),
        timeout_keep_alive=30,
    )

