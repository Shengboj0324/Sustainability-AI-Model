"""
Knowledge Graph Service - Material relationships and upcycling paths

This service provides:
- Material property queries
- Upcycling path discovery
- Relationship traversal
- Graph-based reasoning
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
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.input_validation import InputValidator
from common.structured_logging import get_logger, log_context, set_correlation_id
from common.health_checks import HealthChecker, check_neo4j_health, HealthStatus
from common.alerting import init_alerting, send_alert, AlertSeverity
from common.circuit_breaker import CircuitBreaker

# Try to import optional monitoring components
try:
    from common.tracing import init_tracing, trace_operation, get_tracer
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    logger.warning("Tracing not available. Install OpenTelemetry for distributed tracing.")
    def trace_operation(name, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from common.error_tracking import init_sentry, capture_exception, add_breadcrumb
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    logger.warning("Sentry not available. Install sentry-sdk for error tracking.")
    def capture_exception(exc, **kwargs):
        pass
    def add_breadcrumb(msg, **kwargs):
        pass

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    logging.warning("prometheus_client not installed. Metrics disabled.")
    # Create dummy metrics
    class DummyMetric:
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    Counter = Histogram = Gauge = lambda *args, **kwargs: DummyMetric()
    generate_latest = lambda: b""
    CONTENT_TYPE_LATEST = "text/plain"

# Neo4j imports
try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError as e:
    logging.error(f"Missing neo4j dependency: {e}. Install with: pip install neo4j")
    raise

# Configure structured logging
logger = get_logger(__name__)

# Prometheus metrics
REQUESTS_TOTAL = Counter('kg_requests_total', 'Total KG requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('kg_request_duration_seconds', 'Request duration', ['endpoint'])
QUERY_DURATION = Histogram('kg_query_duration_seconds', 'Neo4j query duration', ['query_type'])
CACHE_HITS = Counter('kg_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('kg_cache_misses_total', 'Cache misses')
ACTIVE_REQUESTS = Gauge('kg_active_requests', 'Active requests')
NEO4J_ERRORS = Counter('kg_neo4j_errors_total', 'Neo4j errors', ['error_type'])

# Initialize FastAPI app
app = FastAPI(
    title="ReleAF AI Knowledge Graph Service",
    description="Material relationships and upcycling knowledge graph",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web and iOS clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize monitoring components
health_checker = HealthChecker(service_name="kg_service", check_timeout=5.0)
alert_manager = None  # Initialized in startup
neo4j_circuit_breaker = CircuitBreaker(
    name="neo4j",
    failure_threshold=5,
    recovery_timeout=30.0,
    expected_exception=Exception
)


class QueryType(str, Enum):
    """Knowledge graph query types"""
    MATERIAL_PROPERTIES = "material_properties"
    UPCYCLING_PATHS = "upcycling_paths"
    SIMILAR_MATERIALS = "similar_materials"
    RECYCLING_PROCESS = "recycling_process"
    COMPATIBILITY = "compatibility"


class MaterialQuery(BaseModel):
    """Material property query"""
    material_name: str = Field(..., min_length=1, max_length=100)
    include_properties: bool = Field(default=True)
    include_relationships: bool = Field(default=True)


class UpcyclingPathQuery(BaseModel):
    """Upcycling path discovery query"""
    source_material: str = Field(..., min_length=1, max_length=100)
    target_product: Optional[str] = Field(default=None, max_length=100)
    max_depth: int = Field(default=3, ge=1, le=5)
    # CRITICAL FIX: Pydantic v2 uses 'pattern' instead of 'regex'
    difficulty_level: Optional[str] = Field(default=None, pattern="^(easy|medium|hard)$")


class RelationshipQuery(BaseModel):
    """General relationship query"""
    entity: str = Field(..., min_length=1)
    relationship_type: Optional[str] = None
    max_hops: int = Field(default=2, ge=1, le=4)


class SimilarMaterialsQuery(BaseModel):
    """Similar materials query"""
    material_name: str = Field(..., min_length=1, max_length=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=50)


class RecyclingProcessQuery(BaseModel):
    """Recycling process query"""
    material_name: str = Field(..., min_length=1, max_length=100)
    include_steps: bool = Field(default=True)
    include_facilities: bool = Field(default=True)


class CompatibilityQuery(BaseModel):
    """Material compatibility query"""
    material1: str = Field(..., min_length=1, max_length=100)
    material2: str = Field(..., min_length=1, max_length=100)
    context: Optional[str] = Field(default=None, max_length=200)


class KGResponse(BaseModel):
    """Knowledge graph response"""
    results: List[Dict[str, Any]]
    query_type: str
    num_results: int
    query_time_ms: float
    metadata: Dict[str, Any]


class QueryCache:
    """Thread-safe query cache with TTL for KG queries"""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 600):
        """
        Initialize cache

        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries (default: 10 minutes)
        """
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()

    def _make_key(self, query_type: str, params: Dict[str, Any]) -> str:
        """Generate cache key from query type and parameters"""
        # Sort params for consistent hashing
        param_str = str(sorted(params.items()))
        key_str = f"{query_type}:{param_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def get(self, query_type: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if exists and not expired"""
        async with self._lock:
            key = self._make_key(query_type, params)
            if key in self.cache:
                result, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    CACHE_HITS.inc()
                    logger.debug(f"Cache hit for {query_type}")
                    return result
                else:
                    # Expired
                    del self.cache[key]
            CACHE_MISSES.inc()
            return None

    async def set(self, query_type: str, params: Dict[str, Any], result: Any):
        """Cache query result with TTL"""
        async with self._lock:
            key = self._make_key(query_type, params)

            # LRU eviction if cache is full
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
                logger.debug(f"Cache eviction: {oldest_key}")

            self.cache[key] = (result, time.time())
            logger.debug(f"Cached result for {query_type}")

    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
            logger.info("Cache cleared")


# Global cache instance
query_cache = QueryCache(
    max_size=int(os.getenv("KG_CACHE_SIZE", "500")),
    ttl_seconds=int(os.getenv("KG_CACHE_TTL", "600"))
)


class KnowledgeGraphService:
    """
    Production-grade Knowledge Graph service

    Manages connections to Neo4j and provides high-level
    query interfaces for material knowledge and upcycling paths.
    """

    def __init__(self, config_path: str = "configs/gnn.yaml"):
        """Initialize KG service"""
        self.config = self._load_config(config_path)
        self.driver: Optional[AsyncDriver] = None
        self.database = self.config.get("neo4j", {}).get("database", "neo4j")
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
            "neo4j": {
                "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "user": os.getenv("NEO4J_USER", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "releaf_password"),
                "database": os.getenv("NEO4J_DATABASE", "neo4j"),
                "max_connection_pool_size": int(os.getenv("NEO4J_POOL_SIZE", "50")),
                "max_connection_lifetime": int(os.getenv("NEO4J_CONN_LIFETIME", "3600")),
                "connection_timeout": int(os.getenv("NEO4J_CONN_TIMEOUT", "60"))
            },
            "query": {
                "timeout": int(os.getenv("NEO4J_QUERY_TIMEOUT", "30")),
                "max_results": int(os.getenv("NEO4J_MAX_RESULTS", "100"))
            }
        }
    
    async def initialize(self):
        """Initialize Neo4j connection with production settings"""
        try:
            logger.info("Initializing Knowledge Graph service...")

            neo4j_config = self.config["neo4j"]
            uri = neo4j_config.get("uri", "bolt://localhost:7687")
            user = neo4j_config.get("user", "neo4j")
            password = neo4j_config.get("password", "releaf_password")
            max_pool_size = neo4j_config.get("max_connection_pool_size", 50)
            max_lifetime = neo4j_config.get("max_connection_lifetime", 3600)
            conn_timeout = neo4j_config.get("connection_timeout", 60)

            logger.info(f"Connecting to Neo4j at {uri} (pool_size={max_pool_size})")

            # Production-grade connection with pooling
            self.driver = AsyncGraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_lifetime=max_lifetime,
                max_connection_pool_size=max_pool_size,
                connection_acquisition_timeout=conn_timeout,
                keep_alive=True,
                max_transaction_retry_time=30
            )

            # Verify connectivity with timeout
            await asyncio.wait_for(
                self.verify_connectivity(),
                timeout=10.0
            )

            logger.info("Knowledge Graph service initialized successfully")

        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            NEO4J_ERRORS.labels(error_type="auth").inc()
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            NEO4J_ERRORS.labels(error_type="unavailable").inc()
            raise
        except asyncio.TimeoutError:
            logger.error("Neo4j connection timeout")
            NEO4J_ERRORS.labels(error_type="timeout").inc()
            raise
        except Exception as e:
            logger.error(f"Failed to initialize KG service: {e}", exc_info=True)
            NEO4J_ERRORS.labels(error_type="unknown").inc()
            raise

    async def verify_connectivity(self):
        """Verify Neo4j connection"""
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 AS num")
                record = await result.single()
                if record["num"] != 1:
                    raise RuntimeError("Connectivity test failed")
                logger.info("Neo4j connectivity verified")
        except Exception as e:
            logger.error(f"Connectivity verification failed: {e}")
            raise

    async def close(self):
        """Graceful shutdown - close Neo4j connection and cleanup"""
        try:
            self._shutdown = True
            logger.info("Shutting down Knowledge Graph service...")

            if self.driver:
                await self.driver.close()
                logger.info("Neo4j connection closed")

            await query_cache.clear()
            logger.info("Cache cleared")

            logger.info("Knowledge Graph service shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    async def query_material_properties(
        self,
        material_name: str,
        include_properties: bool = True,
        include_relationships: bool = True
    ) -> Dict[str, Any]:
        """
        Query material properties and relationships with caching

        Args:
            material_name: Name of the material
            include_properties: Include material properties
            include_relationships: Include related materials

        Returns:
            Material data with properties and relationships
        """
        start_time = time.time()

        # Validate and sanitize input
        try:
            material_name = InputValidator.validate_material_name(material_name)
        except ValueError as e:
            logger.warning(f"Invalid material name: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

        # Check cache first
        cache_params = {
            "material_name": material_name,
            "include_properties": include_properties,
            "include_relationships": include_relationships
        }
        cached_result = await query_cache.get("material_properties", cache_params)
        if cached_result is not None:
            logger.info(f"Cache hit for material: {material_name}")
            return cached_result

        try:
            # Use circuit breaker for Neo4j queries
            async def execute_neo4j_query():
                async with self.driver.session(database=self.database) as session:
                    # Build Cypher query
                    query = """
                    MATCH (m:Material {name: $material_name})
                    OPTIONAL MATCH (m)-[r]->(related)
                    RETURN m, collect({type: type(r), node: related}) as relationships
                    """

                    # Execute with timeout
                    timeout = self.config.get("query", {}).get("timeout", 30)
                    result = await asyncio.wait_for(
                        session.run(query, material_name=material_name),
                        timeout=timeout
                    )
                    return await result.single()

            record = await neo4j_circuit_breaker.call(execute_neo4j_query)

            if not record:
                response = {
                    "material": material_name,
                    "found": False,
                    "properties": {},
                    "relationships": []
                }
            else:
                material_node = record["m"]
                relationships = record["relationships"]

                # Extract properties
                properties = dict(material_node) if include_properties else {}

                # Extract relationships
                rel_list = []
                if include_relationships:
                    for rel in relationships:
                        if rel["node"]:
                            rel_list.append({
                                "type": rel["type"],
                                "target": dict(rel["node"])
                            })

                response = {
                    "material": material_name,
                    "found": True,
                    "properties": properties,
                    "relationships": rel_list
                }

            # Cache the result
            await query_cache.set("material_properties", cache_params, response)

            # Record metrics
            duration = time.time() - start_time
            QUERY_DURATION.labels(query_type="material_properties").observe(duration)

            return response

        except asyncio.TimeoutError:
            logger.error(f"Material query timeout for: {material_name}")
            NEO4J_ERRORS.labels(error_type="timeout").inc()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Query timeout"
            )
        except Exception as e:
            logger.error(f"Material query failed: {e}", exc_info=True)
            NEO4J_ERRORS.labels(error_type="query_error").inc()
            raise

    async def find_upcycling_paths(
        self,
        source_material: str,
        target_product: Optional[str] = None,
        max_depth: int = 3,
        difficulty_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # Validate and sanitize inputs
        try:
            source_material = InputValidator.validate_material_name(source_material)
            if target_product:
                target_product = InputValidator.validate_material_name(target_product)
            max_depth = int(InputValidator.validate_numeric_range(max_depth, 1, 5, "max_depth"))
            if difficulty_level and difficulty_level not in ("easy", "medium", "hard"):
                raise ValueError("difficulty_level must be 'easy', 'medium', or 'hard'")
        except ValueError as e:
            logger.warning(f"Invalid upcycling path parameters: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

        try:
            async with self.driver.session(database=self.database) as session:
                # Build Cypher query based on parameters
                if target_product:
                    # Find specific path
                    query = """
                    MATCH path = (m:Material {name: $source})-[:CAN_BECOME*1..%d]->(p:Product {name: $target})
                    WHERE ALL(r IN relationships(path) WHERE
                        CASE WHEN $difficulty IS NOT NULL
                        THEN r.difficulty = $difficulty
                        ELSE true END
                    )
                    RETURN path,
                           [r IN relationships(path) | r.difficulty] as difficulties,
                           [r IN relationships(path) | r.tools_required] as tools,
                           length(path) as steps
                    LIMIT 10
                    """ % max_depth

                    result = await session.run(
                        query,
                        source=source_material,
                        target=target_product,
                        difficulty=difficulty_level
                    )
                else:
                    # Find all possible products
                    query = """
                    MATCH path = (m:Material {name: $source})-[:CAN_BECOME*1..%d]->(p:Product)
                    WHERE ALL(r IN relationships(path) WHERE
                        CASE WHEN $difficulty IS NOT NULL
                        THEN r.difficulty = $difficulty
                        ELSE true END
                    )
                    RETURN path,
                           p.name as product,
                           [r IN relationships(path) | r.difficulty] as difficulties,
                           [r IN relationships(path) | r.tools_required] as tools,
                           length(path) as steps
                    ORDER BY steps ASC
                    LIMIT 20
                    """ % max_depth

                    result = await session.run(
                        query,
                        source=source_material,
                        difficulty=difficulty_level
                    )

                # Process results
                paths = []
                async for record in result:
                    path_data = {
                        "source": source_material,
                        "target": target_product or record.get("product"),
                        "steps": record["steps"],
                        "difficulties": record["difficulties"],
                        "tools_required": record["tools"],
                        "path_nodes": self._extract_path_nodes(record["path"])
                    }
                    paths.append(path_data)

                return paths

        except Exception as e:
            logger.error(f"Upcycling path query failed: {e}", exc_info=True)
            raise

    def _extract_path_nodes(self, path) -> List[Dict[str, Any]]:
        """Extract nodes from Neo4j path object"""
        try:
            nodes = []
            for node in path.nodes:
                nodes.append({
                    "id": node.id,
                    "labels": list(node.labels),
                    "properties": dict(node)
                })
            return nodes
        except Exception as e:
            logger.warning(f"Failed to extract path nodes: {e}")
            return []

    async def query_relationships(
        self,
        entity: str,
        relationship_type: Optional[str] = None,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Query general relationships from an entity

        Args:
            entity: Starting entity name
            relationship_type: Optional relationship type filter
            max_hops: Maximum number of hops

        Returns:
            List of related entities with relationship info
        """
        try:
            async with self.driver.session(database=self.database) as session:
                # Build query
                if relationship_type:
                    query = f"""
                    MATCH path = (start)-[r:{relationship_type}*1..{max_hops}]->(end)
                    WHERE start.name = $entity
                    RETURN path, length(path) as hops
                    LIMIT 50
                    """
                else:
                    query = f"""
                    MATCH path = (start)-[*1..{max_hops}]->(end)
                    WHERE start.name = $entity
                    RETURN path, length(path) as hops
                    LIMIT 50
                    """

                result = await session.run(query, entity=entity)

                relationships = []
                async for record in result:
                    relationships.append({
                        "hops": record["hops"],
                        "path": self._extract_path_nodes(record["path"])
                    })

                return relationships

        except Exception as e:
            logger.error(f"Relationship query failed: {e}", exc_info=True)
            raise

    async def find_similar_materials(
        self,
        material_name: str,
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find materials similar to the given material

        Args:
            material_name: Reference material name
            similarity_threshold: Minimum similarity score (0-1)
            max_results: Maximum number of results

        Returns:
            List of similar materials with similarity scores
        """
        start_time = time.time()

        # Validate and sanitize inputs
        try:
            material_name = InputValidator.validate_material_name(material_name)
            similarity_threshold = InputValidator.validate_numeric_range(
                similarity_threshold, 0.0, 1.0, "similarity_threshold"
            )
            max_results = int(InputValidator.validate_numeric_range(
                max_results, 1, 50, "max_results"
            ))
        except ValueError as e:
            logger.warning(f"Invalid similar materials parameters: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

        # Check cache first
        cache_params = {
            "material_name": material_name,
            "similarity_threshold": similarity_threshold,
            "max_results": max_results
        }
        cached_result = await query_cache.get("similar_materials", cache_params)
        if cached_result is not None:
            return cached_result

        try:
            async with self.driver.session(database=self.database) as session:
                # Find similar materials based on shared properties and relationships
                query = """
                MATCH (m1:Material {name: $material_name})
                MATCH (m2:Material)
                WHERE m1 <> m2

                // Calculate similarity based on shared properties
                WITH m1, m2,
                     // Shared recycling category
                     CASE WHEN m1.recycling_category = m2.recycling_category THEN 0.3 ELSE 0.0 END +
                     // Shared material type
                     CASE WHEN m1.material_type = m2.material_type THEN 0.3 ELSE 0.0 END +
                     // Shared environmental impact level
                     CASE WHEN m1.environmental_impact = m2.environmental_impact THEN 0.2 ELSE 0.0 END +
                     // Shared biodegradability
                     CASE WHEN m1.biodegradable = m2.biodegradable THEN 0.2 ELSE 0.0 END
                     AS similarity_score

                WHERE similarity_score >= $threshold

                RETURN m2.name as material_name,
                       m2 as properties,
                       similarity_score
                ORDER BY similarity_score DESC
                LIMIT $max_results
                """

                timeout = self.config.get("query", {}).get("timeout", 30)
                result = await asyncio.wait_for(
                    session.run(
                        query,
                        material_name=material_name,
                        threshold=similarity_threshold,
                        max_results=max_results
                    ),
                    timeout=timeout
                )

                similar_materials = []
                async for record in result:
                    similar_materials.append({
                        "material_name": record["material_name"],
                        "similarity_score": record["similarity_score"],
                        "properties": dict(record["properties"])
                    })

                # Cache the result
                await query_cache.set("similar_materials", cache_params, similar_materials)

                # Record metrics
                duration = time.time() - start_time
                QUERY_DURATION.labels(query_type="similar_materials").observe(duration)

                return similar_materials

        except asyncio.TimeoutError:
            logger.error(f"Similar materials query timeout for: {material_name}")
            NEO4J_ERRORS.labels(error_type="timeout").inc()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Query timeout"
            )
        except Exception as e:
            logger.error(f"Similar materials query failed: {e}", exc_info=True)
            NEO4J_ERRORS.labels(error_type="query_error").inc()
            raise

    async def query_recycling_process(
        self,
        material_name: str,
        include_steps: bool = True,
        include_facilities: bool = True
    ) -> Dict[str, Any]:
        """
        Query recycling process for a material

        Args:
            material_name: Material to query
            include_steps: Include detailed recycling steps
            include_facilities: Include facility requirements

        Returns:
            Recycling process information
        """
        start_time = time.time()

        # Validate and sanitize inputs
        try:
            material_name = InputValidator.validate_material_name(material_name)
        except ValueError as e:
            logger.warning(f"Invalid recycling process parameters: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

        # Check cache first
        cache_params = {
            "material_name": material_name,
            "include_steps": include_steps,
            "include_facilities": include_facilities
        }
        cached_result = await query_cache.get("recycling_process", cache_params)
        if cached_result is not None:
            return cached_result

        try:
            async with self.driver.session(database=self.database) as session:
                # Query recycling process
                query = """
                MATCH (m:Material {name: $material_name})
                OPTIONAL MATCH (m)-[:RECYCLED_BY]->(process:Process)
                OPTIONAL MATCH (process)-[:REQUIRES]->(facility:Facility)
                OPTIONAL MATCH (process)-[:HAS_STEP]->(step:ProcessStep)

                RETURN m,
                       collect(DISTINCT process) as processes,
                       collect(DISTINCT facility) as facilities,
                       collect(DISTINCT step) as steps
                """

                timeout = self.config.get("query", {}).get("timeout", 30)
                result = await asyncio.wait_for(
                    session.run(query, material_name=material_name),
                    timeout=timeout
                )
                record = await result.single()

                if not record:
                    response = {
                        "material": material_name,
                        "found": False,
                        "recyclable": False,
                        "processes": [],
                        "facilities": [],
                        "steps": []
                    }
                else:
                    material_node = record["m"]
                    processes = [dict(p) for p in record["processes"] if p]
                    facilities = [dict(f) for f in record["facilities"] if f] if include_facilities else []
                    steps = [dict(s) for s in record["steps"] if s] if include_steps else []

                    response = {
                        "material": material_name,
                        "found": True,
                        "recyclable": material_node.get("recyclable", False),
                        "recycling_category": material_node.get("recycling_category", "unknown"),
                        "processes": processes,
                        "facilities": facilities,
                        "steps": sorted(steps, key=lambda x: x.get("order", 0)) if steps else []
                    }

                # Cache the result
                await query_cache.set("recycling_process", cache_params, response)

                # Record metrics
                duration = time.time() - start_time
                QUERY_DURATION.labels(query_type="recycling_process").observe(duration)

                return response

        except asyncio.TimeoutError:
            logger.error(f"Recycling process query timeout for: {material_name}")
            NEO4J_ERRORS.labels(error_type="timeout").inc()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Query timeout"
            )
        except Exception as e:
            logger.error(f"Recycling process query failed: {e}", exc_info=True)
            NEO4J_ERRORS.labels(error_type="query_error").inc()
            raise

    async def check_material_compatibility(
        self,
        material1: str,
        material2: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if two materials are compatible for upcycling/combination

        Args:
            material1: First material name
            material2: Second material name
            context: Optional context (e.g., "adhesive", "structural", "decorative")

        Returns:
            Compatibility information with score and reasons
        """
        start_time = time.time()

        # Validate and sanitize inputs
        try:
            material1 = InputValidator.validate_material_name(material1)
            material2 = InputValidator.validate_material_name(material2)
            if context:
                context = InputValidator.sanitize_string(context, max_length=200)
                context = InputValidator.validate_no_injection(context, input_type="general")
        except ValueError as e:
            logger.warning(f"Invalid compatibility check parameters: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

        # Check cache first
        cache_params = {
            "material1": material1,
            "material2": material2,
            "context": context
        }
        cached_result = await query_cache.get("compatibility", cache_params)
        if cached_result is not None:
            return cached_result

        try:
            async with self.driver.session(database=self.database) as session:
                # Query material compatibility
                query = """
                MATCH (m1:Material {name: $material1})
                MATCH (m2:Material {name: $material2})

                // Check for explicit compatibility relationships
                OPTIONAL MATCH (m1)-[r:COMPATIBLE_WITH]->(m2)
                OPTIONAL MATCH (m1)-[r2:INCOMPATIBLE_WITH]->(m2)

                // Calculate compatibility score based on properties
                WITH m1, m2, r, r2,
                     // Same recycling category is good
                     CASE WHEN m1.recycling_category = m2.recycling_category THEN 0.2 ELSE 0.0 END +
                     // Similar melting points (within 50°C)
                     CASE WHEN abs(coalesce(m1.melting_point, 0) - coalesce(m2.melting_point, 0)) < 50 THEN 0.2 ELSE 0.0 END +
                     // Both biodegradable or both not
                     CASE WHEN m1.biodegradable = m2.biodegradable THEN 0.1 ELSE 0.0 END +
                     // Similar density (within 0.5 g/cm³)
                     CASE WHEN abs(coalesce(m1.density, 0) - coalesce(m2.density, 0)) < 0.5 THEN 0.1 ELSE 0.0 END +
                     // Explicit compatibility bonus
                     CASE WHEN r IS NOT NULL THEN 0.4 ELSE 0.0 END +
                     // Explicit incompatibility penalty
                     CASE WHEN r2 IS NOT NULL THEN -1.0 ELSE 0.0 END
                     AS compatibility_score

                RETURN m1, m2,
                       compatibility_score,
                       r IS NOT NULL as explicitly_compatible,
                       r2 IS NOT NULL as explicitly_incompatible,
                       r.reason as compatibility_reason,
                       r2.reason as incompatibility_reason
                """

                timeout = self.config.get("query", {}).get("timeout", 30)
                result = await asyncio.wait_for(
                    session.run(
                        query,
                        material1=material1,
                        material2=material2
                    ),
                    timeout=timeout
                )
                record = await result.single()

                if not record:
                    response = {
                        "material1": material1,
                        "material2": material2,
                        "found": False,
                        "compatible": False,
                        "compatibility_score": 0.0,
                        "reasons": ["One or both materials not found in knowledge graph"]
                    }
                else:
                    score = record["compatibility_score"]
                    explicitly_compatible = record["explicitly_compatible"]
                    explicitly_incompatible = record["explicitly_incompatible"]

                    # Determine compatibility
                    compatible = score >= 0.5 or explicitly_compatible
                    if explicitly_incompatible:
                        compatible = False

                    # Build reasons list
                    reasons = []
                    if explicitly_compatible:
                        reasons.append(f"Explicitly marked as compatible: {record['compatibility_reason']}")
                    if explicitly_incompatible:
                        reasons.append(f"Explicitly marked as incompatible: {record['incompatibility_reason']}")

                    m1 = record["m1"]
                    m2 = record["m2"]

                    if m1.get("recycling_category") == m2.get("recycling_category"):
                        reasons.append(f"Same recycling category: {m1.get('recycling_category')}")

                    if m1.get("biodegradable") == m2.get("biodegradable"):
                        reasons.append(f"Both {'biodegradable' if m1.get('biodegradable') else 'non-biodegradable'}")

                    if context:
                        reasons.append(f"Context: {context}")

                    response = {
                        "material1": material1,
                        "material2": material2,
                        "found": True,
                        "compatible": compatible,
                        "compatibility_score": round(max(0.0, min(1.0, score)), 2),
                        "confidence": "high" if explicitly_compatible or explicitly_incompatible else "medium",
                        "reasons": reasons,
                        "context": context
                    }

                # Cache the result
                await query_cache.set("compatibility", cache_params, response)

                # Record metrics
                duration = time.time() - start_time
                QUERY_DURATION.labels(query_type="compatibility").observe(duration)

                return response

        except asyncio.TimeoutError:
            logger.error(f"Compatibility query timeout for: {material1}, {material2}")
            NEO4J_ERRORS.labels(error_type="timeout").inc()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Query timeout"
            )
        except Exception as e:
            logger.error(f"Compatibility query failed: {e}", exc_info=True)
            NEO4J_ERRORS.labels(error_type="query_error").inc()
            raise


# Initialize service
kg_service = KnowledgeGraphService()


@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    global alert_manager

    logger.info("Starting KG Service", service="kg_service", version="0.1.0")

    # Initialize distributed tracing
    if TRACING_AVAILABLE:
        try:
            jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
            if jaeger_endpoint:
                init_tracing(
                    service_name="kg_service",
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
                    service_name="kg_service",
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
            pagerduty_key=os.getenv("PAGERDUTY_KEY"),
            smtp_host=os.getenv("SMTP_HOST"),
            smtp_user=os.getenv("SMTP_USER"),
            smtp_password=os.getenv("SMTP_PASSWORD")
        )
        logger.info("Alerting system initialized")
    except Exception as e:
        logger.warning("Failed to initialize alerting", error=str(e))

    # CRITICAL FIX: Initialize KG service with retry logic
    max_retries = int(os.getenv("STARTUP_MAX_RETRIES", "3"))
    retry_delay = float(os.getenv("STARTUP_RETRY_DELAY", "2.0"))

    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing KG service (attempt {attempt + 1}/{max_retries})")
            await kg_service.initialize()
            logger.info("KG service initialized successfully")

            # Add health checks
            health_checker.add_check("neo4j", lambda: check_neo4j_health(kg_service.driver))
            health_checker.mark_ready()
            health_checker.mark_startup_complete()

            logger.info("Health checks configured")
            break  # Success, exit retry loop

        except Exception as e:
            logger.error(
                f"Failed to initialize KG service (attempt {attempt + 1}/{max_retries})",
                exc_info=True,
                attempt=attempt + 1,
                max_retries=max_retries
            )
            capture_exception(e, extra={"component": "startup", "attempt": attempt + 1})

            if attempt < max_retries - 1:
                # Not the last attempt, wait and retry
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # Last attempt failed, start in degraded mode
                logger.error("All initialization attempts failed, starting in degraded mode")
                health_checker.mark_not_ready()

                # Send critical alert
                if alert_manager:
                    await send_alert(
                        title="KG Service Startup Failed",
                        message=f"Failed to initialize KG service after {max_retries} attempts: {str(e)}",
                        severity=AlertSeverity.CRITICAL,
                        service="kg_service",
                        component="startup"
                    )

                # Don't raise - allow service to start in degraded mode
                # Circuit breaker will protect endpoints
                logger.warning("Service started in degraded mode - circuit breaker will protect endpoints")


@app.on_event("shutdown")
async def shutdown():
    """Close connections on shutdown"""
    logger.info("Shutting down KG Service")
    health_checker.mark_not_ready()
    await kg_service.close()
    logger.info("KG Service shutdown complete")


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
        status_code = 200  # Still accepting traffic

    return Response(
        content=result.model_dump_json() if hasattr(result, 'model_dump_json') else str(result),
        status_code=status_code,
        media_type="application/json"
    )


@app.post("/material/properties", response_model=KGResponse)
@trace_operation("kg_material_properties")
async def get_material_properties(query: MaterialQuery, http_request: Request):
    """
    Get material properties and relationships

    Returns comprehensive information about a material including
    its properties, recycling process, and related materials.
    """
    endpoint = "material_properties"
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    # Set correlation ID from request header or generate new one
    correlation_id = http_request.headers.get("X-Correlation-ID")
    set_correlation_id(correlation_id)

    try:
        with log_context(
            correlation_id=correlation_id,
            endpoint=endpoint,
            material=query.material_name
        ):
            logger.info(
                "Material properties query started",
                material=query.material_name,
                include_properties=query.include_properties,
                include_relationships=query.include_relationships
            )

            add_breadcrumb(
                "Material properties query",
                category="kg",
                data={"material": query.material_name}
            )

            result = await kg_service.query_material_properties(
                material_name=query.material_name,
                include_properties=query.include_properties,
                include_relationships=query.include_relationships
            )

            query_time = (time.time() - start_time) * 1000

            response = KGResponse(
                results=[result],
                query_type="material_properties",
                num_results=1 if result["found"] else 0,
                query_time_ms=query_time,
                metadata={
                    "material": query.material_name,
                    "found": result["found"]
                }
            )

            # Record metrics
            REQUESTS_TOTAL.labels(endpoint=endpoint, status="success").inc()
            REQUEST_DURATION.labels(endpoint=endpoint).observe(time.time() - start_time)

            logger.info(
                "Material properties query completed",
                found=result["found"],
                query_time_ms=query_time
            )

            return response

    except HTTPException:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        logger.warning("Material properties query failed - HTTP exception", exc_info=True)
        raise
    except Exception as e:
        logger.error("Material properties query failed", exc_info=True, material=query.material_name)
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()

        # Capture exception in Sentry
        capture_exception(
            e,
            extra={"material": query.material_name, "endpoint": endpoint},
            tags={"service": "kg_service", "endpoint": endpoint}
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/upcycling/paths", response_model=KGResponse)
async def find_upcycling_paths_endpoint(query: UpcyclingPathQuery, http_request: Request):
    """
    Find upcycling paths from material to products

    Discovers creative ways to transform waste materials into
    useful products, with step-by-step guidance.
    """
    # CRITICAL FIX: Add correlation ID propagation for distributed tracing
    correlation_id = http_request.headers.get("X-Correlation-ID")
    set_correlation_id(correlation_id)

    # CRITICAL FIX: Add request timeout wrapper to prevent hanging
    async def _handle_request():
        try:
            start_time = datetime.now()

            paths = await kg_service.find_upcycling_paths(
                source_material=query.source_material,
                target_product=query.target_product,
                max_depth=query.max_depth,
                difficulty_level=query.difficulty_level
            )

            query_time = (datetime.now() - start_time).total_seconds() * 1000

            return KGResponse(
                results=paths,
                query_type="upcycling_paths",
                num_results=len(paths),
                query_time_ms=query_time,
                metadata={
                    "source": query.source_material,
                    "target": query.target_product,
                    "max_depth": query.max_depth,
                    "difficulty": query.difficulty_level,
                    "correlation_id": correlation_id
                }
            )

        except Exception as e:
            logger.error(f"Upcycling paths request failed: {e}", exc_info=True, correlation_id=correlation_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {str(e)}"
            )

    try:
        return await asyncio.wait_for(
            _handle_request(),
            timeout=float(os.getenv("REQUEST_TIMEOUT", "30.0"))
        )
    except asyncio.TimeoutError:
        logger.error("Request timeout", correlation_id=correlation_id, endpoint="upcycling_paths")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timeout - query took too long"
        )


@app.post("/relationships", response_model=KGResponse)
async def query_relationships_endpoint(query: RelationshipQuery, http_request: Request):
    """
    Query general relationships from an entity

    Explores connections between materials, products, and processes
    in the knowledge graph.
    """
    # CRITICAL FIX: Add correlation ID propagation
    correlation_id = http_request.headers.get("X-Correlation-ID")
    set_correlation_id(correlation_id)

    # CRITICAL FIX: Add request timeout wrapper
    async def _handle_request():
        try:
            start_time = datetime.now()

            relationships = await kg_service.query_relationships(
                entity=query.entity,
                relationship_type=query.relationship_type,
                max_hops=query.max_hops
            )

            query_time = (datetime.now() - start_time).total_seconds() * 1000

            return KGResponse(
                results=relationships,
                query_type="relationships",
                num_results=len(relationships),
                query_time_ms=query_time,
                metadata={
                    "entity": query.entity,
                    "relationship_type": query.relationship_type,
                    "max_hops": query.max_hops,
                    "correlation_id": correlation_id
                }
            )

        except Exception as e:
            logger.error(f"Relationships request failed: {e}", exc_info=True, correlation_id=correlation_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {str(e)}"
            )

    try:
        return await asyncio.wait_for(
            _handle_request(),
            timeout=float(os.getenv("REQUEST_TIMEOUT", "30.0"))
        )
    except asyncio.TimeoutError:
        logger.error("Request timeout", correlation_id=correlation_id, endpoint="relationships")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timeout - query took too long"
        )


@app.post("/similar-materials", response_model=KGResponse)
async def find_similar_materials_endpoint(query: SimilarMaterialsQuery, http_request: Request):
    """
    Find materials similar to the given material

    Returns materials with similar properties, recycling categories,
    and environmental characteristics.
    """
    # CRITICAL FIX: Add correlation ID propagation
    correlation_id = http_request.headers.get("X-Correlation-ID")
    set_correlation_id(correlation_id)

    # CRITICAL FIX: Add request timeout wrapper
    async def _handle_request():
        try:
            start_time = datetime.now()

            similar_materials = await kg_service.find_similar_materials(
                material_name=query.material_name,
                similarity_threshold=query.similarity_threshold,
                max_results=query.max_results
            )

            query_time = (datetime.now() - start_time).total_seconds() * 1000

            return KGResponse(
                results=similar_materials,
                query_type="similar_materials",
                num_results=len(similar_materials),
                query_time_ms=query_time,
                metadata={
                    "material": query.material_name,
                    "threshold": query.similarity_threshold,
                    "max_results": query.max_results,
                    "correlation_id": correlation_id
                }
            )

        except Exception as e:
            logger.error(f"Similar materials request failed: {e}", exc_info=True, correlation_id=correlation_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {str(e)}"
            )

    try:
        return await asyncio.wait_for(
            _handle_request(),
            timeout=float(os.getenv("REQUEST_TIMEOUT", "30.0"))
        )
    except asyncio.TimeoutError:
        logger.error("Request timeout", correlation_id=correlation_id, endpoint="similar_materials")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timeout - query took too long"
        )


@app.post("/recycling-process", response_model=KGResponse)
async def query_recycling_process_endpoint(query: RecyclingProcessQuery, http_request: Request):
    """
    Query recycling process for a material

    Returns detailed recycling process information including steps,
    facilities, and requirements.
    """
    # CRITICAL FIX: Add correlation ID propagation
    correlation_id = http_request.headers.get("X-Correlation-ID")
    set_correlation_id(correlation_id)

    # CRITICAL FIX: Add request timeout wrapper
    async def _handle_request():
        try:
            start_time = datetime.now()

            process_info = await kg_service.query_recycling_process(
                material_name=query.material_name,
                include_steps=query.include_steps,
                include_facilities=query.include_facilities
            )

            query_time = (datetime.now() - start_time).total_seconds() * 1000

            return KGResponse(
                results=[process_info],
                query_type="recycling_process",
                num_results=1,
                query_time_ms=query_time,
                metadata={
                    "material": query.material_name,
                    "include_steps": query.include_steps,
                    "include_facilities": query.include_facilities,
                    "correlation_id": correlation_id
                }
            )

        except Exception as e:
            logger.error(f"Recycling process request failed: {e}", exc_info=True, correlation_id=correlation_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {str(e)}"
            )

    try:
        return await asyncio.wait_for(
            _handle_request(),
            timeout=float(os.getenv("REQUEST_TIMEOUT", "30.0"))
        )
    except asyncio.TimeoutError:
        logger.error("Request timeout", correlation_id=correlation_id, endpoint="recycling_process")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timeout - query took too long"
        )


@app.post("/compatibility", response_model=KGResponse)
async def check_compatibility_endpoint(query: CompatibilityQuery, http_request: Request):
    """
    Check material compatibility for upcycling

    Returns compatibility score and reasons for combining two materials
    in upcycling projects.
    """
    # CRITICAL FIX: Add correlation ID propagation
    correlation_id = http_request.headers.get("X-Correlation-ID")
    set_correlation_id(correlation_id)

    # CRITICAL FIX: Add request timeout wrapper
    async def _handle_request():
        try:
            start_time = datetime.now()

            compatibility_info = await kg_service.check_material_compatibility(
                material1=query.material1,
                material2=query.material2,
                context=query.context
            )

            query_time = (datetime.now() - start_time).total_seconds() * 1000

            return KGResponse(
                results=[compatibility_info],
                query_type="compatibility",
                num_results=1,
                query_time_ms=query_time,
                metadata={
                    "material1": query.material1,
                    "material2": query.material2,
                    "context": query.context,
                    "correlation_id": correlation_id
                }
            )

        except Exception as e:
            logger.error(f"Compatibility check request failed: {e}", exc_info=True, correlation_id=correlation_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {str(e)}"
            )

    try:
        return await asyncio.wait_for(
            _handle_request(),
            timeout=float(os.getenv("REQUEST_TIMEOUT", "30.0"))
        )
    except asyncio.TimeoutError:
        logger.error("Request timeout", correlation_id=correlation_id, endpoint="compatibility")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timeout - query took too long"
        )


@app.get("/stats")
async def get_stats():
    """Get knowledge graph and cache statistics"""
    try:
        if kg_service.driver is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j not connected"
            )

        async with kg_service.driver.session(database=kg_service.database) as session:
            # Count nodes by label with timeout
            node_counts = {}
            for label in ["Material", "Product", "Process"]:
                result = await asyncio.wait_for(
                    session.run(f"MATCH (n:{label}) RETURN count(n) as count"),
                    timeout=10.0
                )
                record = await result.single()
                node_counts[label.lower()] = record["count"] if record else 0

            # Count relationships
            result = await asyncio.wait_for(
                session.run("MATCH ()-[r]->() RETURN count(r) as count"),
                timeout=10.0
            )
            record = await result.single()
            relationship_count = record["count"] if record else 0

            return {
                "database": kg_service.database,
                "nodes": node_counts,
                "relationships": relationship_count,
                "total_nodes": sum(node_counts.values()),
                "cache_size": len(query_cache.cache),
                "cache_max_size": query_cache.max_size,
                "cache_ttl_seconds": query_cache.ttl_seconds
            }

    except asyncio.TimeoutError:
        logger.error("Stats query timeout")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Query timeout"
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
        port=int(os.getenv("PORT", "8004")),
        workers=int(os.getenv("WORKERS", "1")),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
        # Production optimizations
        limit_concurrency=int(os.getenv("MAX_CONCURRENT", "100")),
        timeout_keep_alive=30,
    )

