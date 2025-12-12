"""
Health Check System

CRITICAL: Production-grade health checks for Kubernetes/Docker deployments
- Liveness probes (is service alive?)
- Readiness probes (is service ready to accept traffic?)
- Dependency health checks (databases, external services)
- Startup probes (has service finished initialization?)
- Health check aggregation
- Graceful degradation

Features:
- Multiple health check types
- Dependency checking (Neo4j, Qdrant, PostgreSQL, Redis)
- Timeout handling
- Async health checks
- Health status aggregation
- FastAPI integration
- Kubernetes-compatible endpoints

Usage:
    from services.common.health_checks import HealthChecker, HealthStatus
    
    # Initialize health checker
    health_checker = HealthChecker(service_name="llm_service")
    
    # Add dependency checks
    health_checker.add_check("neo4j", check_neo4j_health)
    health_checker.add_check("redis", check_redis_health)
    
    # Check health
    status = await health_checker.check_health()
    
    # FastAPI endpoints
    @app.get("/health/live")
    async def liveness():
        return await health_checker.liveness()
    
    @app.get("/health/ready")
    async def readiness():
        return await health_checker.readiness()
"""

import logging
import asyncio
from typing import Optional, Dict, Any, Callable, Awaitable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """
    Health check result
    
    Attributes:
        status: Health status
        message: Status message
        details: Additional details
        timestamp: Check timestamp
        duration_ms: Check duration in milliseconds
    """
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0


class HealthChecker:
    """
    Health checker for services
    
    Features:
    - Liveness checks (is service alive?)
    - Readiness checks (is service ready?)
    - Dependency checks
    - Timeout handling
    - Health aggregation
    """
    
    def __init__(
        self,
        service_name: str,
        check_timeout: float = 5.0,
        startup_timeout: float = 60.0
    ):
        """
        Initialize health checker
        
        Args:
            service_name: Name of the service
            check_timeout: Timeout for individual checks (seconds)
            startup_timeout: Timeout for startup (seconds)
        """
        self.service_name = service_name
        self.check_timeout = check_timeout
        self.startup_timeout = startup_timeout
        
        # Health checks
        self.checks: Dict[str, Callable[[], Awaitable[HealthCheckResult]]] = {}
        
        # State
        self.is_alive = True
        self.is_ready = False
        self.startup_complete = False
        self.startup_time: Optional[datetime] = None
        
        logger.info(f"Health checker initialized for service: {service_name}")
    
    def add_check(self, name: str, check_func: Callable[[], Awaitable[HealthCheckResult]]):
        """
        Add a health check
        
        Args:
            name: Check name
            check_func: Async function that returns HealthCheckResult
        
        Example:
            async def check_database():
                try:
                    await db.execute("SELECT 1")
                    return HealthCheckResult(status=HealthStatus.HEALTHY)
                except Exception as e:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message=str(e)
                    )
            
            health_checker.add_check("database", check_database)
        """
        self.checks[name] = check_func
        logger.debug(f"Added health check: {name}")
    
    def remove_check(self, name: str):
        """Remove a health check"""
        if name in self.checks:
            del self.checks[name]
            logger.debug(f"Removed health check: {name}")
    
    async def liveness(self) -> Dict[str, Any]:
        """
        Liveness probe - is service alive?
        
        Returns:
            Liveness status
        
        Example:
            @app.get("/health/live")
            async def liveness():
                return await health_checker.liveness()
        """
        return {
            "status": "alive" if self.is_alive else "dead",
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def readiness(self) -> Dict[str, Any]:
        """
        Readiness probe - is service ready to accept traffic?
        
        Returns:
            Readiness status with dependency checks
        
        Example:
            @app.get("/health/ready")
            async def readiness():
                return await health_checker.readiness()
        """
        if not self.is_ready:
            return {
                "status": "not_ready",
                "service": self.service_name,
                "message": "Service not ready",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Run all health checks
        health_result = await self.check_health()
        
        return {
            "status": "ready" if health_result.status == HealthStatus.HEALTHY else "not_ready",
            "service": self.service_name,
            "health": health_result.status,
            "checks": health_result.details,
            "timestamp": health_result.timestamp.isoformat()
        }

    async def startup(self) -> Dict[str, Any]:
        """
        Startup probe - has service finished initialization?

        Returns:
            Startup status

        Example:
            @app.get("/health/startup")
            async def startup():
                return await health_checker.startup()
        """
        return {
            "status": "started" if self.startup_complete else "starting",
            "service": self.service_name,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def check_health(self) -> HealthCheckResult:
        """
        Run all health checks and aggregate results

        Returns:
            Aggregated health check result

        Example:
            result = await health_checker.check_health()
            if result.status == HealthStatus.HEALTHY:
                print("All systems operational")
        """
        import time
        start_time = time.time()

        if not self.checks:
            # No checks configured
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="No health checks configured",
                duration_ms=(time.time() - start_time) * 1000
            )

        # Run all checks concurrently
        check_results = {}
        tasks = {}

        for name, check_func in self.checks.items():
            tasks[name] = asyncio.create_task(
                self._run_check_with_timeout(name, check_func)
            )

        # Wait for all checks
        for name, task in tasks.items():
            try:
                check_results[name] = await task
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}", exc_info=True)
                check_results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}"
                )

        # Aggregate results
        all_healthy = all(r.status == HealthStatus.HEALTHY for r in check_results.values())
        any_unhealthy = any(r.status == HealthStatus.UNHEALTHY for r in check_results.values())

        if all_healthy:
            overall_status = HealthStatus.HEALTHY
            message = "All checks passed"
        elif any_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
            unhealthy_checks = [name for name, r in check_results.items() if r.status == HealthStatus.UNHEALTHY]
            message = f"Unhealthy checks: {', '.join(unhealthy_checks)}"
        else:
            overall_status = HealthStatus.DEGRADED
            degraded_checks = [name for name, r in check_results.items() if r.status == HealthStatus.DEGRADED]
            message = f"Degraded checks: {', '.join(degraded_checks)}"

        # Build details
        details = {
            name: {
                "status": result.status,
                "message": result.message,
                "duration_ms": result.duration_ms,
                **result.details
            }
            for name, result in check_results.items()
        }

        return HealthCheckResult(
            status=overall_status,
            message=message,
            details=details,
            duration_ms=(time.time() - start_time) * 1000
        )

    async def _run_check_with_timeout(self, name: str, check_func: Callable) -> HealthCheckResult:
        """Run health check with timeout"""
        import time
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                check_func(),
                timeout=self.check_timeout
            )
            result.duration_ms = (time.time() - start_time) * 1000
            return result
        except asyncio.TimeoutError:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {self.check_timeout}s",
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )

    def mark_ready(self):
        """Mark service as ready to accept traffic"""
        self.is_ready = True
        logger.info(f"Service marked as ready: {self.service_name}")

    def mark_not_ready(self):
        """Mark service as not ready"""
        self.is_ready = False
        logger.warning(f"Service marked as not ready: {self.service_name}")

    def mark_startup_complete(self):
        """Mark startup as complete"""
        self.startup_complete = True
        self.startup_time = datetime.utcnow()
        logger.info(f"Startup complete: {self.service_name}")

    def mark_dead(self):
        """Mark service as dead (for graceful shutdown)"""
        self.is_alive = False
        logger.warning(f"Service marked as dead: {self.service_name}")


# Common health check functions

async def check_neo4j_health(driver) -> HealthCheckResult:
    """
    Check Neo4j health

    Args:
        driver: Neo4j driver instance

    Returns:
        Health check result
    """
    try:
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS health")
            await result.single()

        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Neo4j connection healthy"
        )
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message=f"Neo4j connection failed: {str(e)}"
        )


async def check_qdrant_health(client) -> HealthCheckResult:
    """
    Check Qdrant health

    Args:
        client: Qdrant client instance

    Returns:
        Health check result
    """
    try:
        # Try to get collections
        collections = await client.get_collections()

        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Qdrant connection healthy",
            details={"collections": len(collections.collections)}
        )
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message=f"Qdrant connection failed: {str(e)}"
        )


async def check_postgres_health(pool) -> HealthCheckResult:
    """
    Check PostgreSQL health

    Args:
        pool: asyncpg connection pool

    Returns:
        Health check result
    """
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="PostgreSQL connection healthy"
        )
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message=f"PostgreSQL connection failed: {str(e)}"
        )


async def check_redis_health(redis_cache) -> HealthCheckResult:
    """
    Check Redis health

    Args:
        redis_cache: RedisCache instance

    Returns:
        Health check result
    """
    try:
        # Try to set and get a test key
        test_key = "__health_check__"
        await redis_cache.set(test_key, "ok", ttl=10)
        value = await redis_cache.get(test_key)
        await redis_cache.delete(test_key)

        if value == "ok":
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Redis connection healthy"
            )
        else:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message="Redis connection degraded (using fallback)"
            )
    except Exception as e:
        return HealthCheckResult(
            status=HealthStatus.DEGRADED,
            message=f"Redis unavailable (using fallback): {str(e)}"
        )
