"""
API Gateway - Main entry point for ReleAF AI platform
Handles authentication, rate limiting, and request routing
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import httpx
import time
from datetime import datetime
import logging

from .routers import chat, vision, organizations
from .middleware import RateLimitMiddleware, AuthMiddleware
from .schemas import HealthResponse

# Import monitoring components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.structured_logging import get_logger, log_context, set_correlation_id
from common.health_checks import HealthChecker, HealthStatus
from common.alerting import init_alerting, send_alert, AlertSeverity

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

# Initialize FastAPI app
app = FastAPI(
    title="ReleAF AI API",
    description="AI-powered sustainability and waste intelligence platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize monitoring components
health_checker = HealthChecker(service_name="api_gateway", check_timeout=5.0)
alert_manager = None  # Initialized in startup

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(vision.router, prefix="/api/v1/vision", tags=["vision"])
app.include_router(organizations.router, prefix="/api/v1/organizations", tags=["organizations"])


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "ReleAF AI API Gateway",
        "version": "0.1.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Check downstream services
    services_status = await check_downstream_services()
    
    all_healthy = all(status["healthy"] for status in services_status.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.utcnow(),
        services=services_status
    )


async def check_downstream_services() -> Dict[str, Dict[str, Any]]:
    """Check health of downstream services"""
    services = {
        "orchestrator": "http://localhost:8000/health",
        "vision": "http://localhost:8001/health",
        "llm": "http://localhost:8002/health",
        "rag": "http://localhost:8003/health",
        "kg": "http://localhost:8004/health",
        "org_search": "http://localhost:8005/health",
    }
    
    status = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, url in services.items():
            try:
                response = await client.get(url)
                status[service_name] = {
                    "healthy": response.status_code == 200,
                    "latency_ms": response.elapsed.total_seconds() * 1000
                }
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
                status[service_name] = {
                    "healthy": False,
                    "error": str(e)
                }
    
    return status


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    global alert_manager

    logger.info("Starting API Gateway", service="api_gateway", version="0.1.0")

    # Initialize distributed tracing
    if TRACING_AVAILABLE:
        try:
            import os
            jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
            if jaeger_endpoint:
                init_tracing(
                    service_name="api_gateway",
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
            import os
            sentry_dsn = os.getenv("SENTRY_DSN")
            if sentry_dsn:
                init_sentry(
                    dsn=sentry_dsn,
                    service_name="api_gateway",
                    environment=os.getenv("ENVIRONMENT", "development"),
                    release=os.getenv("RELEASE_VERSION", "0.1.0"),
                    traces_sample_rate=float(os.getenv("SENTRY_TRACE_SAMPLE_RATE", "0.1"))
                )
                logger.info("Error tracking initialized", sentry_enabled=True)
        except Exception as e:
            logger.warning("Failed to initialize Sentry", error=str(e))

    # Initialize alerting
    try:
        import os
        alert_manager = init_alerting(
            slack_webhook=os.getenv("SLACK_WEBHOOK"),
            pagerduty_key=os.getenv("PAGERDUTY_KEY")
        )
        logger.info("Alerting system initialized")
    except Exception as e:
        logger.warning("Failed to initialize alerting", error=str(e))

    # Mark service as ready
    health_checker.mark_ready()
    health_checker.mark_startup_complete()
    logger.info("API Gateway initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down API Gateway")
    health_checker.mark_not_ready()
    logger.info("API Gateway shutdown complete")


# Enhanced health check endpoints
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )

