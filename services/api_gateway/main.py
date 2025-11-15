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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ReleAF AI API",
    description="AI-powered sustainability and waste intelligence platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

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
    logger.info("Starting ReleAF AI API Gateway")
    # Initialize connections, load configs, etc.


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down ReleAF AI API Gateway")
    # Cleanup connections, save state, etc.


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )

