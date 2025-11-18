"""
Authentication Middleware
Handles API key validation and user authentication
"""

import logging
import os
from typing import Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for API key validation
    
    For production, integrate with proper auth service (OAuth2, JWT, etc.)
    """
    
    def __init__(self, app, require_auth: bool = False):
        super().__init__(app)
        self.require_auth = require_auth
        
        # Load valid API keys from environment
        api_keys_str = os.getenv("VALID_API_KEYS", "")
        self.valid_api_keys = set(
            key.strip() for key in api_keys_str.split(",") if key.strip()
        )
        
        # Public endpoints that don't require auth
        self.public_endpoints = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with authentication"""
        # Skip auth for public endpoints
        if request.url.path in self.public_endpoints:
            return await call_next(request)
        
        # Skip auth if not required (development mode)
        if not self.require_auth:
            return await call_next(request)
        
        # Extract API key from header
        api_key = self._extract_api_key(request)
        
        # Validate API key
        if not api_key:
            logger.warning(f"Missing API key for {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Missing API key",
                    "message": "Provide API key in X-API-Key header or Authorization header"
                }
            )
        
        if not self._validate_api_key(api_key):
            logger.warning(f"Invalid API key for {request.url.path}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Invalid API key",
                    "message": "The provided API key is not valid"
                }
            )
        
        # Add user info to request state
        request.state.api_key = api_key
        request.state.authenticated = True
        
        return await call_next(request)
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers"""
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        # Check Authorization header (Bearer token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Check query parameter (less secure, for testing only)
        api_key = request.query_params.get("api_key")
        if api_key:
            logger.warning("API key provided in query parameter (insecure)")
            return api_key
        
        return None
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        # If no valid keys configured, allow all (development mode)
        if not self.valid_api_keys:
            logger.warning("No API keys configured - allowing all requests")
            return True
        
        # Check if key is in valid set
        return api_key in self.valid_api_keys

