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
    
    def __init__(self, app, require_auth: bool = True):
        super().__init__(app)

        # CRITICAL FIX: Check environment mode
        env = os.getenv("ENV", "production").lower()
        self.is_dev_mode = env in ("dev", "development", "local")

        # CRITICAL FIX: In production, auth is ALWAYS required
        if env == "production" and not require_auth:
            logger.error("SECURITY: Cannot disable auth in production mode!")
            raise ValueError("Authentication cannot be disabled in production")

        self.require_auth = require_auth if self.is_dev_mode else True

        # Load valid API keys from environment
        api_keys_str = os.getenv("VALID_API_KEYS", "")
        self.valid_api_keys = set(
            key.strip() for key in api_keys_str.split(",") if key.strip()
        )

        # CRITICAL FIX: Fail closed - require keys in production
        if not self.valid_api_keys and not self.is_dev_mode:
            logger.error("SECURITY: No API keys configured in production mode!")
            raise ValueError("VALID_API_KEYS must be set in production")

        # Public endpoints that don't require auth
        self.public_endpoints = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        }

        logger.info(f"Auth middleware initialized: env={env}, require_auth={self.require_auth}, keys_configured={len(self.valid_api_keys)}")
    
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
        """
        Extract API key from request headers

        CRITICAL FIX: Removed query param support - keys must be in headers only
        """
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        # Check Authorization header (Bearer token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        # CRITICAL FIX: Query params removed - insecure (leak in logs, analytics, referrers)
        # API keys MUST be in headers only

        return None
    
    def _validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key

        CRITICAL FIX: Fail closed - no keys = reject all (except in dev mode)
        """
        # CRITICAL FIX: If no valid keys configured, DENY in production
        if not self.valid_api_keys:
            if self.is_dev_mode:
                logger.warning("DEV MODE: No API keys configured - allowing all requests")
                return True
            else:
                logger.error("PRODUCTION: No API keys configured - denying request")
                return False

        # Check if key is in valid set
        return api_key in self.valid_api_keys

