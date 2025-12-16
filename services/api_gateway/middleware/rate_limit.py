"""
Rate Limiting Middleware
Implements token bucket algorithm for API rate limiting
"""

import time
import logging
import asyncio
from typing import Dict, Tuple
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import os

logger = logging.getLogger(__name__)


class TokenBucket:
    """Thread-safe token bucket for rate limiting"""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens (thread-safe)

        Returns:
            True if tokens were consumed, False otherwise
        """
        async with self.lock:
            # Refill tokens based on time elapsed
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.refill_rate
            )
            self.last_refill = now

            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait until tokens are available (thread-safe)"""
        async with self.lock:
            if self.tokens >= tokens:
                return 0.0
            needed = tokens - self.tokens
            return needed / self.refill_rate


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm
    
    Default: 100 requests per minute per IP
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        burst_size: int = 20
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        self.buckets: Dict[str, TokenBucket] = {}
        self.buckets_lock = asyncio.Lock()

        # Cleanup old buckets periodically
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes

    def _get_rate_limit_tier(self, request: Request) -> Tuple[int, int]:
        """
        Determine rate limit tier based on API key or user tier

        Returns:
            Tuple of (requests_per_minute, burst_size)
        """
        # Check for premium tier (from API key validation)
        api_key = request.headers.get("X-API-Key", "")
        user_tier = getattr(request.state, "user_tier", "standard") if hasattr(request, "state") else "standard"

        # Check environment variable for tier-based limits
        enable_tiers = os.getenv("RATE_LIMIT_TIERS_ENABLED", "true").lower() == "true"

        if not enable_tiers:
            return (self.requests_per_minute, self.burst_size)

        # Determine tier
        if user_tier == "premium" or api_key.startswith("premium_"):
            return (500, 100)  # Premium: 500 req/min
        elif user_tier == "enterprise" or api_key.startswith("enterprise_"):
            return (1000, 200)  # Enterprise: 1000 req/min
        else:
            return (100, 20)  # Standard: 100 req/min

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting (thread-safe)"""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/ios", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get client identifier (IP address or API key)
        client_ip = self._get_client_ip(request)
        api_key = request.headers.get("X-API-Key", "")
        client_id = api_key if api_key else client_ip

        # Get rate limit tier
        requests_per_minute, burst_size = self._get_rate_limit_tier(request)

        # Get or create bucket for this client (thread-safe)
        async with self.buckets_lock:
            if client_id not in self.buckets:
                self.buckets[client_id] = TokenBucket(
                    capacity=burst_size,
                    refill_rate=requests_per_minute / 60.0
                )
            bucket = self.buckets[client_id]

        # Try to consume a token
        if not await bucket.consume(1):
            wait_time = await bucket.get_wait_time(1)
            logger.warning(
                f"Rate limit exceeded for {client_id}. "
                f"Wait time: {wait_time:.2f}s, Tier limit: {requests_per_minute}/min"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": int(wait_time) + 1,
                    "limit": requests_per_minute,
                    "window": "1 minute"
                },
                headers={
                    "Retry-After": str(int(wait_time) + 1),
                    "X-RateLimit-Limit": str(requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + wait_time))
                }
            )

        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
        
        # Periodic cleanup
        self._cleanup_old_buckets()
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check X-Forwarded-For header (for proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client
        return request.client.host if request.client else "unknown"
    
    def _cleanup_old_buckets(self):
        """Remove old buckets to prevent memory leak"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        # Remove buckets that haven't been used recently
        to_remove = []
        for ip, bucket in self.buckets.items():
            if now - bucket.last_refill > 600:  # 10 minutes
                to_remove.append(ip)
        
        for ip in to_remove:
            del self.buckets[ip]
        
        self.last_cleanup = now
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old rate limit buckets")

