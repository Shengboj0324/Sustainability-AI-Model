"""
Rate Limiting Middleware
Implements token bucket algorithm for API rate limiting
"""

import time
import logging
from typing import Dict, Tuple
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket for rate limiting"""
    
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
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens
        
        Returns:
            True if tokens were consumed, False otherwise
        """
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
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait until tokens are available"""
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
        self.buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(burst_size, self.refill_rate)
        )
        
        # Cleanup old buckets periodically
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = self._get_client_ip(request)
        
        # Get or create bucket for this client
        bucket = self.buckets[client_ip]
        
        # Try to consume a token
        if not bucket.consume(1):
            wait_time = bucket.get_wait_time(1)
            logger.warning(
                f"Rate limit exceeded for {client_ip}. "
                f"Wait time: {wait_time:.2f}s"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": int(wait_time) + 1,
                    "limit": self.requests_per_minute,
                    "window": "1 minute"
                },
                headers={
                    "Retry-After": str(int(wait_time) + 1),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        
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

