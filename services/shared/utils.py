"""
Shared Utility Classes - Consolidated from all services

CRITICAL: Single source of truth for common utilities
- RateLimiter: Prevents DoS attacks across all services
- RequestCache: LRU cache with TTL for expensive operations
- QueryCache: Specialized cache for query-based operations

This eliminates duplicate implementations across:
- services/llm_service/server_v2.py
- services/vision_service/server_v2.py
- services/rag_service/server.py
- services/kg_service/server.py
- services/org_search_service/server.py
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple


class RateLimiter:
    """
    Thread-safe in-memory rate limiter with sliding window

    CRITICAL: Prevents DoS attacks - protects expensive operations

    Usage:
        rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
        if await rate_limiter.check_rate_limit(client_ip):
            # Process request
        else:
            # Reject request (429 Too Many Requests)
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def check_rate_limit(self, client_ip: str) -> bool:
        """
        Check if request is within rate limit

        Args:
            client_ip: Client IP address or identifier

        Returns:
            True if within limit, False if exceeded
        """
        async with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window_seconds)

            # Remove old requests outside window
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if req_time > cutoff
            ]

            # Check if limit exceeded
            if len(self.requests[client_ip]) >= self.max_requests:
                return False

            # Add current request
            self.requests[client_ip].append(now)
            return True

    async def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for client"""
        async with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window_seconds)

            # Count valid requests
            valid_requests = [
                req_time for req_time in self.requests.get(client_ip, [])
                if req_time > cutoff
            ]

            return max(0, self.max_requests - len(valid_requests))

    async def reset(self, client_ip: str):
        """Reset rate limit for client"""
        async with self.lock:
            if client_ip in self.requests:
                del self.requests[client_ip]


class RequestCache:
    """
    Thread-safe LRU cache with TTL for expensive operations

    CRITICAL: Caches expensive operations (LLM inference, vision processing)

    Usage:
        cache = RequestCache(max_size=500, ttl_seconds=600)

        # Try to get from cache
        result = await cache.get(cache_key)
        if result is None:
            # Compute expensive operation
            result = expensive_operation()
            await cache.set(cache_key, result)
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 600):
        """
        Initialize request cache

        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """
        Get cached result if not expired

        Args:
            key: Cache key

        Returns:
            Cached result or None if not found/expired
        """
        async with self.lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    return result
                else:
                    # Expired - remove
                    del self.cache[key]
            return None

    async def set(self, key: str, value: Any):
        """
        Set cache entry with TTL

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self.lock:
            # Evict oldest if at capacity (LRU)
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]

            # Add new entry
            self.cache[key] = (value, datetime.now())

    async def clear(self):
        """Clear all cache entries"""
        async with self.lock:
            self.cache.clear()

    async def size(self) -> int:
        """Get current cache size"""
        async with self.lock:
            return len(self.cache)


class QueryCache:
    """
    Thread-safe query cache with TTL - specialized for query operations

    CRITICAL: Optimized for RAG, KG, and search operations

    Features:
    - Automatic key generation from query parameters
    - LRU eviction when at capacity
    - TTL-based expiration
    - Thread-safe operations

    Usage:
        cache = QueryCache(max_size=1000, ttl_seconds=300)

        # Try to get from cache
        result = await cache.get(query, top_k=5, mode="hybrid", doc_types=["guide"])
        if result is None:
            # Perform query
            result = perform_query()
            await cache.set(query, top_k=5, mode="hybrid", doc_types=["guide"], result=result)
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize query cache

        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live in seconds
        """
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()

    def _make_key(self, query: str, top_k: int, mode: str, doc_types: Optional[List[str]]) -> str:
        """
        Create cache key from query parameters

        Args:
            query: Query string
            top_k: Number of results
            mode: Retrieval mode
            doc_types: Document types filter

        Returns:
            MD5 hash of parameters
        """
        key_str = f"{query}:{top_k}:{mode}:{sorted(doc_types) if doc_types else []}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def get(self, query: str, top_k: int, mode: str, doc_types: Optional[List[str]] = None) -> Optional[Any]:
        """
        Get cached result if not expired

        Args:
            query: Query string
            top_k: Number of results
            mode: Retrieval mode
            doc_types: Document types filter

        Returns:
            Cached result or None if not found/expired
        """
        async with self._lock:
            key = self._make_key(query, top_k, mode, doc_types)
            if key in self.cache:
                result, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return result
                else:
                    # Expired - remove
                    del self.cache[key]
            return None

    async def set(self, query: str, top_k: int, mode: str, doc_types: Optional[List[str]], result: Any):
        """
        Set cache entry with TTL

        Args:
            query: Query string
            top_k: Number of results
            mode: Retrieval mode
            doc_types: Document types filter
            result: Result to cache
        """
        async with self._lock:
            # Evict oldest if at capacity (LRU)
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]

            key = self._make_key(query, top_k, mode, doc_types)
            self.cache[key] = (result, time.time())

    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()

    async def size(self) -> int:
        """Get current cache size"""
        async with self._lock:
            return len(self.cache)

