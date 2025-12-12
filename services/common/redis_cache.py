"""
Redis-based Distributed Caching System

CRITICAL: Production-grade distributed caching for all services
- Replaces in-memory LRU cache with Redis for cross-service sharing
- Supports cache warming, intelligent invalidation, and TTL
- Connection pooling and automatic reconnection
- Prometheus metrics integration
- Async/await throughout

Features:
- Distributed caching across multiple service instances
- Automatic serialization/deserialization (JSON + pickle fallback)
- Cache warming for frequently accessed data
- Pattern-based cache invalidation
- Cache statistics and monitoring
- Graceful degradation (falls back to in-memory if Redis unavailable)

Usage:
    cache = RedisCache(
        host="localhost",
        port=6379,
        db=0,
        password=None,
        max_connections=50
    )

    # Initialize connection pool
    await cache.initialize()

    # Get/Set with TTL
    await cache.set("key", value, ttl=300)
    result = await cache.get("key")

    # Pattern-based operations
    await cache.delete_pattern("user:*")
    keys = await cache.keys_pattern("session:*")

    # Cache warming
    await cache.warm_cache({"key1": value1, "key2": value2}, ttl=600)

    # Cleanup
    await cache.close()
"""

import asyncio
import json
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Set
from datetime import timedelta

try:
    import redis.asyncio as aioredis
    from redis.asyncio import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    ConnectionPool = None

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics
REDIS_OPERATIONS = Counter('redis_operations_total', 'Redis operations', ['operation', 'status'])
REDIS_LATENCY = Histogram('redis_operation_latency_seconds', 'Redis operation latency', ['operation'])
REDIS_CACHE_SIZE = Gauge('redis_cache_size_bytes', 'Estimated Redis cache size')
REDIS_CONNECTIONS = Gauge('redis_active_connections', 'Active Redis connections')


class RedisCache:
    """
    Production-grade Redis-based distributed cache

    Features:
    - Connection pooling with automatic reconnection
    - JSON serialization with pickle fallback
    - TTL support for all keys
    - Pattern-based operations (delete, list)
    - Cache warming for frequently accessed data
    - Graceful degradation to in-memory cache
    - Comprehensive metrics and logging
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        key_prefix: str = "releaf:"
    ):
        """
        Initialize Redis cache

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number (0-15)
            password: Redis password (optional)
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            retry_on_timeout: Retry operations on timeout
            health_check_interval: Health check interval in seconds
            key_prefix: Prefix for all cache keys (namespace isolation)
        """
        if not REDIS_AVAILABLE:
            logger.warning("redis package not installed, falling back to in-memory cache")
            self.redis_available = False
            self.fallback_cache: Dict[str, tuple[Any, float]] = {}
            self.fallback_lock = asyncio.Lock()
            return

        self.redis_available = True
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        self.key_prefix = key_prefix

        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[aioredis.Redis] = None
        self._initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()



    async def initialize(self) -> bool:
        """
        Initialize Redis connection pool

        Returns:
            True if successful, False if fallback to in-memory
        """
        if not self.redis_available:
            logger.warning("Redis not available, using in-memory fallback")
            return False

        try:
            async with self._lock:
                if self._initialized:
                    logger.warning("Redis cache already initialized")
                    return True

                # Create connection pool
                self.pool = ConnectionPool(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    max_connections=self.max_connections,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_connect_timeout,
                    retry_on_timeout=self.retry_on_timeout,
                    decode_responses=False  # We handle encoding/decoding
                )

                # Create Redis client
                self.client = aioredis.Redis(connection_pool=self.pool)

                # Test connection
                await self.client.ping()

                self._initialized = True
                REDIS_CONNECTIONS.set(self.max_connections)

                # Start health check task
                self._health_check_task = asyncio.create_task(self._health_check_loop())

                logger.info(f"Redis cache initialized: {self.host}:{self.port} (db={self.db})")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            logger.warning("Falling back to in-memory cache")
            self.redis_available = False
            self.fallback_cache = {}
            self.fallback_lock = asyncio.Lock()
            return False

    async def _health_check_loop(self):
        """Background task to check Redis health"""
        while self._initialized:
            try:
                await asyncio.sleep(self.health_check_interval)
                if self.client:
                    await self.client.ping()
                    logger.debug("Redis health check: OK")
            except Exception as e:
                logger.error(f"Redis health check failed: {e}")
                REDIS_OPERATIONS.labels(operation="health_check", status="error").inc()

    def _make_key(self, key: str) -> str:
        """Add prefix to key for namespace isolation"""
        return f"{self.key_prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for Redis storage

        Tries JSON first (faster, human-readable), falls back to pickle
        """
        try:
            # Try JSON first (works for most cases)
            return json.dumps(value).encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize value from Redis

        Tries JSON first, falls back to pickle
        """
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        start_time = time.time()

        try:
            # Fallback to in-memory cache
            if not self.redis_available or not self._initialized:
                return await self._get_fallback(key)

            redis_key = self._make_key(key)
            data = await self.client.get(redis_key)

            if data is None:
                REDIS_OPERATIONS.labels(operation="get", status="miss").inc()
                REDIS_LATENCY.labels(operation="get").observe(time.time() - start_time)
                return None

            value = self._deserialize(data)
            REDIS_OPERATIONS.labels(operation="get", status="hit").inc()
            REDIS_LATENCY.labels(operation="get").observe(time.time() - start_time)
            return value

        except Exception as e:
            logger.error(f"Redis GET error for key '{key}': {e}")
            REDIS_OPERATIONS.labels(operation="get", status="error").inc()
            return await self._get_fallback(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with optional TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = no expiration)

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()

        try:
            # Fallback to in-memory cache
            if not self.redis_available or not self._initialized:
                return await self._set_fallback(key, value, ttl)

            redis_key = self._make_key(key)
            data = self._serialize(value)

            if ttl:
                await self.client.setex(redis_key, ttl, data)
            else:
                await self.client.set(redis_key, data)

            REDIS_OPERATIONS.labels(operation="set", status="success").inc()
            REDIS_LATENCY.labels(operation="set").observe(time.time() - start_time)
            return True

        except Exception as e:
            logger.error(f"Redis SET error for key '{key}': {e}")
            REDIS_OPERATIONS.labels(operation="set", status="error").inc()
            return await self._set_fallback(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            if not self.redis_available or not self._initialized:
                return await self._delete_fallback(key)

            redis_key = self._make_key(key)
            result = await self.client.delete(redis_key)

            REDIS_OPERATIONS.labels(operation="delete", status="success").inc()
            return result > 0

        except Exception as e:
            logger.error(f"Redis DELETE error for key '{key}': {e}")
            REDIS_OPERATIONS.labels(operation="delete", status="error").inc()
            return await self._delete_fallback(key)

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern

        Args:
            pattern: Key pattern (e.g., "user:*", "session:123:*")

        Returns:
            Number of keys deleted
        """
        try:
            if not self.redis_available or not self._initialized:
                return await self._delete_pattern_fallback(pattern)

            redis_pattern = self._make_key(pattern)
            keys = []

            # Scan for matching keys (cursor-based to avoid blocking)
            cursor = 0
            while True:
                cursor, batch = await self.client.scan(cursor, match=redis_pattern, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break

            if not keys:
                return 0

            # Delete in batches
            deleted = await self.client.delete(*keys)

            REDIS_OPERATIONS.labels(operation="delete_pattern", status="success").inc()
            logger.info(f"Deleted {deleted} keys matching pattern '{pattern}'")
            return deleted

        except Exception as e:
            logger.error(f"Redis DELETE_PATTERN error for pattern '{pattern}': {e}")
            REDIS_OPERATIONS.labels(operation="delete_pattern", status="error").inc()
            return await self._delete_pattern_fallback(pattern)

    async def keys_pattern(self, pattern: str) -> List[str]:
        """
        Get all keys matching pattern

        Args:
            pattern: Key pattern

        Returns:
            List of matching keys (without prefix)
        """
        try:
            if not self.redis_available or not self._initialized:
                return await self._keys_pattern_fallback(pattern)

            redis_pattern = self._make_key(pattern)
            keys = []

            # Scan for matching keys
            cursor = 0
            while True:
                cursor, batch = await self.client.scan(cursor, match=redis_pattern, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break

            # Remove prefix from keys
            prefix_len = len(self.key_prefix)
            return [k.decode('utf-8')[prefix_len:] if isinstance(k, bytes) else k[prefix_len:] for k in keys]

        except Exception as e:
            logger.error(f"Redis KEYS_PATTERN error for pattern '{pattern}': {e}")
            return await self._keys_pattern_fallback(pattern)

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if not self.redis_available or not self._initialized:
                return await self._exists_fallback(key)

            redis_key = self._make_key(key)
            result = await self.client.exists(redis_key)
            return result > 0

        except Exception as e:
            logger.error(f"Redis EXISTS error for key '{key}': {e}")
            return await self._exists_fallback(key)

    async def ttl(self, key: str) -> int:
        """
        Get remaining TTL for key

        Returns:
            TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        try:
            if not self.redis_available or not self._initialized:
                return -1  # Fallback doesn't support TTL query

            redis_key = self._make_key(key)
            return await self.client.ttl(redis_key)

        except Exception as e:
            logger.error(f"Redis TTL error for key '{key}': {e}")
            return -1

    async def warm_cache(self, data: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """
        Warm cache with multiple key-value pairs

        Args:
            data: Dictionary of key-value pairs to cache
            ttl: TTL for all keys

        Returns:
            Number of keys successfully cached
        """
        count = 0
        for key, value in data.items():
            if await self.set(key, value, ttl):
                count += 1

        logger.info(f"Cache warmed with {count}/{len(data)} keys")
        return count

    async def clear(self) -> bool:
        """Clear all cache entries with this prefix"""
        try:
            if not self.redis_available or not self._initialized:
                return await self._clear_fallback()

            # Delete all keys with our prefix
            deleted = await self.delete_pattern("*")
            logger.info(f"Cache cleared: {deleted} keys deleted")
            return True

        except Exception as e:
            logger.error(f"Redis CLEAR error: {e}")
            return await self._clear_fallback()

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.redis_available or not self._initialized:
                return {
                    "backend": "in-memory",
                    "size": len(self.fallback_cache),
                    "max_size": "unlimited"
                }

            info = await self.client.info("stats")
            memory = await self.client.info("memory")

            return {
                "backend": "redis",
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "total_connections": info.get("total_connections_received", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "used_memory": memory.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "max_connections": self.max_connections
            }

        except Exception as e:
            logger.error(f"Redis STATS error: {e}")
            return {"backend": "error", "error": str(e)}

    async def close(self):
        """Close Redis connection and cleanup"""
        try:
            async with self._lock:
                if not self._initialized:
                    return

                # Cancel health check task
                if self._health_check_task:
                    self._health_check_task.cancel()
                    try:
                        await self._health_check_task
                    except asyncio.CancelledError:
                        pass

                # Close Redis client
                if self.client:
                    await self.client.close()

                # Close connection pool
                if self.pool:
                    await self.pool.disconnect()

                self._initialized = False
                REDIS_CONNECTIONS.set(0)
                logger.info("Redis cache closed")

        except Exception as e:
            logger.error(f"Error closing Redis cache: {e}")

    # Fallback methods (in-memory cache)

    async def _get_fallback(self, key: str) -> Optional[Any]:
        """Get from in-memory fallback cache"""
        async with self.fallback_lock:
            if key in self.fallback_cache:
                value, expiry = self.fallback_cache[key]
                if expiry == 0 or time.time() < expiry:
                    return value
                else:
                    del self.fallback_cache[key]
            return None

    async def _set_fallback(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in in-memory fallback cache"""
        async with self.fallback_lock:
            expiry = time.time() + ttl if ttl else 0
            self.fallback_cache[key] = (value, expiry)
            return True

    async def _delete_fallback(self, key: str) -> bool:
        """Delete from in-memory fallback cache"""
        async with self.fallback_lock:
            if key in self.fallback_cache:
                del self.fallback_cache[key]
                return True
            return False

    async def _delete_pattern_fallback(self, pattern: str) -> int:
        """Delete pattern from in-memory fallback cache"""
        import fnmatch
        async with self.fallback_lock:
            keys_to_delete = [k for k in self.fallback_cache.keys() if fnmatch.fnmatch(k, pattern)]
            for key in keys_to_delete:
                del self.fallback_cache[key]
            return len(keys_to_delete)

    async def _keys_pattern_fallback(self, pattern: str) -> List[str]:
        """Get keys matching pattern from in-memory fallback cache"""
        import fnmatch
        async with self.fallback_lock:
            return [k for k in self.fallback_cache.keys() if fnmatch.fnmatch(k, pattern)]

    async def _exists_fallback(self, key: str) -> bool:
        """Check if key exists in in-memory fallback cache"""
        async with self.fallback_lock:
            if key in self.fallback_cache:
                value, expiry = self.fallback_cache[key]
                if expiry == 0 or time.time() < expiry:
                    return True
                else:
                    del self.fallback_cache[key]
            return False

    async def _clear_fallback(self) -> bool:
        """Clear in-memory fallback cache"""
        async with self.fallback_lock:
            self.fallback_cache.clear()
            return True


