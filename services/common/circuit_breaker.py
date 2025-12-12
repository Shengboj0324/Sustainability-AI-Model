"""
Circuit Breaker Pattern Implementation

CRITICAL: Prevents cascade failures in distributed systems
- Automatically opens circuit when failure threshold is reached
- Half-open state for testing recovery
- Configurable failure thresholds and timeouts
- Per-service circuit breakers
- Prometheus metrics integration

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Circuit is open, requests fail fast
- HALF_OPEN: Testing if service has recovered

Usage:
    breaker = CircuitBreaker(
        name="neo4j",
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=Neo4jError
    )

    # Wrap external calls
    try:
        result = await breaker.call(async_function, *args, **kwargs)
    except CircuitBreakerOpen:
        # Handle circuit open (fail fast)
        return fallback_response
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional, Type
from datetime import datetime, timedelta

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
CIRCUIT_BREAKER_STATE = Gauge('circuit_breaker_state', 'Circuit breaker state (0=closed, 1=open, 2=half_open)', ['name'])
CIRCUIT_BREAKER_FAILURES = Counter('circuit_breaker_failures_total', 'Circuit breaker failures', ['name'])
CIRCUIT_BREAKER_SUCCESSES = Counter('circuit_breaker_successes_total', 'Circuit breaker successes', ['name'])
CIRCUIT_BREAKER_OPENS = Counter('circuit_breaker_opens_total', 'Circuit breaker opens', ['name'])
CIRCUIT_BREAKER_CLOSES = Counter('circuit_breaker_closes_total', 'Circuit breaker closes', ['name'])
CIRCUIT_BREAKER_CALL_DURATION = Histogram('circuit_breaker_call_duration_seconds', 'Circuit breaker call duration', ['name', 'status'])


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = 0      # Normal operation
    OPEN = 1        # Circuit is open, fail fast
    HALF_OPEN = 2   # Testing recovery


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Production-grade circuit breaker implementation

    Features:
    - Automatic failure detection and circuit opening
    - Configurable failure threshold and recovery timeout
    - Half-open state for testing recovery
    - Per-service isolation
    - Comprehensive metrics and logging
    - Thread-safe operations

    The circuit breaker prevents cascade failures by:
    1. Tracking failures for each external service
    2. Opening the circuit when failure threshold is reached
    3. Failing fast while circuit is open (no calls to failing service)
    4. Periodically testing recovery in half-open state
    5. Closing circuit when service recovers
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception,
        success_threshold: int = 2,
        timeout: float = 30.0
    ):
        """
        Initialize circuit breaker

        Args:
            name: Circuit breaker name (e.g., "neo4j", "qdrant", "postgres")
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch (default: Exception)
            success_threshold: Successes needed in half-open to close circuit
            timeout: Timeout for individual calls in seconds
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.timeout = timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change: float = time.time()

        self._lock = asyncio.Lock()

        # Initialize metrics
        CIRCUIT_BREAKER_STATE.labels(name=self.name).set(CircuitState.CLOSED.value)

        logger.info(
            f"Circuit breaker '{self.name}' initialized: "
            f"failure_threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s"
        )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection

        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpen: If circuit is open
            TimeoutError: If call exceeds timeout
            Exception: If func raises an exception
        """
        start_time = time.time()

        async with self._lock:
            # Check if circuit should transition to half-open
            if self.state == CircuitState.OPEN:
                if self.last_failure_time and (time.time() - self.last_failure_time) >= self.recovery_timeout:
                    self._transition_to_half_open()
                else:
                    # Circuit is still open, fail fast
                    CIRCUIT_BREAKER_CALL_DURATION.labels(name=self.name, status="rejected").observe(time.time() - start_time)
                    raise CircuitBreakerOpen(f"Circuit breaker '{self.name}' is OPEN")

        # Execute the function
        try:
            # Call with timeout
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)

            # Success - record it
            await self._on_success()

            CIRCUIT_BREAKER_CALL_DURATION.labels(name=self.name, status="success").observe(time.time() - start_time)
            return result

        except asyncio.TimeoutError:
            # Timeout is treated as failure
            await self._on_failure()
            CIRCUIT_BREAKER_CALL_DURATION.labels(name=self.name, status="timeout").observe(time.time() - start_time)
            raise

        except self.expected_exception as e:
            # Expected exception - record failure
            await self._on_failure()
            CIRCUIT_BREAKER_CALL_DURATION.labels(name=self.name, status="failure").observe(time.time() - start_time)
            raise

        except Exception as e:
            # Unexpected exception - still record as failure
            logger.error(f"Unexpected exception in circuit breaker '{self.name}': {e}")
            await self._on_failure()
            CIRCUIT_BREAKER_CALL_DURATION.labels(name=self.name, status="error").observe(time.time() - start_time)
            raise

    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            CIRCUIT_BREAKER_SUCCESSES.labels(name=self.name).inc()

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                logger.info(f"Circuit breaker '{self.name}' success in HALF_OPEN: {self.success_count}/{self.success_threshold}")

                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()

            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            CIRCUIT_BREAKER_FAILURES.labels(name=self.name).inc()
            self.failure_count += 1
            self.last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self.name}' failure: "
                f"{self.failure_count}/{self.failure_threshold} (state={self.state.name})"
            )

            if self.state == CircuitState.HALF_OPEN:
                # Failure in half-open immediately opens circuit
                self._transition_to_open()

            elif self.state == CircuitState.CLOSED:
                # Check if we should open circuit
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.last_state_change = time.time()
        self.success_count = 0

        CIRCUIT_BREAKER_STATE.labels(name=self.name).set(CircuitState.OPEN.value)
        CIRCUIT_BREAKER_OPENS.labels(name=self.name).inc()

        logger.error(
            f"Circuit breaker '{self.name}' OPENED after {self.failure_count} failures. "
            f"Will attempt recovery in {self.recovery_timeout}s"
        )

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.time()
        self.failure_count = 0
        self.success_count = 0

        CIRCUIT_BREAKER_STATE.labels(name=self.name).set(CircuitState.HALF_OPEN.value)

        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN (testing recovery)")

    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.failure_count = 0
        self.success_count = 0

        CIRCUIT_BREAKER_STATE.labels(name=self.name).set(CircuitState.CLOSED.value)
        CIRCUIT_BREAKER_CLOSES.labels(name=self.name).inc()

        logger.info(f"Circuit breaker '{self.name}' CLOSED (service recovered)")

    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": self.last_failure_time,
            "last_state_change": self.last_state_change,
            "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time else None,
            "time_in_current_state": time.time() - self.last_state_change
        }

    async def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        async with self._lock:
            self._transition_to_closed()
            logger.info(f"Circuit breaker '{self.name}' manually reset")

    async def force_open(self):
        """Manually force circuit breaker to OPEN state"""
        async with self._lock:
            self._transition_to_open()
            logger.warning(f"Circuit breaker '{self.name}' manually forced OPEN")

