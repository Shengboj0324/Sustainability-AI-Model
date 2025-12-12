"""
Common utilities and shared modules for ReleAF AI services

CRITICAL: Production-grade monitoring and infrastructure components
- Structured logging with JSON format and correlation IDs
- Distributed tracing with OpenTelemetry
- Error tracking with Sentry
- Alerting system (email, Slack, PagerDuty, webhooks)
- Health checks (liveness, readiness, startup probes)
- Circuit breakers for resilience
- Redis-based distributed caching
- Input validation and security
"""

# Structured logging
from .structured_logging import (
    get_logger,
    set_correlation_id,
    get_correlation_id,
    set_request_context,
    clear_request_context,
    log_context,
    log_performance,
    JSONFormatter,
    StructuredLogger
)

# Distributed tracing
from .tracing import (
    init_tracing,
    get_tracer,
    trace_operation,
    add_span_attributes,
    add_span_event,
    record_exception,
    get_trace_context,
    set_trace_context
)

# Error tracking
from .error_tracking import (
    init_sentry,
    capture_exception,
    capture_message,
    add_breadcrumb,
    set_user,
    set_tag,
    set_context,
    start_transaction,
    flush
)

# Alerting
from .alerting import (
    AlertManager,
    Alert,
    AlertSeverity,
    init_alerting,
    get_alert_manager,
    send_alert
)

# Health checks
from .health_checks import (
    HealthChecker,
    HealthStatus,
    HealthCheckResult,
    check_neo4j_health,
    check_qdrant_health,
    check_postgres_health,
    check_redis_health
)

# Circuit breakers
from .circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpen
)

# Redis cache
from .redis_cache import RedisCache

# Input validation
from .input_validation import InputValidator

__all__ = [
    # Structured logging
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "set_request_context",
    "clear_request_context",
    "log_context",
    "log_performance",
    "JSONFormatter",
    "StructuredLogger",
    # Distributed tracing
    "init_tracing",
    "get_tracer",
    "trace_operation",
    "add_span_attributes",
    "add_span_event",
    "record_exception",
    "get_trace_context",
    "set_trace_context",
    # Error tracking
    "init_sentry",
    "capture_exception",
    "capture_message",
    "add_breadcrumb",
    "set_user",
    "set_tag",
    "set_context",
    "start_transaction",
    "flush",
    # Alerting
    "AlertManager",
    "Alert",
    "AlertSeverity",
    "init_alerting",
    "get_alert_manager",
    "send_alert",
    # Health checks
    "HealthChecker",
    "HealthStatus",
    "HealthCheckResult",
    "check_neo4j_health",
    "check_qdrant_health",
    "check_postgres_health",
    "check_redis_health",
    # Circuit breakers
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpen",
    # Redis cache
    "RedisCache",
    # Input validation
    "InputValidator"
]

