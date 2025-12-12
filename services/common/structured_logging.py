"""
Structured Logging System

CRITICAL: Production-grade JSON logging with correlation IDs, context, and log aggregation support
- JSON format for easy parsing by log aggregators (ELK, Loki, CloudWatch)
- Correlation IDs for request tracing across services
- Structured fields for filtering and searching
- Log levels with proper severity
- Performance tracking
- Error context capture
- Integration with OpenTelemetry

Features:
- JSON structured logging
- Correlation ID propagation
- Request context tracking
- Performance metrics
- Error stack traces
- Service metadata
- Environment tagging
- Log sampling for high-volume scenarios

Usage:
    from services.common.structured_logging import get_logger, log_context
    
    logger = get_logger(__name__)
    
    # Basic logging
    logger.info("User logged in", user_id=123, email="user@example.com")
    
    # With context
    with log_context(request_id="abc-123", user_id=456):
        logger.info("Processing request")
        # All logs in this block will include request_id and user_id
    
    # Error logging with exception
    try:
        risky_operation()
    except Exception as e:
        logger.error("Operation failed", exc_info=True, operation="risky_op")
"""

import logging
import json
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
import uuid
import os

# Context variables for correlation IDs and request context
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
request_context_var: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})

# Service metadata
SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
HOSTNAME = os.getenv("HOSTNAME", "localhost")


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging
    
    Outputs logs in JSON format with:
    - Timestamp (ISO 8601)
    - Log level
    - Logger name
    - Message
    - Correlation ID
    - Request context
    - Service metadata
    - Exception info (if present)
    - Custom fields
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": {
                "name": SERVICE_NAME,
                "version": SERVICE_VERSION,
                "environment": ENVIRONMENT,
                "hostname": HOSTNAME
            }
        }
        
        # Add correlation ID if present
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        # Add request context if present
        request_context = request_context_var.get()
        if request_context:
            log_entry["context"] = request_context
        
        # Add source location
        log_entry["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": traceback.format_exception(*record.exc_info)
            }
        
        # Add custom fields from extra
        if hasattr(record, 'custom_fields'):
            log_entry.update(record.custom_fields)
        
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info', 'custom_fields']:
                try:
                    # Only add JSON-serializable values
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)
        
        return json.dumps(log_entry)


class StructuredLogger(logging.LoggerAdapter):
    """
    Logger adapter that adds structured fields to log records

    Allows passing keyword arguments as structured fields:
        logger.info("User action", user_id=123, action="login")
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and add custom fields"""

        # Extract custom fields from kwargs
        custom_fields = {}
        for key in list(kwargs.keys()):
            if key not in ['exc_info', 'stack_info', 'stacklevel', 'extra']:
                custom_fields[key] = kwargs.pop(key)

        # Add custom fields to extra
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra']['custom_fields'] = custom_fields

        return msg, kwargs


def get_logger(name: str, level: int = logging.INFO) -> StructuredLogger:
    """
    Get a structured logger instance

    Args:
        name: Logger name (usually __name__)
        level: Log level (default: INFO)

    Returns:
        StructuredLogger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing request", request_id="abc-123", user_id=456)
    """

    # Get base logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    # Don't propagate to root logger
    logger.propagate = False

    # Wrap in StructuredLogger adapter
    return StructuredLogger(logger, {})


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for current context

    Args:
        correlation_id: Correlation ID (generates UUID if None)

    Returns:
        The correlation ID that was set

    Example:
        correlation_id = set_correlation_id()
        # All logs in this context will include this correlation_id
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return correlation_id_var.get()


def set_request_context(**kwargs):
    """
    Set request context for current context

    Args:
        **kwargs: Context key-value pairs

    Example:
        set_request_context(user_id=123, tenant_id=456, ip="1.2.3.4")
    """
    current_context = request_context_var.get().copy()
    current_context.update(kwargs)
    request_context_var.set(current_context)


def clear_request_context():
    """Clear request context"""
    request_context_var.set({})


class log_context:
    """
    Context manager for setting correlation ID and request context

    Example:
        with log_context(request_id="abc-123", user_id=456):
            logger.info("Processing request")
            # All logs here will include request_id and user_id
    """

    def __init__(self, correlation_id: Optional[str] = None, **context):
        """
        Initialize log context

        Args:
            correlation_id: Correlation ID (generates UUID if None)
            **context: Additional context key-value pairs
        """
        self.correlation_id = correlation_id
        self.context = context
        self.previous_correlation_id = None
        self.previous_context = None

    def __enter__(self):
        """Enter context"""
        # Save previous values
        self.previous_correlation_id = get_correlation_id()
        self.previous_context = request_context_var.get().copy()

        # Set new values
        set_correlation_id(self.correlation_id)
        set_request_context(**self.context)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        # Restore previous values
        if self.previous_correlation_id:
            correlation_id_var.set(self.previous_correlation_id)
        else:
            correlation_id_var.set(None)

        request_context_var.set(self.previous_context)


# Performance tracking decorator
def log_performance(logger: StructuredLogger, operation: str):
    """
    Decorator to log function performance

    Args:
        logger: Logger instance
        operation: Operation name

    Example:
        @log_performance(logger, "database_query")
        async def query_database():
            ...
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"{operation} completed",
                    operation=operation,
                    duration_ms=duration_ms,
                    status="success"
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{operation} failed",
                    operation=operation,
                    duration_ms=duration_ms,
                    status="error",
                    error=str(e),
                    exc_info=True
                )
                raise

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"{operation} completed",
                    operation=operation,
                    duration_ms=duration_ms,
                    status="success"
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{operation} failed",
                    operation=operation,
                    duration_ms=duration_ms,
                    status="error",
                    error=str(e),
                    exc_info=True
                )
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

