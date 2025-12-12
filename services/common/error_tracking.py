"""
Error Tracking with Sentry

CRITICAL: Production-grade error tracking and performance monitoring
- Automatic error capture and grouping
- Breadcrumbs for debugging
- User context and tags
- Performance monitoring
- Release tracking
- Environment tagging
- Custom error fingerprinting

Features:
- Sentry SDK integration
- Automatic exception capture
- Breadcrumb tracking
- User context
- Custom tags and context
- Performance transactions
- Release tracking
- Error sampling
- Before-send hooks for PII filtering

Usage:
    from services.common.error_tracking import init_sentry, capture_exception, add_breadcrumb
    
    # Initialize Sentry (in startup)
    init_sentry(
        dsn="https://...@sentry.io/...",
        service_name="llm_service",
        environment="production",
        release="1.0.0"
    )
    
    # Capture exception
    try:
        risky_operation()
    except Exception as e:
        capture_exception(e, extra={"operation": "risky_op"})
        raise
    
    # Add breadcrumb
    add_breadcrumb("database_query", category="db", data={"query": "SELECT ..."})
"""

import os
import logging
from typing import Optional, Dict, Any, Callable
import traceback

# Sentry imports
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.asyncio import AsyncioIntegration
    from sentry_sdk.integrations.httpx import HttpxIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    logging.warning("Sentry SDK not installed. Error tracking disabled. Install with: pip install sentry-sdk[fastapi]")

logger = logging.getLogger(__name__)

# Global state
_initialized = False


def init_sentry(
    dsn: Optional[str] = None,
    service_name: str = "unknown",
    environment: str = "development",
    release: Optional[str] = None,
    sample_rate: float = 1.0,
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1,
    send_default_pii: bool = False,
    attach_stacktrace: bool = True,
    max_breadcrumbs: int = 100,
    debug: bool = False
) -> bool:
    """
    Initialize Sentry error tracking
    
    Args:
        dsn: Sentry DSN (Data Source Name)
        service_name: Name of the service
        environment: Environment (development, staging, production)
        release: Release version
        sample_rate: Error sampling rate (0.0 to 1.0)
        traces_sample_rate: Performance tracing sample rate (0.0 to 1.0)
        profiles_sample_rate: Profiling sample rate (0.0 to 1.0)
        send_default_pii: Send personally identifiable information
        attach_stacktrace: Attach stack traces to messages
        max_breadcrumbs: Maximum number of breadcrumbs
        debug: Enable debug mode
    
    Returns:
        True if Sentry initialized successfully, False otherwise
    
    Example:
        init_sentry(
            dsn=os.getenv("SENTRY_DSN"),
            service_name="llm_service",
            environment="production",
            release="1.0.0",
            sample_rate=1.0,
            traces_sample_rate=0.1
        )
    """
    global _initialized
    
    if not SENTRY_AVAILABLE:
        logger.warning("Sentry SDK not available. Error tracking disabled.")
        return False
    
    if _initialized:
        logger.warning("Sentry already initialized")
        return True
    
    # Get DSN from environment if not provided
    if dsn is None:
        dsn = os.getenv("SENTRY_DSN")
    
    if not dsn:
        logger.warning("Sentry DSN not provided. Error tracking disabled.")
        return False
    
    try:
        # Initialize Sentry
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release,
            sample_rate=sample_rate,
            traces_sample_rate=traces_sample_rate,
            profiles_sample_rate=profiles_sample_rate,
            send_default_pii=send_default_pii,
            attach_stacktrace=attach_stacktrace,
            max_breadcrumbs=max_breadcrumbs,
            debug=debug,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                AsyncioIntegration(),
                HttpxIntegration(),
                LoggingIntegration(
                    level=logging.INFO,  # Capture info and above as breadcrumbs
                    event_level=logging.ERROR  # Send errors as events
                )
            ],
            before_send=_before_send_hook,
            before_breadcrumb=_before_breadcrumb_hook
        )
        
        # Set service name as tag
        sentry_sdk.set_tag("service", service_name)
        sentry_sdk.set_tag("hostname", os.getenv("HOSTNAME", "localhost"))
        
        _initialized = True
        logger.info(f"Sentry initialized for service: {service_name} (environment: {environment})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}", exc_info=True)
        return False


def _before_send_hook(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Hook called before sending event to Sentry
    
    Use this to filter PII, modify events, or drop events
    """
    # Example: Filter out specific exceptions
    if 'exc_info' in hint:
        exc_type, exc_value, tb = hint['exc_info']
        if isinstance(exc_value, KeyboardInterrupt):
            return None  # Don't send KeyboardInterrupt
    
    return event


def _before_breadcrumb_hook(crumb: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Hook called before adding breadcrumb
    
    Use this to filter or modify breadcrumbs
    """
    # Example: Filter out noisy breadcrumbs
    if crumb.get('category') == 'httpx' and crumb.get('data', {}).get('url', '').endswith('/health'):
        return None  # Don't track health check requests
    
    return crumb


def capture_exception(
    exception: Exception,
    level: str = "error",
    extra: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    user: Optional[Dict[str, Any]] = None,
    fingerprint: Optional[list] = None
) -> Optional[str]:
    """
    Capture an exception and send to Sentry

    Args:
        exception: Exception to capture
        level: Severity level (fatal, error, warning, info, debug)
        extra: Extra context data
        tags: Tags for grouping and filtering
        user: User context (id, email, username, ip_address)
        fingerprint: Custom fingerprint for grouping

    Returns:
        Event ID if sent, None otherwise

    Example:
        try:
            risky_operation()
        except Exception as e:
            capture_exception(
                e,
                level="error",
                extra={"operation": "risky_op", "user_id": 123},
                tags={"component": "database"},
                user={"id": 123, "email": "user@example.com"}
            )
            raise
    """
    if not SENTRY_AVAILABLE or not _initialized:
        logger.error(f"Exception: {exception}", exc_info=True)
        return None

    try:
        with sentry_sdk.push_scope() as scope:
            # Set level
            scope.level = level

            # Add extra context
            if extra:
                for key, value in extra.items():
                    scope.set_extra(key, value)

            # Add tags
            if tags:
                for key, value in tags.items():
                    scope.set_tag(key, value)

            # Set user context
            if user:
                scope.set_user(user)

            # Set custom fingerprint
            if fingerprint:
                scope.fingerprint = fingerprint

            # Capture exception
            event_id = sentry_sdk.capture_exception(exception)
            return event_id

    except Exception as e:
        logger.error(f"Failed to capture exception in Sentry: {e}", exc_info=True)
        return None


def capture_message(
    message: str,
    level: str = "info",
    extra: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """
    Capture a message and send to Sentry

    Args:
        message: Message to capture
        level: Severity level (fatal, error, warning, info, debug)
        extra: Extra context data
        tags: Tags for grouping and filtering

    Returns:
        Event ID if sent, None otherwise

    Example:
        capture_message(
            "User performed unusual action",
            level="warning",
            extra={"action": "delete_all", "user_id": 123},
            tags={"component": "user_management"}
        )
    """
    if not SENTRY_AVAILABLE or not _initialized:
        logger.log(logging.getLevelName(level.upper()), message)
        return None

    try:
        with sentry_sdk.push_scope() as scope:
            # Set level
            scope.level = level

            # Add extra context
            if extra:
                for key, value in extra.items():
                    scope.set_extra(key, value)

            # Add tags
            if tags:
                for key, value in tags.items():
                    scope.set_tag(key, value)

            # Capture message
            event_id = sentry_sdk.capture_message(message)
            return event_id

    except Exception as e:
        logger.error(f"Failed to capture message in Sentry: {e}", exc_info=True)
        return None


def add_breadcrumb(
    message: str,
    category: str = "default",
    level: str = "info",
    data: Optional[Dict[str, Any]] = None
):
    """
    Add a breadcrumb for debugging

    Args:
        message: Breadcrumb message
        category: Category (http, db, navigation, etc.)
        level: Severity level (fatal, error, warning, info, debug)
        data: Additional data

    Example:
        add_breadcrumb(
            "Database query executed",
            category="db",
            level="info",
            data={"query": "SELECT * FROM users", "duration_ms": 45}
        )
    """
    if not SENTRY_AVAILABLE or not _initialized:
        return

    try:
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {}
        )
    except Exception as e:
        logger.error(f"Failed to add breadcrumb: {e}")


def set_user(user_id: Optional[str] = None, email: Optional[str] = None,
             username: Optional[str] = None, ip_address: Optional[str] = None,
             **extra):
    """
    Set user context for error tracking

    Args:
        user_id: User ID
        email: User email
        username: Username
        ip_address: IP address
        **extra: Additional user data

    Example:
        set_user(user_id="123", email="user@example.com", subscription="premium")
    """
    if not SENTRY_AVAILABLE or not _initialized:
        return

    try:
        user_data = {}
        if user_id:
            user_data["id"] = user_id
        if email:
            user_data["email"] = email
        if username:
            user_data["username"] = username
        if ip_address:
            user_data["ip_address"] = ip_address
        user_data.update(extra)

        sentry_sdk.set_user(user_data)
    except Exception as e:
        logger.error(f"Failed to set user context: {e}")


def set_tag(key: str, value: str):
    """
    Set a tag for filtering and grouping

    Args:
        key: Tag key
        value: Tag value

    Example:
        set_tag("tenant_id", "acme-corp")
    """
    if not SENTRY_AVAILABLE or not _initialized:
        return

    try:
        sentry_sdk.set_tag(key, value)
    except Exception as e:
        logger.error(f"Failed to set tag: {e}")


def set_context(name: str, context: Dict[str, Any]):
    """
    Set custom context for debugging

    Args:
        name: Context name
        context: Context data

    Example:
        set_context("request", {
            "method": "POST",
            "url": "/api/users",
            "body_size": 1024
        })
    """
    if not SENTRY_AVAILABLE or not _initialized:
        return

    try:
        sentry_sdk.set_context(name, context)
    except Exception as e:
        logger.error(f"Failed to set context: {e}")


def start_transaction(name: str, op: str = "http.server") -> Any:
    """
    Start a performance transaction

    Args:
        name: Transaction name
        op: Operation type (http.server, db.query, etc.)

    Returns:
        Transaction object (use as context manager)

    Example:
        with start_transaction("process_request", op="task"):
            process()
    """
    if not SENTRY_AVAILABLE or not _initialized:
        # Return dummy context manager
        class DummyTransaction:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyTransaction()

    try:
        return sentry_sdk.start_transaction(name=name, op=op)
    except Exception as e:
        logger.error(f"Failed to start transaction: {e}")
        class DummyTransaction:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyTransaction()


def flush(timeout: float = 2.0) -> bool:
    """
    Flush pending events to Sentry

    Args:
        timeout: Timeout in seconds

    Returns:
        True if flushed successfully

    Example:
        # Before shutdown
        flush(timeout=5.0)
    """
    if not SENTRY_AVAILABLE or not _initialized:
        return False

    try:
        return sentry_sdk.flush(timeout=timeout)
    except Exception as e:
        logger.error(f"Failed to flush Sentry events: {e}")
        return False
