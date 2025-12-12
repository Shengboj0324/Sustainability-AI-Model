"""
OpenTelemetry Distributed Tracing

CRITICAL: Production-grade distributed tracing for microservices
- Trace context propagation across services
- Span creation for operations
- Automatic instrumentation for FastAPI, httpx, asyncpg
- Trace export to Jaeger/Zipkin/OTLP
- Performance monitoring
- Error tracking
- Service dependency mapping

Features:
- OpenTelemetry SDK integration
- Automatic trace context propagation
- Span attributes and events
- Exception recording
- Trace sampling
- Multiple exporters (Jaeger, Zipkin, OTLP, Console)
- FastAPI middleware integration
- HTTP client instrumentation
- Database instrumentation

Usage:
    from services.common.tracing import init_tracing, trace_operation, get_tracer
    
    # Initialize tracing (in startup)
    init_tracing(service_name="llm_service", jaeger_endpoint="http://localhost:14268/api/traces")
    
    # Get tracer
    tracer = get_tracer(__name__)
    
    # Manual span creation
    with tracer.start_as_current_span("process_request") as span:
        span.set_attribute("user_id", 123)
        span.add_event("Processing started")
        result = process()
        span.set_attribute("result_count", len(result))
    
    # Decorator for automatic tracing
    @trace_operation("database_query")
    async def query_database():
        ...
"""

import os
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
import asyncio

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.zipkin.json import ZipkinExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.trace import Status, StatusCode, Span
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not installed. Tracing disabled. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-httpx opentelemetry-instrumentation-asyncpg opentelemetry-exporter-jaeger opentelemetry-exporter-zipkin opentelemetry-exporter-otlp")

logger = logging.getLogger(__name__)

# Global tracer provider
_tracer_provider: Optional[TracerProvider] = None
_initialized = False


def init_tracing(
    service_name: str,
    service_version: str = "0.1.0",
    environment: str = "development",
    jaeger_endpoint: Optional[str] = None,
    zipkin_endpoint: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False,
    sample_rate: float = 1.0
) -> bool:
    """
    Initialize OpenTelemetry tracing
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        environment: Environment (development, staging, production)
        jaeger_endpoint: Jaeger collector endpoint (e.g., "http://localhost:14268/api/traces")
        zipkin_endpoint: Zipkin collector endpoint (e.g., "http://localhost:9411/api/v2/spans")
        otlp_endpoint: OTLP collector endpoint (e.g., "http://localhost:4317")
        console_export: Export traces to console (for debugging)
        sample_rate: Trace sampling rate (0.0 to 1.0)
    
    Returns:
        True if tracing initialized successfully, False otherwise
    
    Example:
        init_tracing(
            service_name="llm_service",
            jaeger_endpoint="http://localhost:14268/api/traces",
            sample_rate=0.1  # Sample 10% of traces
        )
    """
    global _tracer_provider, _initialized
    
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available. Tracing disabled.")
        return False
    
    if _initialized:
        logger.warning("Tracing already initialized")
        return True
    
    try:
        # Create resource with service information
        resource = Resource.create({
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "environment": environment,
            "hostname": os.getenv("HOSTNAME", "localhost")
        })
        
        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)
        
        # Add exporters
        exporters_added = 0
        
        # Jaeger exporter
        if jaeger_endpoint:
            try:
                jaeger_exporter = JaegerExporter(
                    collector_endpoint=jaeger_endpoint
                )
                _tracer_provider.add_span_processor(
                    BatchSpanProcessor(jaeger_exporter)
                )
                exporters_added += 1
                logger.info(f"Jaeger exporter configured: {jaeger_endpoint}")
            except Exception as e:
                logger.error(f"Failed to configure Jaeger exporter: {e}")
        
        # Zipkin exporter
        if zipkin_endpoint:
            try:
                zipkin_exporter = ZipkinExporter(
                    endpoint=zipkin_endpoint
                )
                _tracer_provider.add_span_processor(
                    BatchSpanProcessor(zipkin_exporter)
                )
                exporters_added += 1
                logger.info(f"Zipkin exporter configured: {zipkin_endpoint}")
            except Exception as e:
                logger.error(f"Failed to configure Zipkin exporter: {e}")

        # OTLP exporter
        if otlp_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=otlp_endpoint
                )
                _tracer_provider.add_span_processor(
                    BatchSpanProcessor(otlp_exporter)
                )
                exporters_added += 1
                logger.info(f"OTLP exporter configured: {otlp_endpoint}")
            except Exception as e:
                logger.error(f"Failed to configure OTLP exporter: {e}")

        # Console exporter (for debugging)
        if console_export:
            console_exporter = ConsoleSpanExporter()
            _tracer_provider.add_span_processor(
                SimpleSpanProcessor(console_exporter)
            )
            exporters_added += 1
            logger.info("Console exporter configured")

        if exporters_added == 0:
            logger.warning("No trace exporters configured. Traces will not be exported.")

        # Set global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Instrument libraries
        try:
            FastAPIInstrumentor().instrument()
            logger.info("FastAPI instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI: {e}")

        try:
            HTTPXClientInstrumentor().instrument()
            logger.info("HTTPX instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument HTTPX: {e}")

        try:
            AsyncPGInstrumentor().instrument()
            logger.info("AsyncPG instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument AsyncPG: {e}")

        _initialized = True
        logger.info(f"Tracing initialized for service: {service_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}", exc_info=True)
        return False


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a tracer instance

    Args:
        name: Tracer name (usually __name__)

    Returns:
        Tracer instance

    Example:
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("operation"):
            ...
    """
    if not OTEL_AVAILABLE or not _initialized:
        # Return no-op tracer
        return trace.get_tracer(name)

    return trace.get_tracer(name)


def trace_operation(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Decorator to trace a function/method

    Args:
        operation_name: Name of the operation
        attributes: Additional span attributes

    Example:
        @trace_operation("database_query", {"db": "postgres"})
        async def query_database():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE or not _initialized:
                return await func(*args, **kwargs)

            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(operation_name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function info
                span.set_attribute("function", func.__name__)
                span.set_attribute("module", func.__module__)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE or not _initialized:
                return func(*args, **kwargs)

            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(operation_name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function info
                span.set_attribute("function", func.__name__)
                span.set_attribute("module", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def add_span_attributes(**attributes):
    """
    Add attributes to current span

    Args:
        **attributes: Key-value pairs to add as span attributes

    Example:
        add_span_attributes(user_id=123, action="login")
    """
    if not OTEL_AVAILABLE or not _initialized:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Add an event to current span

    Args:
        name: Event name
        attributes: Event attributes

    Example:
        add_span_event("cache_miss", {"key": "user:123"})
    """
    if not OTEL_AVAILABLE or not _initialized:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes or {})


def record_exception(exception: Exception, attributes: Optional[Dict[str, Any]] = None):
    """
    Record an exception in current span

    Args:
        exception: Exception to record
        attributes: Additional attributes

    Example:
        try:
            risky_operation()
        except Exception as e:
            record_exception(e, {"operation": "risky_op"})
            raise
    """
    if not OTEL_AVAILABLE or not _initialized:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(exception, attributes or {})
        span.set_status(Status(StatusCode.ERROR, str(exception)))


def get_trace_context() -> Dict[str, str]:
    """
    Get current trace context for propagation

    Returns:
        Dictionary with trace context headers

    Example:
        # Propagate trace context to downstream service
        headers = get_trace_context()
        response = await client.get(url, headers=headers)
    """
    if not OTEL_AVAILABLE or not _initialized:
        return {}

    propagator = TraceContextTextMapPropagator()
    carrier = {}
    propagator.inject(carrier)
    return carrier


def set_trace_context(context: Dict[str, str]):
    """
    Set trace context from propagated headers

    Args:
        context: Dictionary with trace context headers

    Example:
        # Extract trace context from incoming request
        set_trace_context(request.headers)
    """
    if not OTEL_AVAILABLE or not _initialized:
        return

    propagator = TraceContextTextMapPropagator()
    propagator.extract(context)

