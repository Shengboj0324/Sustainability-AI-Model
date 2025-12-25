"""
LLM Service V2 - Production-grade domain-specialized language model

CRITICAL FEATURES:
- Rate limiting (50 req/min per IP for LLM)
- Request caching (LRU + TTL)
- Prometheus metrics
- Timeouts on all operations
- Graceful shutdown
- CORS for web + iOS
- Comprehensive error handling
- Token usage tracking
- Model warmup
"""

import asyncio
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml

# CRITICAL FIX: Make transformers optional to handle x86/ARM environment issues
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_ERROR = str(e)
    # Create dummy classes for type hints
    class AutoModelForCausalLM:
        pass
    class AutoTokenizer:
        pass
    class PeftModel:
        pass

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

# Import shared utilities - CRITICAL: Single source of truth
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.utils import RateLimiter, RequestCache

# Import monitoring components
from common.structured_logging import get_logger, log_context, set_correlation_id
from common.health_checks import HealthChecker, HealthStatus
from common.alerting import init_alerting, send_alert, AlertSeverity
from common.circuit_breaker import CircuitBreaker
from common.environment import check_environment, get_optimal_device

# Try to import optional monitoring components
try:
    from common.tracing import init_tracing, trace_operation
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    def trace_operation(name, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from common.error_tracking import init_sentry, capture_exception, add_breadcrumb
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    def capture_exception(exc, **kwargs):
        pass
    def add_breadcrumb(msg, **kwargs):
        pass

# Import NLP modules - CRITICAL FIX: Use relative imports
from .intent_classifier import IntentClassifier, IntentCategory
from .entity_extractor import EntityExtractor, Entity
from .language_handler import LanguageHandler, Language

# Configure structured logging
logger = get_logger(__name__)

# CRITICAL: Check environment and warn about issues
env_info = check_environment(raise_on_issues=False)
if not TRANSFORMERS_AVAILABLE:
    logger.error(
        "⚠️  TRANSFORMERS NOT AVAILABLE - LLM service will run in degraded mode\n"
        f"   Error: {TRANSFORMERS_ERROR}\n"
        "   The service will start but LLM endpoints will return errors.\n"
        "   See ENVIRONMENT_FIX_GUIDE.md for solutions."
    )

# Prometheus metrics
REQUESTS_TOTAL = Counter('llm_requests_total', 'Total LLM requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('llm_request_duration_seconds', 'Request duration', ['endpoint'])
ACTIVE_REQUESTS = Gauge('llm_active_requests', 'Active requests')
GENERATION_TIME = Histogram('llm_generation_time_ms', 'Generation time in ms')
TOKENS_GENERATED = Counter('llm_tokens_generated_total', 'Total tokens generated')
PROMPT_TOKENS = Histogram('llm_prompt_tokens', 'Prompt tokens')
COMPLETION_TOKENS = Histogram('llm_completion_tokens', 'Completion tokens')

# Initialize FastAPI app
app = FastAPI(
    title="ReleAF AI LLM Service V2",
    description="Production-grade domain-specialized language model service",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for web and iOS clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize monitoring components
health_checker = HealthChecker(service_name="llm_service", check_timeout=5.0)
alert_manager = None  # Initialized in startup
model_circuit_breaker = CircuitBreaker(
    name="llm_model",
    failure_threshold=3,
    recovery_timeout=60.0,
    expected_exception=Exception
)


# Request/Response Models
class LLMRequest(BaseModel):
    """LLM request"""
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context from other services")
    max_tokens: int = Field(512, ge=1, le=2048, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling")
    stream: bool = Field(False, description="Stream response (not implemented)")


class LLMResponse(BaseModel):
    """LLM response"""
    response: str
    usage: Dict[str, int]
    model: str
    generation_time_ms: float
    cached: bool = False
    # NLP metadata
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    entities: Optional[List[Dict[str, Any]]] = None

# REMOVED: RateLimiter and RequestCache now imported from shared.utils
# This eliminates code duplication and ensures single source of truth


# Initialize components
rate_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "50")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60"))
)

request_cache = RequestCache(
    max_size=int(os.getenv("CACHE_MAX_SIZE", "500")),
    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "600"))
)


class LLMServiceV2:
    """
    Production-grade LLM service

    CRITICAL: Handles expensive LLM inference with proper resource management
    """
    def __init__(self, config_path: str = "configs/llm_sft.yaml"):
        self.config = self._load_config(config_path)
        self.device = None
        self.model = None
        self.tokenizer = None
        self.system_prompt = self.config.get("system_prompt", "You are a helpful sustainability assistant.")
        self._shutdown = False

        # Performance tracking (thread-safe)
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        self.stats_lock = asyncio.Lock()

        # Initialize NLP modules
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.language_handler = LanguageHandler()

        logger.info("LLMServiceV2 initialized with NLP modules")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}, using defaults")
            return {
                "model": {"base_model_name": "meta-llama/Llama-3-8B"},
                "training": {"output_dir": "checkpoints/llm_sft", "bf16": True},
                "data": {"max_length": 2048}
            }

    def _setup_device(self):
        """
        Setup device with proper CUDA and MPS handling

        CRITICAL FIX: Handle missing torch gracefully
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Torch not available - skipping device setup")
            return None

        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🔥 CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("🍎 Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("💻 Using CPU for inference")

        return device

    async def initialize(self):
        """
        Load model and tokenizer

        CRITICAL FIX: Gracefully degrade if transformers not available
        """
        # Check if transformers is available
        if not TRANSFORMERS_AVAILABLE:
            logger.error(
                "⚠️  Cannot initialize LLM - transformers not available\n"
                f"   Error: {TRANSFORMERS_ERROR}\n"
                "   Service will run in degraded mode (health checks only)\n"
                "   See ENVIRONMENT_FIX_GUIDE.md for solutions"
            )
            self.model = None
            self.tokenizer = None
            self.device = None
            return

        try:
            logger.info("Loading LLM model...")
            start_time = time.time()

            # Setup device
            self.device = self._setup_device()

            base_model_name = self.config["model"]["base_model_name"]
            adapter_path = self.config["training"]["output_dir"]

            # Load tokenizer
            logger.info(f"Loading tokenizer: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model with timeout
            logger.info(f"Loading base model: {base_model_name}")
            self.model = await asyncio.wait_for(
                asyncio.to_thread(
                    AutoModelForCausalLM.from_pretrained,
                    base_model_name,
                    torch_dtype=torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                ),
                timeout=300.0  # 5 min timeout for model loading
            )

            # Load LoRA adapter if available
            try:
                logger.info(f"Loading LoRA adapter from: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                self.model = self.model.merge_and_unload()  # Merge for faster inference
                logger.info("LoRA adapter loaded and merged")
            except Exception as e:
                logger.warning(f"Could not load adapter: {e}. Using base model only.")

            # Set to eval mode
            self.model.eval()

            # Model warmup
            await self._warmup_model()

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")

        except asyncio.TimeoutError:
            logger.error("Model loading timeout")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    async def _warmup_model(self):
        """Warmup model for consistent latency"""
        logger.info("Warming up model...")
        try:
            dummy_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "Hello"}
            ]

            for i in range(3):
                _ = await asyncio.to_thread(
                    self._generate_sync,
                    dummy_messages,
                    max_tokens=10,
                    temperature=0.7,
                    top_p=0.9
                )
                # Synchronize GPU operations
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                elif self.device.type == "mps":
                    torch.mps.synchronize()

            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")


    async def preprocess_with_nlp(self, user_query: str) -> Dict[str, Any]:
        """
        Preprocess user query with NLP modules

        Args:
            user_query: User input text

        Returns:
            Dictionary with NLP metadata
        """
        try:
            # Detect language
            detected_lang, lang_confidence = self.language_handler.detect_language(user_query)
            logger.info(f"Detected language: {detected_lang.value} ({lang_confidence:.2f})")

            # Translate to English if needed
            query_en = user_query
            if detected_lang != Language.ENGLISH:
                query_en = self.language_handler.translate_to_english(user_query, detected_lang)
                logger.info(f"Translated to English: {query_en[:100]}...")

            # Classify intent
            intent, intent_confidence = self.intent_classifier.classify(query_en)
            logger.info(f"Intent: {intent.value} ({intent_confidence:.2f})")

            # Extract entities
            entities = self.entity_extractor.extract(query_en)
            logger.info(f"Extracted {len(entities)} entities")

            # Get context hints for LLM
            context_hints = self.intent_classifier.get_context_hints(intent)

            return {
                "detected_language": detected_lang.value,
                "language_confidence": lang_confidence,
                "intent": intent.value,
                "intent_confidence": intent_confidence,
                "entities": [
                    {
                        "text": e.text,
                        "type": e.type,
                        "start": e.start,
                        "end": e.end,
                        "confidence": e.confidence
                    }
                    for e in entities
                ],
                "context_hints": context_hints,
                "query_en": query_en
            }

        except Exception as e:
            logger.error(f"NLP preprocessing failed: {e}", exc_info=True)
            # Return minimal metadata on error
            return {
                "detected_language": "en",
                "language_confidence": 0.5,
                "intent": "general_question",
                "intent_confidence": 0.5,
                "entities": [],
                "context_hints": {},
                "query_en": user_query
            }

    def _format_messages(self, messages: List[Dict[str, str]], context: Optional[Dict] = None) -> str:
        """Format messages for the model"""
        # Add system prompt if not present
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Add context if provided
        if context:
            context_str = self._format_context(context)
            if context_str:
                messages.insert(1, {"role": "system", "content": context_str})

        # Use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                formatted += f"<|{role}|>\n{content}\n"
            formatted += "<|assistant|>\n"
            return formatted

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information"""
        parts = []

        # NLP metadata
        if "nlp_metadata" in context:
            nlp = context["nlp_metadata"]
            intent = nlp.get("intent", "")
            entities = nlp.get("entities", [])

            if intent:
                parts.append(f"User intent: {intent}")

            if entities:
                entity_strs = [f"{e['text']} ({e['type']})" for e in entities[:5]]  # Limit to 5
                parts.append(f"Key entities: {', '.join(entity_strs)}")

            # Add context hints
            hints = nlp.get("context_hints", {})
            if hints.get("response_style"):
                parts.append(f"Response style: {hints['response_style']}")

        # Vision results
        if "vision" in context:
            vision = context["vision"]
            parts.append(f"Image analysis: {vision}")

        # RAG results
        if "rag" in context:
            rag = context["rag"]
            parts.append(f"Relevant information: {rag}")

        # KG results
        if "kg" in context:
            kg = context["kg"]
            parts.append(f"Related concepts: {kg}")

        return "\n\n".join(parts) if parts else ""

    def _generate_sync(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        context: Optional[Dict] = None
    ) -> Tuple[str, int, int]:
        """
        Synchronous generation (called from async context)

        Returns: (response, prompt_tokens, completion_tokens)
        """
        with torch.inference_mode():
            # Format input
            prompt = self._format_messages(messages, context)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["data"]["max_length"]
            ).to(self.device)

            prompt_tokens = inputs["input_ids"].shape[1]

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            completion_tokens = outputs.shape[1] - prompt_tokens

            return response.strip(), prompt_tokens, completion_tokens

    async def generate(
        self,
        request: LLMRequest,
        timeout: float = 60.0
    ) -> Tuple[str, int, int, float]:
        """
        Generate response with timeout

        Returns: (response, prompt_tokens, completion_tokens, generation_time_ms)
        """
        try:
            start_time = time.time()

            # Generate in thread pool to avoid blocking
            response, prompt_tokens, completion_tokens = await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_sync,
                    request.messages,
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                    request.context
                ),
                timeout=timeout
            )

            generation_time = (time.time() - start_time) * 1000

            # Update stats (thread-safe)
            async with self.stats_lock:
                self.total_requests += 1
                self.total_tokens_generated += completion_tokens
                self.total_generation_time += generation_time

            return response, prompt_tokens, completion_tokens, generation_time

        except asyncio.TimeoutError:
            logger.error(f"Generation timeout after {timeout}s")
            raise HTTPException(status_code=504, detail="Generation timeout")
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise

    async def close(self):
        """Graceful shutdown"""
        self._shutdown = True
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.device and self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("LLM service shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        avg_time = self.total_generation_time / self.total_requests if self.total_requests > 0 else 0
        avg_tokens = self.total_tokens_generated / self.total_requests if self.total_requests > 0 else 0

        return {
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "total_generation_time_ms": self.total_generation_time,
            "average_generation_time_ms": avg_time,
            "average_tokens_per_request": avg_tokens,
            "device": str(self.device) if self.device else "none",
            "model_loaded": self.model is not None
        }


# Initialize service
llm_service = LLMServiceV2()



# Lifecycle hooks
@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    global alert_manager

    logger.info("Starting LLM Service V2", service="llm_service", version="2.0.0")

    # Initialize distributed tracing
    if TRACING_AVAILABLE:
        try:
            jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
            if jaeger_endpoint:
                init_tracing(
                    service_name="llm_service",
                    service_version="2.0.0",
                    environment=os.getenv("ENVIRONMENT", "development"),
                    jaeger_endpoint=jaeger_endpoint,
                    sample_rate=float(os.getenv("TRACE_SAMPLE_RATE", "0.1"))
                )
                logger.info("Distributed tracing initialized", jaeger_endpoint=jaeger_endpoint)
        except Exception as e:
            logger.warning("Failed to initialize tracing", error=str(e))

    # Initialize error tracking
    if SENTRY_AVAILABLE:
        try:
            sentry_dsn = os.getenv("SENTRY_DSN")
            if sentry_dsn:
                init_sentry(
                    dsn=sentry_dsn,
                    service_name="llm_service",
                    environment=os.getenv("ENVIRONMENT", "development"),
                    release=os.getenv("RELEASE_VERSION", "2.0.0"),
                    traces_sample_rate=float(os.getenv("SENTRY_TRACE_SAMPLE_RATE", "0.1"))
                )
                logger.info("Error tracking initialized", sentry_enabled=True)
        except Exception as e:
            logger.warning("Failed to initialize Sentry", error=str(e))

    # Initialize alerting
    try:
        alert_manager = init_alerting(
            slack_webhook=os.getenv("SLACK_WEBHOOK"),
            pagerduty_key=os.getenv("PAGERDUTY_KEY")
        )
        logger.info("Alerting system initialized")
    except Exception as e:
        logger.warning("Failed to initialize alerting", error=str(e))

    # Initialize LLM service
    try:
        await llm_service.initialize()
        logger.info("LLM service initialized successfully")

        # Add health checks
        async def check_model_health():
            if llm_service.model is None or llm_service.tokenizer is None:
                return {"status": "unhealthy", "reason": "Model not loaded"}
            return {"status": "healthy"}

        health_checker.add_check("llm_model", check_model_health)
        health_checker.mark_ready()
        health_checker.mark_startup_complete()

        logger.info("Health checks configured")
    except Exception as e:
        logger.error("Failed to initialize LLM service", exc_info=True)
        capture_exception(e, extra={"component": "startup"})

        # Send critical alert
        if alert_manager:
            await send_alert(
                title="LLM Service Startup Failed",
                message=f"Failed to initialize LLM service: {str(e)}",
                severity=AlertSeverity.CRITICAL,
                service="llm_service",
                component="startup"
            )
        raise


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown"""
    logger.info("Shutting down LLM Service")
    health_checker.mark_not_ready()
    await llm_service.close()
    logger.info("LLM Service shutdown complete")


# Health check endpoints
@app.get("/health/live")
async def liveness():
    """Liveness probe - is service alive?"""
    return await health_checker.liveness()


@app.get("/health/ready")
async def readiness():
    """Readiness probe - is service ready for traffic?"""
    return await health_checker.readiness()


@app.get("/health/startup")
async def startup_probe():
    """Startup probe - has service finished initialization?"""
    return await health_checker.startup()


@app.get("/health")
async def health():
    """Detailed health check with all dependencies"""
    result = await health_checker.check_health()

    status_code = 200
    if result.status == HealthStatus.UNHEALTHY:
        status_code = 503
    elif result.status == HealthStatus.DEGRADED:
        status_code = 200

    return Response(
        content=result.model_dump_json() if hasattr(result, 'model_dump_json') else str(result),
        status_code=status_code,
        media_type="application/json"
    )


# Helper function to create cache key
def create_cache_key(request: LLMRequest) -> str:
    """
    Create deterministic cache key from request

    CRITICAL FIX: Use JSON with sorted keys for deterministic hashing
    - Prevents cache misses due to dict ordering
    - Uses SHA256 instead of MD5 (no collision risk)
    - Includes all relevant parameters
    """
    import json

    # Create deterministic representation with sorted keys
    content = json.dumps({
        "messages": request.messages,
        "context": request.context if request.context else {},
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p
    }, sort_keys=True)

    # Use SHA256 for better collision resistance
    return hashlib.sha256(content.encode()).hexdigest()


# API Endpoints
@app.post("/generate", response_model=LLMResponse)
@trace_operation("llm_generate")
async def generate_text(request: LLMRequest, http_request: Request):
    """
    Generate text endpoint

    CRITICAL: LLM inference is expensive - use rate limiting and caching
    """
    # CRITICAL FIX: Check if model is available
    if not TRANSFORMERS_AVAILABLE or llm_service.model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "LLM service unavailable",
                "message": "Transformers library not available or model not loaded. "
                          "This is likely due to x86 Python on ARM Mac. "
                          "See ENVIRONMENT_FIX_GUIDE.md for solutions.",
                "transformers_available": TRANSFORMERS_AVAILABLE,
                "model_loaded": llm_service.model is not None
            }
        )

    ACTIVE_REQUESTS.inc()
    endpoint = "generate"
    start_time = time.time()

    # Set correlation ID from request header or generate new one
    correlation_id = http_request.headers.get("X-Correlation-ID")
    set_correlation_id(correlation_id)

    try:
        # Get client IP
        client_ip = http_request.client.host

        with log_context(
            correlation_id=correlation_id,
            endpoint=endpoint,
            client_ip=client_ip
        ):
            logger.info("LLM generation request started", max_tokens=request.max_tokens)

            add_breadcrumb(
                "LLM generation request",
                category="llm",
                data={"max_tokens": request.max_tokens, "temperature": request.temperature}
            )

            # Check rate limit
            if not await rate_limiter.check_rate_limit(client_ip):
                REQUESTS_TOTAL.labels(endpoint=endpoint, status="rate_limited").inc()
                logger.warning("Rate limit exceeded", client_ip=client_ip)
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            # Preprocess with NLP if user message exists
            nlp_metadata = None
            if request.messages and len(request.messages) > 0:
                # Get last user message
                user_message = None
                for msg in reversed(request.messages):
                    if msg.get("role") == "user":
                        user_message = msg.get("content", "")
                        break

                if user_message:
                    nlp_metadata = await llm_service.preprocess_with_nlp(user_message)
                    # Add NLP context hints to request context
                    if request.context is None:
                        request.context = {}
                    request.context["nlp_metadata"] = nlp_metadata

            # Check cache
            cache_key = create_cache_key(request)
            cached_result = await request_cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {cache_key[:16]}...")
                REQUESTS_TOTAL.labels(endpoint=endpoint, status="cache_hit").inc()
                return cached_result

            # CRITICAL FIX: Validate token limits before generation
            max_context_length = llm_service.config["data"]["max_length"]

            # Estimate prompt tokens (rough estimate: 1 token ≈ 4 chars)
            prompt_text = str(request.messages) + str(request.context if request.context else "")
            estimated_prompt_tokens = len(prompt_text) // 4

            # Check if total tokens would exceed context window
            total_estimated_tokens = estimated_prompt_tokens + request.max_tokens
            if total_estimated_tokens > max_context_length:
                logger.warning(
                    "Token limit exceeded",
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    requested_max_tokens=request.max_tokens,
                    total=total_estimated_tokens,
                    max_context=max_context_length
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Token limit exceeded",
                        "estimated_prompt_tokens": estimated_prompt_tokens,
                        "requested_max_tokens": request.max_tokens,
                        "total_estimated_tokens": total_estimated_tokens,
                        "max_context_length": max_context_length,
                        "suggestion": f"Reduce max_tokens to at most {max_context_length - estimated_prompt_tokens}"
                    }
                )

            # Generate response
            response, prompt_tokens, completion_tokens, generation_time = await llm_service.generate(
                request,
                timeout=60.0
            )

            # Build response
            llm_response = LLMResponse(
                response=response,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                model=llm_service.config["model"]["base_model_name"],
                generation_time_ms=generation_time,
                cached=False,
                # Add NLP metadata
                detected_language=nlp_metadata.get("detected_language") if nlp_metadata else None,
                language_confidence=nlp_metadata.get("language_confidence") if nlp_metadata else None,
                intent=nlp_metadata.get("intent") if nlp_metadata else None,
                intent_confidence=nlp_metadata.get("intent_confidence") if nlp_metadata else None,
                entities=nlp_metadata.get("entities") if nlp_metadata else None
            )

            # Cache result
            await request_cache.set(cache_key, llm_response)

            # Update metrics
            REQUESTS_TOTAL.labels(endpoint=endpoint, status="success").inc()
            REQUEST_DURATION.labels(endpoint=endpoint).observe(time.time() - start_time)
            GENERATION_TIME.observe(generation_time)
            TOKENS_GENERATED.inc(completion_tokens)
            PROMPT_TOKENS.observe(prompt_tokens)
            COMPLETION_TOKENS.observe(completion_tokens)

            logger.info(
                "LLM generation completed",
                tokens_generated=completion_tokens,
                generation_time_ms=generation_time
            )

            return llm_response

    except HTTPException:
        logger.warning("LLM generation failed - HTTP exception", exc_info=True)
        raise
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        logger.error("LLM generation failed", exc_info=True)

        # Capture exception in Sentry
        capture_exception(
            e,
            extra={"endpoint": endpoint, "max_tokens": request.max_tokens},
            tags={"service": "llm_service", "endpoint": endpoint}
        )

        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/synthesize_decision", response_model=LLMResponse)
async def synthesize_decision(request: LLMRequest, http_request: Request):
    """Synthesize bin decision"""
    return await generate_text(request, http_request)


@app.post("/generate_ideas", response_model=LLMResponse)
async def generate_ideas(request: LLMRequest, http_request: Request):
    """Generate upcycling ideas"""
    return await generate_text(request, http_request)


@app.post("/answer_question", response_model=LLMResponse)
async def answer_question(request: LLMRequest, http_request: Request):
    """Answer sustainability question"""
    return await generate_text(request, http_request)


@app.post("/rank_and_explain", response_model=LLMResponse)
async def rank_and_explain(request: LLMRequest, http_request: Request):
    """Rank and explain organizations"""
    return await generate_text(request, http_request)


@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    return {
        "service": "llm_v2",
        "cache_size": len(request_cache.cache),
        "cache_max_size": request_cache.max_size,
        "cache_ttl_seconds": request_cache.ttl_seconds,
        "rate_limit_requests": rate_limiter.max_requests,
        "rate_limit_window": rate_limiter.window_seconds,
        "llm_stats": llm_service.get_stats()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/cache/clear")
async def clear_cache():
    """Clear request cache"""
    request_cache.clear()
    return {"status": "cache_cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_v2:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8002")),
        reload=False,
        log_level="info"
    )

