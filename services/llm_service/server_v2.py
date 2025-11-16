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
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import hashlib

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


# Rate limiting
class RateLimiter:
    """
    Simple in-memory rate limiter

    CRITICAL: Prevents DoS attacks - LLM is expensive
    """
    def __init__(self, max_requests: int = 50, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def check_rate_limit(self, client_ip: str) -> bool:
        """Check if request is within rate limit"""
        async with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window_seconds)

            # Remove old requests
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if req_time > cutoff
            ]

            # Check limit
            if len(self.requests[client_ip]) >= self.max_requests:
                return False

            # Add current request
            self.requests[client_ip].append(now)
            return True


# Request cache
class RequestCache:
    """
    LRU cache with TTL for LLM requests

    CRITICAL: LLM inference is expensive - cache aggressively
    """
    def __init__(self, max_size: int = 500, ttl_seconds: int = 600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get cached result"""
        async with self.lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    return result
                else:
                    del self.cache[key]
            return None


    def clear(self):
        """Clear cache"""
        self.cache.clear()


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

        # Performance tracking
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0

        logger.info("LLMServiceV2 initialized")

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

    def _setup_device(self) -> torch.device:
        """Setup device with proper CUDA handling"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for inference")

        return device

    async def initialize(self):
        """Load model and tokenizer"""
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
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

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

            # Update stats
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
    await llm_service.initialize()


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown"""
    await llm_service.close()


# Helper function to create cache key
def create_cache_key(request: LLMRequest) -> str:
    """Create cache key from request"""
    # Hash messages + context
    content = str(request.messages) + str(request.context)
    return hashlib.md5(content.encode()).hexdigest()


# API Endpoints
@app.post("/generate", response_model=LLMResponse)
async def generate_text(request: LLMRequest, http_request: Request):
    """
    Generate text endpoint

    CRITICAL: LLM inference is expensive - use rate limiting and caching
    """
    ACTIVE_REQUESTS.inc()
    endpoint = "generate"
    start_time = time.time()

    try:
        # Get client IP
        client_ip = http_request.client.host

        # Check rate limit
        if not await rate_limiter.check_rate_limit(client_ip):
            REQUESTS_TOTAL.labels(endpoint=endpoint, status="rate_limited").inc()
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Check cache
        cache_key = create_cache_key(request)
        cached_result = await request_cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for {cache_key[:16]}...")
            REQUESTS_TOTAL.labels(endpoint=endpoint, status="cache_hit").inc()
            return cached_result

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
            cached=False
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

        return llm_response

    except HTTPException:
        raise
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        logger.error(f"Generation failed: {e}", exc_info=True)
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


@app.get("/health")
async def health():
    """
    Health check endpoint for load balancer

    Returns detailed health status for monitoring
    """
    is_healthy = (
        llm_service.model is not None and
        not llm_service._shutdown
    )

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "llm_v2",
        "version": "2.0.0",
        "model_loaded": llm_service.model is not None,
        "tokenizer_loaded": llm_service.tokenizer is not None,
        "device": str(llm_service.device) if llm_service.device else "none",
        "shutdown": llm_service._shutdown,
        "cache_size": len(request_cache.cache)
    }


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

