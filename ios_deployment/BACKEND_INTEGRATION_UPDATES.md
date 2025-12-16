# 🔧 Backend Integration Updates for iOS Deployment

**Critical updates needed to make the backend fully iOS-ready**

---

## 📋 OVERVIEW

This document lists all the specific code changes needed in the existing backend to support iOS deployment. These are **production-critical** updates that must be applied before deploying to production with iOS support.

---

## 🚨 CRITICAL UPDATES REQUIRED

### 1. API Gateway CORS Configuration

**File:** `services/api_gateway/main.py`  
**Lines:** 65-72  
**Priority:** CRITICAL

**Current Code:**
```python
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Updated Code:**
```python
import os
from typing import List

# Load CORS origins from environment (iOS-ready)
ALLOWED_ORIGINS: List[str] = os.getenv(
    "CORS_ORIGINS",
    "https://releaf.ai,https://www.releaf.ai,https://app.releaf.ai,capacitor://localhost,ionic://localhost"
).split(",")

# CORS middleware (iOS-ready)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Request-ID",
        "User-Agent",
        "Accept",
        "Accept-Language",
        "Cache-Control"
    ],
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
        "X-Request-ID"
    ],
    max_age=3600,
)
```

**Why:** iOS apps use `capacitor://localhost` and `ionic://localhost` as origins. The wildcard `*` won't work with credentials.

---

### 2. Environment Variables Configuration

**File:** `.env.example`  
**Lines:** 84-85  
**Priority:** CRITICAL

**Current Code:**
```bash
# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

**Updated Code:**
```bash
# CORS (iOS-ready)
CORS_ORIGINS=https://releaf.ai,https://www.releaf.ai,https://app.releaf.ai,capacitor://localhost,ionic://localhost,http://localhost:3000,http://localhost:8080
```

**Why:** Must include iOS app origins for production deployment.

---

### 3. Kubernetes ConfigMap CORS

**File:** `k8s/configmaps/app-config.yaml`  
**Lines:** 66-67  
**Priority:** CRITICAL

**Current Code:**
```yaml
  # CORS
  CORS_ORIGINS: "*"
```

**Updated Code:**
```yaml
  # CORS (iOS-ready)
  CORS_ORIGINS: "https://releaf.ai,https://www.releaf.ai,https://app.releaf.ai,capacitor://localhost,ionic://localhost"
```

**Why:** Production Kubernetes deployment must have explicit CORS origins.

---

### 4. Kubernetes Ingress CORS

**File:** `k8s/networking/ingress.yaml`  
**Lines:** 28-32  
**Priority:** CRITICAL

**Current Code:**
```yaml
    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "*"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "DNT,X-CustomHeader,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Authorization"
```

**Updated Code:**
```yaml
    # CORS (iOS-ready)
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://releaf.ai,https://www.releaf.ai,https://app.releaf.ai,capacitor://localhost,ionic://localhost"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "Content-Type,Authorization,X-API-Key,X-Request-ID,User-Agent,Accept,Accept-Language,Cache-Control"
    nginx.ingress.kubernetes.io/cors-expose-headers: "X-RateLimit-Limit,X-RateLimit-Remaining,X-RateLimit-Reset,Retry-After,X-Request-ID"
    nginx.ingress.kubernetes.io/cors-allow-credentials: "true"
    nginx.ingress.kubernetes.io/cors-max-age: "3600"
```

**Why:** Ingress-level CORS must match application-level CORS for iOS apps.

---

### 5. Rate Limiting Tier Support

**File:** `services/api_gateway/middleware/rate_limit.py`  
**Lines:** 89-100  
**Priority:** HIGH

**Add after line 88:**
```python
    def _get_rate_limit_tier(self, request: Request) -> Tuple[int, int]:
        """
        Determine rate limit tier based on API key or user tier
        
        Returns:
            Tuple of (requests_per_minute, burst_size)
        """
        # Check for premium tier (from API key validation)
        api_key = request.headers.get("X-API-Key", "")
        user_tier = request.state.user_tier if hasattr(request.state, "user_tier") else "standard"
        
        if user_tier == "premium" or api_key.startswith("premium_"):
            return (500, 100)  # Premium: 500 req/min
        elif user_tier == "enterprise" or api_key.startswith("enterprise_"):
            return (1000, 200)  # Enterprise: 1000 req/min
        else:
            return (100, 20)  # Standard: 100 req/min
```

**Update dispatch method (line 89):**
```python
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting (thread-safe)"""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get client identifier (IP address or API key)
        client_ip = self._get_client_ip(request)
        api_key = request.headers.get("X-API-Key", "")
        client_id = api_key if api_key else client_ip
        
        # Get rate limit tier
        requests_per_minute, burst_size = self._get_rate_limit_tier(request)

        # Get or create bucket for this client (thread-safe)
        async with self.buckets_lock:
            if client_id not in self.buckets:
                self.buckets[client_id] = TokenBucket(
                    capacity=burst_size,
                    refill_rate=requests_per_minute / 60.0
                )
            bucket = self.buckets[client_id]
```

**Why:** iOS apps need tier-based rate limiting (standard vs premium users).

---

### 6. User-Agent Logging

**File:** `services/api_gateway/main.py`  
**Add after line 76:**

```python
# User-Agent tracking middleware
@app.middleware("http")
async def log_user_agent(request: Request, call_next):
    """Log User-Agent for iOS analytics"""
    user_agent = request.headers.get("User-Agent", "Unknown")
    
    # Track iOS SDK usage
    if "ReleAF-iOS-SDK" in user_agent:
        logger.info(f"iOS SDK request: {user_agent}", extra={
            "user_agent": user_agent,
            "path": request.url.path,
            "method": request.method
        })
    
    response = await call_next(request)
    return response
```

**Why:** Track iOS SDK usage for analytics and debugging.

---

## ⚠️ RECOMMENDED UPDATES

### 7. Add iOS-Specific Health Check Endpoint

**File:** `services/api_gateway/main.py`  
**Add after health_check endpoint:**

```python
@app.get("/health/ios", response_model=Dict[str, Any])
async def ios_health_check():
    """iOS-specific health check with client info"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ios_support": True,
        "api_version": "v1",
        "features": {
            "chat": True,
            "vision": True,
            "organization_search": True,
            "offline_support": True
        },
        "rate_limits": {
            "standard": {"requests_per_minute": 100, "burst": 20},
            "premium": {"requests_per_minute": 500, "burst": 100}
        }
    }
```

**Why:** iOS apps can check backend capabilities and rate limits.

---

### 8. Add Request ID Middleware

**File:** `services/api_gateway/main.py`  
**Add after line 76:**

```python
# Request ID middleware for tracing
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID for tracing"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    # Set correlation ID for structured logging
    set_correlation_id(request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

**Why:** iOS SDK can track requests end-to-end for debugging.

---

## 📝 CONFIGURATION FILES TO UPDATE

### 9. Production Config Template

**Create:** `config/production_ios.yaml`

```yaml
# Copy from ios_deployment/production_config.yaml
# This is the production configuration for iOS deployment
```

**Why:** Centralized iOS production configuration.

---

### 10. Docker Compose for iOS Testing

**Update:** `docker-compose.yml`  
**Add environment variables:**

```yaml
environment:
  - CORS_ORIGINS=https://releaf.ai,capacitor://localhost,ionic://localhost
  - RATE_LIMIT_STANDARD=100
  - RATE_LIMIT_PREMIUM=500
```

**Why:** Local testing with iOS-like configuration.

---

## ✅ VALIDATION CHECKLIST

After applying all updates, validate:

- [ ] CORS origins include iOS app origins
- [ ] Rate limiting supports multiple tiers
- [ ] User-Agent logging is working
- [ ] Request ID tracking is working
- [ ] Health check endpoints return correct data
- [ ] All environment variables are set
- [ ] Kubernetes manifests are updated
- [ ] Docker Compose is updated
- [ ] Documentation is updated

---

## 🚀 DEPLOYMENT STEPS

1. **Apply code changes** (updates 1-8)
2. **Update environment variables** (update 2)
3. **Update Kubernetes manifests** (updates 3-4)
4. **Test locally** with Docker Compose
5. **Deploy to staging** with blue-green strategy
6. **Run iOS deployment simulation**
7. **Monitor metrics** for 24 hours
8. **Deploy to production**

---

## 📊 EXPECTED IMPACT

After applying these updates:

✅ **iOS apps can connect** with proper CORS  
✅ **Rate limiting works** for different user tiers  
✅ **User-Agent tracking** for analytics  
✅ **Request tracing** for debugging  
✅ **Health checks** provide iOS-specific info  
✅ **Production-ready** for Digital Ocean deployment

---

**Status:** READY FOR IMPLEMENTATION  
**Priority:** CRITICAL  
**Estimated Time:** 2-3 hours  
**Risk Level:** LOW (all changes are additive)

---

**Last Updated:** 2025-12-15  
**Version:** 1.0.0

