# ReleAF AI - Backend Merge Integration Guide

**Version:** 1.0.0  
**Purpose:** Guide for merging iOS deployment package with existing production backend

---

## Table of Contents

1. [Pre-Merge Checklist](#pre-merge-checklist)
2. [Repository Structure](#repository-structure)
3. [Merge Strategy](#merge-strategy)
4. [Configuration Updates](#configuration-updates)
5. [API Versioning](#api-versioning)
6. [Database Migrations](#database-migrations)
7. [Deployment Strategy](#deployment-strategy)
8. [Testing & Validation](#testing--validation)
9. [Rollback Plan](#rollback-plan)
10. [Post-Merge Monitoring](#post-merge-monitoring)

---

## Pre-Merge Checklist

### 1. Backup Current Production

```bash
# Backup database
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER $POSTGRES_DB > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup Redis
redis-cli --rdb backup_$(date +%Y%m%d_%H%M%S).rdb

# Backup Neo4j
neo4j-admin backup --backup-dir=/backups/neo4j_$(date +%Y%m%d_%H%M%S)

# Tag current production version
git tag -a v1.0.0-pre-ios -m "Pre-iOS deployment backup"
git push origin v1.0.0-pre-ios
```

### 2. Environment Preparation

```bash
# Create new branch for iOS integration
git checkout -b feature/ios-deployment

# Verify all services are healthy
kubectl get pods -n releaf-production
kubectl get svc -n releaf-production

# Check resource availability
kubectl top nodes
kubectl top pods -n releaf-production
```

### 3. Dependencies Check

```bash
# Verify Python version
python --version  # Should be 3.11+

# Verify required packages
pip list | grep -E "fastapi|uvicorn|pydantic|httpx"

# Verify Docker version
docker --version  # Should be 20.10+

# Verify Kubernetes version
kubectl version  # Should be 1.25+
```

---

## Repository Structure

### Current Production Structure

```
releaf-backend/
├── services/
│   ├── api_gateway/
│   ├── orchestrator/
│   ├── llm_service/
│   ├── rag_service/
│   ├── vision_service/
│   ├── kg_service/
│   └── org_search_service/
├── k8s/
├── docker/
├── tests/
└── docs/
```

### iOS Deployment Package Structure

```
ios_deployment/
├── ReleAFSDK.swift
├── ReleAFSDK+Network.swift
├── API_DOCUMENTATION.md
├── FRONTEND_INTEGRATION_GUIDE.md
├── PERFORMANCE_OPTIMIZATION_GUIDE.md
├── ios_deployment_simulation.py
├── production_config.yaml
└── BACKEND_MERGE_GUIDE.md
```

### Merged Structure

```
releaf-backend/
├── services/
│   ├── api_gateway/
│   │   ├── main.py  # UPDATE
│   │   ├── routers/  # UPDATE
│   │   ├── middleware/  # UPDATE
│   │   └── config/  # NEW
│   │       └── production_ios.yaml
│   └── ...
├── k8s/
│   ├── services/
│   │   └── api-gateway.yaml  # UPDATE
│   └── configmaps/
│       └── ios-config.yaml  # NEW
├── sdk/
│   └── ios/  # NEW
│       ├── ReleAFSDK.swift
│       ├── ReleAFSDK+Network.swift
│       └── README.md
├── docs/
│   ├── api/  # NEW
│   │   └── iOS_API_DOCUMENTATION.md
│   └── integration/  # NEW
│       ├── FRONTEND_INTEGRATION_GUIDE.md
│       └── PERFORMANCE_OPTIMIZATION_GUIDE.md
├── tests/
│   └── ios_deployment/  # NEW
│       └── ios_deployment_simulation.py
└── deployment/
    └── ios/  # NEW
        └── production_config.yaml
```

---

## Merge Strategy

### Step 1: Create Feature Branch

```bash
# Create and checkout feature branch
git checkout -b feature/ios-deployment

# Verify clean working directory
git status
```

### Step 2: Add iOS SDK

```bash
# Create SDK directory
mkdir -p sdk/ios

# Copy SDK files
cp ios_deployment/ReleAFSDK.swift sdk/ios/
cp ios_deployment/ReleAFSDK+Network.swift sdk/ios/

# Create SDK README
cat > sdk/ios/README.md << 'EOF'
# ReleAF AI iOS SDK

Production-ready Swift SDK for iOS integration.

## Installation

See [iOS API Documentation](../../docs/api/iOS_API_DOCUMENTATION.md)

## Usage

See [Frontend Integration Guide](../../docs/integration/FRONTEND_INTEGRATION_GUIDE.md)
EOF

# Add to git
git add sdk/ios/
git commit -m "Add iOS SDK"
```

### Step 3: Add Documentation

```bash
# Create documentation directories
mkdir -p docs/api
mkdir -p docs/integration

# Copy documentation
cp ios_deployment/API_DOCUMENTATION.md docs/api/iOS_API_DOCUMENTATION.md
cp ios_deployment/FRONTEND_INTEGRATION_GUIDE.md docs/integration/
cp ios_deployment/PERFORMANCE_OPTIMIZATION_GUIDE.md docs/integration/

# Add to git
git add docs/
git commit -m "Add iOS documentation"
```

### Step 4: Add Testing Tools

```bash
# Create test directory
mkdir -p tests/ios_deployment

# Copy simulation script
cp ios_deployment/ios_deployment_simulation.py tests/ios_deployment/

# Make executable
chmod +x tests/ios_deployment/ios_deployment_simulation.py

# Add to git
git add tests/ios_deployment/
git commit -m "Add iOS deployment simulation"
```

### Step 5: Update API Gateway Configuration

```bash
# Create config directory
mkdir -p services/api_gateway/config

# Copy production config
cp ios_deployment/production_config.yaml services/api_gateway/config/production_ios.yaml

# Add to git
git add services/api_gateway/config/
git commit -m "Add iOS production configuration"
```

---

## Configuration Updates

### 1. Update API Gateway CORS

**File:** `services/api_gateway/main.py`

```python
# BEFORE (Development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AFTER (Production with iOS)
import os
from typing import List

# Load from environment or config
ALLOWED_ORIGINS: List[str] = os.getenv(
    "ALLOWED_ORIGINS",
    "https://releaf.ai,https://www.releaf.ai,https://app.releaf.ai,capacitor://localhost,ionic://localhost"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID", "User-Agent"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
    max_age=3600,
)
```

### 2. Update Rate Limiting

**File:** `services/api_gateway/middleware/rate_limit.py`

```python
# Add tier-based rate limiting
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        burst_size: int = 20,
        premium_requests_per_minute: int = 500,
        premium_burst_size: int = 100
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.premium_requests_per_minute = premium_requests_per_minute
        self.premium_burst_size = premium_burst_size
        # ... rest of implementation
    
    async def dispatch(self, request: Request, call_next):
        # Check if user is premium (has API key)
        api_key = request.headers.get("X-API-Key")
        
        if api_key and self.is_valid_api_key(api_key):
            # Use premium limits
            limit = self.premium_requests_per_minute
            burst = self.premium_burst_size
        else:
            # Use standard limits
            limit = self.requests_per_minute
            burst = self.burst_size
        
        # ... rest of rate limiting logic
```

### 3. Add User-Agent Logging

**File:** `services/api_gateway/middleware/logging.py`

```python
# Add iOS-specific logging
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        user_agent = request.headers.get("User-Agent", "")
        
        # Detect iOS SDK
        is_ios_sdk = "ReleAF-iOS-SDK" in user_agent
        
        # Log with iOS flag
        logger.info(
            "Request",
            extra={
                "path": request.url.path,
                "method": request.method,
                "user_agent": user_agent,
                "is_ios": is_ios_sdk,
                "client_ip": request.client.host
            }
        )
        
        response = await call_next(request)
        return response
```

---

## API Versioning

### Strategy: URL-Based Versioning

Current: `/api/v1/chat`
Future: `/api/v2/chat` (when breaking changes needed)

### Backward Compatibility

```python
# services/api_gateway/main.py

# Include both v1 and v2 routers
app.include_router(chat_router_v1, prefix="/api/v1", tags=["chat-v1"])
app.include_router(chat_router_v2, prefix="/api/v2", tags=["chat-v2"])

# Default to latest version
app.include_router(chat_router_v2, prefix="/api", tags=["chat"])
```

### Version Header Support

```python
# Support version via header
@app.middleware("http")
async def version_middleware(request: Request, call_next):
    api_version = request.headers.get("X-API-Version", "v1")

    # Rewrite path based on version header
    if not request.url.path.startswith("/api/v"):
        request.scope["path"] = f"/api/{api_version}{request.url.path}"

    response = await call_next(request)
    response.headers["X-API-Version"] = api_version
    return response
```

---

## Database Migrations

### 1. Create Migration Scripts

```bash
# Create migrations directory
mkdir -p migrations/ios_deployment

# Create migration script
cat > migrations/ios_deployment/001_add_ios_users.sql << 'EOF'
-- Add iOS-specific user tracking
CREATE TABLE IF NOT EXISTS ios_users (
    id SERIAL PRIMARY KEY,
    device_id VARCHAR(255) UNIQUE NOT NULL,
    api_key VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    app_version VARCHAR(50),
    ios_version VARCHAR(50),
    device_model VARCHAR(100),
    push_token VARCHAR(255),
    preferences JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_ios_users_device_id ON ios_users(device_id);
CREATE INDEX idx_ios_users_api_key ON ios_users(api_key);
CREATE INDEX idx_ios_users_last_active ON ios_users(last_active);

-- Add iOS request tracking
CREATE TABLE IF NOT EXISTS ios_requests (
    id SERIAL PRIMARY KEY,
    device_id VARCHAR(255) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_agent TEXT,
    error_message TEXT
);

CREATE INDEX idx_ios_requests_device_id ON ios_requests(device_id);
CREATE INDEX idx_ios_requests_created_at ON ios_requests(created_at);
CREATE INDEX idx_ios_requests_endpoint ON ios_requests(endpoint);
EOF
```

### 2. Run Migrations

```bash
# Test migration on staging
psql -h $STAGING_POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/ios_deployment/001_add_ios_users.sql

# Verify tables created
psql -h $STAGING_POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c "\dt ios_*"

# Run on production (during maintenance window)
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/ios_deployment/001_add_ios_users.sql
```

---

## Deployment Strategy

### Blue-Green Deployment

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                        │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────▼────────┐                 ┌────────▼───────┐
│  Blue (v1.0)   │                 │ Green (v1.1)   │
│  Current Prod  │                 │  iOS Support   │
│  3 instances   │                 │  3 instances   │
└────────────────┘                 └────────────────┘
```

### Step-by-Step Deployment

#### Phase 1: Deploy Green Environment

```bash
# 1. Build new Docker images with iOS support
docker build -t releaf/api-gateway:v1.1-ios -f docker/api_gateway.Dockerfile .
docker build -t releaf/orchestrator:v1.1-ios -f docker/orchestrator.Dockerfile .

# 2. Push to registry
docker push releaf/api-gateway:v1.1-ios
docker push releaf/orchestrator:v1.1-ios

# 3. Deploy to green environment
kubectl apply -f k8s/services/api-gateway-green.yaml
kubectl apply -f k8s/services/orchestrator-green.yaml

# 4. Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=api-gateway,version=green --timeout=300s

# 5. Verify health
kubectl exec -it $(kubectl get pod -l app=api-gateway,version=green -o jsonpath='{.items[0].metadata.name}') -- curl http://localhost:8080/health
```

#### Phase 2: Canary Testing (10% Traffic)

```bash
# Update ingress to route 10% traffic to green
kubectl apply -f k8s/networking/ingress-canary-10.yaml

# Monitor for 30 minutes
kubectl logs -f -l app=api-gateway,version=green
kubectl top pods -l app=api-gateway,version=green

# Check error rates
kubectl exec -it prometheus-0 -- promtool query instant 'rate(http_requests_total{status=~"5.."}[5m])'
```

#### Phase 3: Gradual Rollout

```bash
# Increase to 25%
kubectl apply -f k8s/networking/ingress-canary-25.yaml
# Wait 15 minutes, monitor

# Increase to 50%
kubectl apply -f k8s/networking/ingress-canary-50.yaml
# Wait 15 minutes, monitor

# Increase to 75%
kubectl apply -f k8s/networking/ingress-canary-75.yaml
# Wait 15 minutes, monitor

# Full rollout (100%)
kubectl apply -f k8s/networking/ingress-canary-100.yaml
```

#### Phase 4: Cleanup Blue Environment

```bash
# After 24 hours of stable green deployment
kubectl delete -f k8s/services/api-gateway-blue.yaml
kubectl delete -f k8s/services/orchestrator-blue.yaml

# Update labels
kubectl label deployment api-gateway version=stable --overwrite
```

---

## Testing & Validation

### 1. Pre-Deployment Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/integration/ -v

# Run iOS deployment simulation
python tests/ios_deployment/ios_deployment_simulation.py

# Expected results:
# - Success rate > 99%
# - Average response time < 300ms
# - No critical errors
```

### 2. Smoke Tests

```bash
# Create smoke test script
cat > tests/smoke_test.sh << 'EOF'
#!/bin/bash

API_URL="${1:-http://localhost:8080}"

echo "Running smoke tests against $API_URL"

# Test 1: Health check
echo "Test 1: Health check"
curl -f "$API_URL/health" || exit 1

# Test 2: Chat endpoint
echo "Test 2: Chat endpoint"
curl -f -X POST "$API_URL/api/v1/chat" \
  -H "Content-Type: application/json" \
  -H "User-Agent: ReleAF-iOS-SDK/1.0.0" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}' || exit 1

# Test 3: Vision endpoint
echo "Test 3: Vision endpoint"
curl -f -X POST "$API_URL/api/v1/vision/analyze" \
  -H "Content-Type: application/json" \
  -H "User-Agent: ReleAF-iOS-SDK/1.0.0" \
  -d '{"image_b64":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==","enable_detection":true}' || exit 1

# Test 4: Organization search
echo "Test 4: Organization search"
curl -f -X POST "$API_URL/api/v1/organizations/search" \
  -H "Content-Type: application/json" \
  -H "User-Agent: ReleAF-iOS-SDK/1.0.0" \
  -d '{"location":{"latitude":37.7749,"longitude":-122.4194},"radius_km":10}' || exit 1

echo "All smoke tests passed!"
EOF

chmod +x tests/smoke_test.sh

# Run smoke tests
./tests/smoke_test.sh http://green-api-gateway:8080
```

### 3. Load Testing

```bash
# Install k6 if not already installed
brew install k6  # macOS
# or
sudo apt-get install k6  # Linux

# Create load test script
cat > tests/load_test.js << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 200 },  // Ramp up to 200 users
    { duration: '5m', target: 200 },  // Stay at 200 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
  },
};

const BASE_URL = __ENV.API_URL || 'http://localhost:8080';

export default function () {
  // Chat request
  let chatRes = http.post(
    `${BASE_URL}/api/v1/chat`,
    JSON.stringify({
      messages: [{ role: 'user', content: 'How to recycle plastic?' }],
    }),
    {
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'ReleAF-iOS-SDK/1.0.0',
      },
    }
  );

  check(chatRes, {
    'chat status is 200': (r) => r.status === 200,
    'chat response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1);
}
EOF

# Run load test
k6 run --env API_URL=http://green-api-gateway:8080 tests/load_test.js
```

---

## Rollback Plan

### Immediate Rollback (< 5 minutes)

```bash
# Switch traffic back to blue
kubectl apply -f k8s/networking/ingress-blue.yaml

# Verify traffic switched
kubectl get ingress -n releaf-production

# Monitor blue environment
kubectl logs -f -l app=api-gateway,version=blue
```

### Database Rollback

```bash
# Rollback migrations
cat > migrations/ios_deployment/rollback_001.sql << 'EOF'
DROP TABLE IF EXISTS ios_requests;
DROP TABLE IF EXISTS ios_users;
EOF

# Execute rollback
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/ios_deployment/rollback_001.sql
```

### Full Rollback

```bash
# Revert to previous git tag
git checkout v1.0.0-pre-ios

# Rebuild and redeploy
docker build -t releaf/api-gateway:v1.0.0 .
docker push releaf/api-gateway:v1.0.0

kubectl set image deployment/api-gateway api-gateway=releaf/api-gateway:v1.0.0
kubectl rollout status deployment/api-gateway
```

---

## Post-Merge Monitoring

### 1. Key Metrics to Monitor

```yaml
# Prometheus queries
metrics:
  # Request rate
  - name: "iOS Request Rate"
    query: 'rate(http_requests_total{user_agent=~".*ReleAF-iOS-SDK.*"}[5m])'

  # Error rate
  - name: "iOS Error Rate"
    query: 'rate(http_requests_total{user_agent=~".*ReleAF-iOS-SDK.*",status=~"5.."}[5m])'

  # Response time
  - name: "iOS P95 Response Time"
    query: 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{user_agent=~".*ReleAF-iOS-SDK.*"}[5m]))'

  # Success rate
  - name: "iOS Success Rate"
    query: 'sum(rate(http_requests_total{user_agent=~".*ReleAF-iOS-SDK.*",status=~"2.."}[5m])) / sum(rate(http_requests_total{user_agent=~".*ReleAF-iOS-SDK.*"}[5m]))'
```

### 2. Alerts

```yaml
# Prometheus alert rules
groups:
  - name: ios_deployment
    interval: 30s
    rules:
      - alert: iOSHighErrorRate
        expr: rate(http_requests_total{user_agent=~".*ReleAF-iOS-SDK.*",status=~"5.."}[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate for iOS clients"

      - alert: iOSSlowResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{user_agent=~".*ReleAF-iOS-SDK.*"}[5m])) > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Slow response times for iOS clients"
```

### 3. Dashboard

Create Grafana dashboard with:
- iOS request rate (requests/second)
- iOS error rate (%)
- iOS response time (P50, P95, P99)
- iOS success rate (%)
- iOS active users
- iOS app versions distribution
- iOS device models distribution

---

## Success Criteria

✅ **Deployment is successful if:**

1. **Availability**: 99.9% uptime maintained
2. **Performance**: P95 response time < 500ms
3. **Reliability**: Error rate < 1%
4. **iOS Adoption**: > 100 iOS users in first week
5. **No Regressions**: Existing web clients unaffected
6. **Monitoring**: All metrics collecting correctly
7. **Rollback**: Rollback plan tested and working

---

## Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Preparation | 1 day | Backups, branch creation, testing |
| Merge | 2 hours | Code merge, configuration updates |
| Testing | 1 day | Unit tests, integration tests, smoke tests |
| Staging Deploy | 4 hours | Deploy to staging, validation |
| Production Deploy | 8 hours | Blue-green deployment, canary testing |
| Monitoring | 7 days | Monitor metrics, gather feedback |
| Cleanup | 2 hours | Remove blue environment, documentation |

**Total: ~10 days**

---

## Contact & Support

- **DevOps Lead**: devops@releaf.ai
- **Backend Lead**: backend@releaf.ai
- **iOS Lead**: ios@releaf.ai
- **On-Call**: oncall@releaf.ai

---

**Last Updated:** 2025-12-15
**Version:** 1.0.0
