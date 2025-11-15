# ReleAF AI - Production Deployment Guide (Digital Ocean)

**Target**: Web + iOS App Backend on Digital Ocean

## 🎯 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Digital Ocean Load Balancer              │
│                    (HTTPS, SSL Termination)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ Droplet │    │ Droplet │    │ Droplet │
    │   #1    │    │   #2    │    │   #3    │
    └─────────┘    └─────────┘    └─────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ Qdrant  │    │  Neo4j  │    │Postgres │
    │ (Vector)│    │ (Graph) │    │  (SQL)  │
    └─────────┘    └─────────┘    └─────────┘
```

## 📋 Infrastructure Requirements

### Compute (Droplets)

**Application Servers** (3x for HA):
- **Type**: CPU-Optimized Droplet
- **Size**: 8 vCPU, 16 GB RAM
- **Storage**: 160 GB SSD
- **Cost**: ~$96/month each
- **Total**: ~$288/month

**Database Servers**:
- **Qdrant**: 4 vCPU, 8 GB RAM, 80 GB SSD (~$48/month)
- **Neo4j**: 4 vCPU, 8 GB RAM, 80 GB SSD (~$48/month)
- **PostgreSQL**: Managed Database (~$60/month)

**Total Monthly Cost**: ~$444/month

### Load Balancer
- Digital Ocean Load Balancer
- SSL/TLS termination
- Health checks
- Cost: ~$12/month

### Storage
- Spaces (S3-compatible) for model files: ~$5/month
- Backups: ~$20/month

**Grand Total**: ~$481/month

---

## 🚀 Deployment Steps

### 1. Prepare Docker Images

```bash
# Build production images
docker build -t releaf-rag-service:latest -f services/rag_service/Dockerfile .
docker build -t releaf-kg-service:latest -f services/kg_service/Dockerfile .
docker build -t releaf-llm-service:latest -f services/llm_service/Dockerfile .
docker build -t releaf-vision-service:latest -f services/vision_service/Dockerfile .
docker build -t releaf-orchestrator:latest -f services/orchestrator/Dockerfile .
docker build -t releaf-api-gateway:latest -f services/api_gateway/Dockerfile .

# Tag for registry
docker tag releaf-rag-service:latest registry.digitalocean.com/releaf/rag-service:latest

# Push to Digital Ocean Container Registry
docker push registry.digitalocean.com/releaf/rag-service:latest
```

### 2. Set Up Managed Databases

**PostgreSQL** (Managed):
```bash
# Create via DO Console or CLI
doctl databases create releaf-postgres \
  --engine pg \
  --region nyc3 \
  --size db-s-2vcpu-4gb \
  --num-nodes 1
```

**Qdrant** (Self-hosted):
```bash
# Deploy on dedicated droplet
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /data/qdrant:/qdrant/storage \
  --restart always \
  qdrant/qdrant:latest
```

**Neo4j** (Self-hosted):
```bash
# Deploy on dedicated droplet
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -v /data/neo4j:/data \
  -e NEO4J_AUTH=neo4j/your_secure_password \
  -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
  --restart always \
  neo4j:latest
```

### 3. Deploy Services with Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  rag-service:
    image: registry.digitalocean.com/releaf/rag-service:latest
    environment:
      - QDRANT_HOST=${QDRANT_HOST}
      - QDRANT_PORT=6333
      - QDRANT_GRPC_PORT=6334
      - QDRANT_PREFER_GRPC=true
      - CACHE_SIZE=1000
      - CACHE_TTL=300
      - LOG_LEVEL=INFO
      - MAX_CONCURRENT=100
    ports:
      - "8003:8003"
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  kg-service:
    image: registry.digitalocean.com/releaf/kg-service:latest
    environment:
      - NEO4J_URI=${NEO4J_URI}
      - NEO4J_USER=${NEO4J_USER}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
    ports:
      - "8004:8004"
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8004/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Add other services...
```

Deploy:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## 🔒 Security Hardening

### 1. Environment Variables
```bash
# Never commit .env files
# Use Digital Ocean Secrets or environment variables
export QDRANT_HOST="10.x.x.x"  # Private IP
export NEO4J_PASSWORD="$(openssl rand -base64 32)"
export JWT_SECRET_KEY="$(openssl rand -base64 64)"
```

### 2. Firewall Rules
```bash
# Allow only necessary ports
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw enable
```

### 3. SSL/TLS
- Use Let's Encrypt for free SSL certificates
- Configure on Load Balancer
- Force HTTPS redirect

---

## 📊 Monitoring & Observability

### Prometheus + Grafana

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rag-service'
    static_configs:
      - targets: ['rag-service:8003']
    metrics_path: '/metrics'
```

### Key Metrics to Monitor

1. **Request Rate**: `rate(rag_requests_total[5m])`
2. **Error Rate**: `rate(rag_requests_total{status="error"}[5m])`
3. **Latency**: `histogram_quantile(0.95, rag_request_duration_seconds)`
4. **Cache Hit Rate**: `rag_cache_hits_total / (rag_cache_hits_total + rag_cache_misses_total)`
5. **Active Requests**: `rag_active_requests`

### Alerts

```yaml
# alerts.yml
groups:
  - name: releaf_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(rag_requests_total{status="error"}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rag_request_duration_seconds) > 2
        for: 5m
        annotations:
          summary: "95th percentile latency > 2s"
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build and push Docker images
        run: |
          docker build -t registry.digitalocean.com/releaf/rag-service:${{ github.sha }} .
          docker push registry.digitalocean.com/releaf/rag-service:${{ github.sha }}
      
      - name: Deploy to Digital Ocean
        run: |
          doctl kubernetes cluster kubeconfig save releaf-cluster
          kubectl set image deployment/rag-service rag-service=registry.digitalocean.com/releaf/rag-service:${{ github.sha }}
```

---

## 📱 Mobile Client Optimizations

### 1. Response Caching
- Cache TTL: 5 minutes (300s)
- LRU eviction policy
- 1000 entry limit

### 2. Request Compression
```python
# Enable gzip compression
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### 3. Connection Pooling
- Qdrant: 100 max connections, 20 keepalive
- Neo4j: 50 max connections
- PostgreSQL: 20 connections per service

---

## 🧪 Load Testing

```bash
# Install k6
brew install k6

# Run load test
k6 run load-test.js
```

`load-test.js`:
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp up
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
};

export default function () {
  let res = http.post('https://api.releaf.ai/retrieve', JSON.stringify({
    query: 'How to recycle plastic bottles?',
    top_k: 5,
    mode: 'hybrid'
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  sleep(1);
}
```

---

## 📝 Deployment Checklist

- [ ] Set up Digital Ocean account
- [ ] Create Container Registry
- [ ] Build and push Docker images
- [ ] Provision droplets (3x app servers)
- [ ] Set up managed PostgreSQL
- [ ] Deploy Qdrant on dedicated droplet
- [ ] Deploy Neo4j on dedicated droplet
- [ ] Configure private networking
- [ ] Set up firewall rules
- [ ] Deploy services with docker-compose
- [ ] Configure Load Balancer
- [ ] Set up SSL certificates
- [ ] Configure DNS records
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure alerts
- [ ] Run load tests
- [ ] Set up backups
- [ ] Document runbooks
- [ ] Train team on operations

---

**Next**: See `MONITORING.md` for detailed monitoring setup

