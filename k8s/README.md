# ReleAF AI - Kubernetes Deployment Manifests

**Production-grade Kubernetes configurations for Digital Ocean deployment**

## 📁 Directory Structure

```
k8s/
├── README.md                          # This file
├── namespace.yaml                     # Namespace definition
├── configmaps/                        # Configuration data
│   ├── app-config.yaml               # Application configurations
│   └── monitoring-config.yaml        # Monitoring configurations
├── secrets/                           # Sensitive data (DO NOT COMMIT)
│   ├── app-secrets.yaml              # Application secrets
│   ├── db-secrets.yaml               # Database credentials
│   └── monitoring-secrets.yaml       # Monitoring credentials
├── storage/                           # Persistent storage
│   ├── postgres-pvc.yaml             # PostgreSQL storage
│   ├── neo4j-pvc.yaml                # Neo4j storage
│   ├── qdrant-pvc.yaml               # Qdrant storage
│   └── redis-pvc.yaml                # Redis storage
├── databases/                         # Database StatefulSets
│   ├── postgres.yaml                 # PostgreSQL StatefulSet
│   ├── neo4j.yaml                    # Neo4j StatefulSet
│   ├── qdrant.yaml                   # Qdrant StatefulSet
│   └── redis.yaml                    # Redis StatefulSet
├── services/                          # Microservices Deployments
│   ├── api-gateway.yaml              # API Gateway
│   ├── orchestrator.yaml             # Orchestrator
│   ├── llm-service.yaml              # LLM Service
│   ├── rag-service.yaml              # RAG Service
│   ├── vision-service.yaml           # Vision Service
│   ├── kg-service.yaml               # Knowledge Graph Service
│   └── org-search-service.yaml       # Organization Search Service
├── networking/                        # Network policies and services
│   ├── services.yaml                 # Service definitions
│   ├── ingress.yaml                  # Ingress configuration
│   └── network-policies.yaml         # Network policies
├── autoscaling/                       # HorizontalPodAutoscalers
│   ├── api-gateway-hpa.yaml
│   ├── orchestrator-hpa.yaml
│   ├── llm-service-hpa.yaml
│   ├── rag-service-hpa.yaml
│   ├── vision-service-hpa.yaml
│   ├── kg-service-hpa.yaml
│   └── org-search-service-hpa.yaml
└── monitoring/                        # Monitoring stack
    ├── prometheus.yaml               # Prometheus deployment
    ├── grafana.yaml                  # Grafana deployment
    ├── jaeger.yaml                   # Jaeger tracing
    └── service-monitors.yaml         # ServiceMonitor CRDs
```

## 🚀 Quick Start

### Prerequisites

1. **Kubernetes Cluster** (Digital Ocean Kubernetes)
2. **kubectl** configured
3. **Helm** (for monitoring stack)
4. **Docker images** pushed to registry

### Deployment Steps

```bash
# 1. Create namespace
kubectl apply -f namespace.yaml

# 2. Create secrets (update with your values first!)
kubectl apply -f secrets/

# 3. Create ConfigMaps
kubectl apply -f configmaps/

# 4. Create persistent storage
kubectl apply -f storage/

# 5. Deploy databases
kubectl apply -f databases/

# 6. Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n releaf-ai --timeout=300s
kubectl wait --for=condition=ready pod -l app=neo4j -n releaf-ai --timeout=300s
kubectl wait --for=condition=ready pod -l app=qdrant -n releaf-ai --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n releaf-ai --timeout=300s

# 7. Deploy microservices
kubectl apply -f services/

# 8. Create network services and ingress
kubectl apply -f networking/

# 9. Deploy autoscaling
kubectl apply -f autoscaling/

# 10. Deploy monitoring stack
kubectl apply -f monitoring/
```

## 📊 Resource Requirements

### Minimum Cluster Size (Digital Ocean)
- **3 nodes** (for high availability)
- **8 vCPUs** per node
- **16 GB RAM** per node
- **100 GB SSD** per node

### Total Resources
- **CPU**: ~24 vCPUs
- **Memory**: ~48 GB
- **Storage**: ~300 GB

## 🔒 Security Best Practices

1. **Secrets Management**: Use sealed-secrets or external secrets operator
2. **Network Policies**: Restrict pod-to-pod communication
3. **RBAC**: Implement least-privilege access
4. **Image Security**: Use private registry with vulnerability scanning
5. **TLS**: Enable TLS for all external endpoints

## 📈 Monitoring & Observability

- **Metrics**: Prometheus + Grafana
- **Tracing**: Jaeger
- **Logging**: Loki (optional)
- **Alerting**: Prometheus Alertmanager + PagerDuty/Slack

## 🔄 CI/CD Integration

See `../docs/CICD.md` for GitHub Actions workflows

## 📝 Configuration

All configurations use environment variables and ConfigMaps for flexibility.

**Key environment variables**:
- `ENVIRONMENT`: production/staging/development
- `LOG_LEVEL`: info/debug/warning/error
- `JAEGER_ENDPOINT`: Jaeger collector endpoint
- `SENTRY_DSN`: Sentry error tracking DSN
- `SLACK_WEBHOOK`: Slack alerting webhook

## 🆘 Troubleshooting

```bash
# Check pod status
kubectl get pods -n releaf-ai

# View logs
kubectl logs -f <pod-name> -n releaf-ai

# Describe pod
kubectl describe pod <pod-name> -n releaf-ai

# Check events
kubectl get events -n releaf-ai --sort-by='.lastTimestamp'
```

## 📚 Additional Resources

- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Digital Ocean Kubernetes Guide](https://docs.digitalocean.com/products/kubernetes/)
- [Production Checklist](../docs/PRODUCTION_CHECKLIST.md)

