#!/bin/bash

# ReleAF AI Kubernetes Deployment Script
# This script deploys the entire ReleAF AI platform to Kubernetes
# Usage: ./deploy.sh [environment]
# Environment: dev, staging, production (default: production)

set -e

ENVIRONMENT=${1:-production}
NAMESPACE="releaf-ai"

echo "🚀 ReleAF AI Kubernetes Deployment"
echo "=================================="
echo "Environment: $ENVIRONMENT"
echo "Namespace: $NAMESPACE"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    log_error "kubectl not found. Please install kubectl."
    exit 1
fi

if ! command -v kubeval &> /dev/null; then
    log_warn "kubeval not found. Skipping manifest validation."
    SKIP_VALIDATION=true
fi

# Validate manifests
if [ "$SKIP_VALIDATION" != "true" ]; then
    log_info "Validating Kubernetes manifests..."
    find . -name "*.yaml" -not -path "*/\.*" -exec kubeval --strict {} \;
    log_info "✅ All manifests are valid"
fi

# Create namespace
log_info "Creating namespace: $NAMESPACE"
kubectl apply -f namespace.yaml

# Create secrets (must be created manually first)
log_info "Checking for secrets..."
if [ ! -f "secrets/app-secrets.yaml" ]; then
    log_error "secrets/app-secrets.yaml not found!"
    log_error "Please create secrets from template: cp secrets/app-secrets.yaml.template secrets/app-secrets.yaml"
    log_error "Then fill in the actual secret values."
    exit 1
fi

kubectl apply -f secrets/app-secrets.yaml

# Create Grafana admin password secret if it doesn't exist
log_info "Creating Grafana admin password secret..."
if ! kubectl get secret grafana-secrets -n $NAMESPACE &> /dev/null; then
    GRAFANA_PASSWORD=$(openssl rand -base64 32)
    kubectl create secret generic grafana-secrets \
        --from-literal=admin-password="$GRAFANA_PASSWORD" \
        --namespace=$NAMESPACE
    log_info "✅ Grafana admin password created: $GRAFANA_PASSWORD"
    log_warn "SAVE THIS PASSWORD! It won't be shown again."
else
    log_info "Grafana secret already exists, skipping creation"
fi

# Deploy ConfigMaps
log_info "Deploying ConfigMaps..."
kubectl apply -f configmaps/

# Deploy Storage
log_info "Deploying PersistentVolumeClaims..."
kubectl apply -f storage/

# Deploy Databases
log_info "Deploying databases (PostgreSQL, Neo4j, Qdrant, Redis)..."
kubectl apply -f databases/

# Wait for databases to be ready
log_info "Waiting for databases to be ready..."
kubectl wait --for=condition=ready pod -l component=database -n $NAMESPACE --timeout=300s

# Deploy Services
log_info "Deploying microservices..."
kubectl apply -f services/

# Deploy Networking
log_info "Deploying networking (Services, Ingress, NetworkPolicies)..."
kubectl apply -f networking/

# Deploy Autoscaling
log_info "Deploying HorizontalPodAutoscalers..."
kubectl apply -f autoscaling/

# Deploy Monitoring
log_info "Deploying monitoring stack (Prometheus, Grafana, Jaeger)..."
kubectl apply -f monitoring/

# Wait for all deployments to be ready
log_info "Waiting for all deployments to be ready..."
kubectl wait --for=condition=available deployment --all -n $NAMESPACE --timeout=600s

# Display deployment status
log_info "Deployment Status:"
kubectl get all -n $NAMESPACE

# Display service endpoints
log_info "Service Endpoints:"
kubectl get svc -n $NAMESPACE

# Display ingress
log_info "Ingress Configuration:"
kubectl get ingress -n $NAMESPACE

echo ""
log_info "✅ Deployment complete!"
echo ""
log_info "Next steps:"
echo "  1. Configure DNS to point to LoadBalancer IP"
echo "  2. Access Grafana: kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE"
echo "  3. Access Jaeger: kubectl port-forward svc/jaeger-query 16686:16686 -n $NAMESPACE"
echo "  4. Monitor logs: kubectl logs -f deployment/api-gateway -n $NAMESPACE"
echo ""

