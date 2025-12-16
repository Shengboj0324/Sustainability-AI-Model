#!/bin/bash

# ReleAF AI Kubernetes Cleanup Script
# This script removes all ReleAF AI resources from Kubernetes
# WARNING: This will delete all data! Use with caution.

set -e

NAMESPACE="releaf-ai"

echo "⚠️  ReleAF AI Kubernetes Cleanup"
echo "================================"
echo "This will DELETE all resources in namespace: $NAMESPACE"
echo "This includes all data in databases!"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Confirmation prompt
read -p "Are you sure you want to delete all resources? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    log_info "Cleanup cancelled."
    exit 0
fi

read -p "Type the namespace name to confirm: " CONFIRM_NS
if [ "$CONFIRM_NS" != "$NAMESPACE" ]; then
    log_error "Namespace mismatch. Cleanup cancelled."
    exit 1
fi

log_warn "Starting cleanup in 5 seconds... Press Ctrl+C to cancel."
sleep 5

# Delete in reverse order of creation
log_info "Deleting monitoring stack..."
kubectl delete -f monitoring/ --ignore-not-found=true

log_info "Deleting autoscaling..."
kubectl delete -f autoscaling/ --ignore-not-found=true

log_info "Deleting networking..."
kubectl delete -f networking/ --ignore-not-found=true

log_info "Deleting services..."
kubectl delete -f services/ --ignore-not-found=true

log_info "Deleting databases..."
kubectl delete -f databases/ --ignore-not-found=true

log_info "Deleting storage..."
kubectl delete -f storage/ --ignore-not-found=true

log_info "Deleting configmaps..."
kubectl delete -f configmaps/ --ignore-not-found=true

log_info "Deleting secrets..."
kubectl delete -f secrets/ --ignore-not-found=true

log_info "Deleting namespace..."
kubectl delete namespace $NAMESPACE --ignore-not-found=true

log_info "✅ Cleanup complete!"
echo ""
log_warn "Note: PersistentVolumes may still exist. Check with:"
echo "  kubectl get pv"
echo ""
log_warn "To delete PersistentVolumes manually:"
echo "  kubectl delete pv <pv-name>"
echo ""

