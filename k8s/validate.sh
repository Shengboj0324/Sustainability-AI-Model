#!/bin/bash

# ReleAF AI Kubernetes Manifest Validation Script
# This script validates all Kubernetes manifests for correctness and best practices

set -e

echo "🔍 ReleAF AI Kubernetes Manifest Validation"
echo "==========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0
PASSED=0

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((ERRORS++))
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

# Check for required tools
log_info "Checking for required tools..."

if ! command -v kubectl &> /dev/null; then
    log_error "kubectl not found. Please install kubectl."
    exit 1
fi

if command -v kubeval &> /dev/null; then
    HAS_KUBEVAL=true
    log_pass "kubeval found"
else
    log_warn "kubeval not found. Install with: brew install kubeval"
    HAS_KUBEVAL=false
fi

if command -v kube-score &> /dev/null; then
    HAS_KUBESCORE=true
    log_pass "kube-score found"
else
    log_warn "kube-score not found. Install with: brew install kube-score"
    HAS_KUBESCORE=false
fi

echo ""

# Validate with kubeval
if [ "$HAS_KUBEVAL" = true ]; then
    log_info "Running kubeval validation..."
    if find . -name "*.yaml" -not -path "*/\.*" -not -name "*.template" -exec kubeval --strict --ignore-missing-schemas {} \; 2>&1 | tee /tmp/kubeval.log; then
        log_pass "kubeval validation passed"
    else
        log_error "kubeval validation failed. See /tmp/kubeval.log for details."
    fi
    echo ""
fi

# Validate with kube-score
if [ "$HAS_KUBESCORE" = true ]; then
    log_info "Running kube-score validation..."
    if find . -name "*.yaml" -not -path "*/\.*" -not -name "*.template" -exec kube-score score {} \; 2>&1 | tee /tmp/kubescore.log; then
        log_pass "kube-score validation completed"
    else
        log_warn "kube-score found issues. See /tmp/kubescore.log for details."
    fi
    echo ""
fi

# Check for common issues
log_info "Checking for common issues..."

# Check for missing resource limits
log_info "Checking for missing resource limits..."
MISSING_LIMITS=$(grep -r "kind: Deployment" . --include="*.yaml" -l | xargs grep -L "limits:" || true)
if [ -n "$MISSING_LIMITS" ]; then
    log_warn "Some deployments missing resource limits:"
    echo "$MISSING_LIMITS"
else
    log_pass "All deployments have resource limits"
fi

# Check for missing health probes
log_info "Checking for missing health probes..."
MISSING_PROBES=$(grep -r "kind: Deployment" . --include="*.yaml" -l | xargs grep -L "livenessProbe:" || true)
if [ -n "$MISSING_PROBES" ]; then
    log_warn "Some deployments missing liveness probes:"
    echo "$MISSING_PROBES"
else
    log_pass "All deployments have liveness probes"
fi

# Check for hardcoded secrets
log_info "Checking for hardcoded secrets..."
HARDCODED_SECRETS=$(grep -r "password\|secret\|token" . --include="*.yaml" -i | grep -v "secretKeyRef\|secretName\|template" || true)
if [ -n "$HARDCODED_SECRETS" ]; then
    log_error "Potential hardcoded secrets found:"
    echo "$HARDCODED_SECRETS"
else
    log_pass "No hardcoded secrets found"
fi

# Check for latest image tags
log_info "Checking for 'latest' image tags..."
LATEST_TAGS=$(grep -r "image:.*:latest" . --include="*.yaml" || true)
if [ -n "$LATEST_TAGS" ]; then
    log_warn "Found 'latest' image tags (not recommended for production):"
    echo "$LATEST_TAGS"
else
    log_pass "No 'latest' image tags found"
fi

# Summary
echo ""
echo "========================================="
echo "Validation Summary"
echo "========================================="
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Errors:${NC} $ERRORS"
echo ""

if [ $ERRORS -gt 0 ]; then
    log_error "Validation failed with $ERRORS errors"
    exit 1
else
    log_pass "Validation completed successfully!"
    exit 0
fi

