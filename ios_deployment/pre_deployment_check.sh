#!/bin/bash

# Pre-Deployment Check Script for iOS Integration
# Validates all files, configurations, and dependencies before deployment

set -e

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║              🍎 iOS DEPLOYMENT PRE-FLIGHT CHECK                              ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Helper functions
check_pass() {
    echo -e "${GREEN}✅ PASS${NC} | $1"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}❌ FAIL${NC} | $1"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠️  WARN${NC} | $1"
    ((WARNINGS++))
}

echo "📦 Checking iOS Deployment Package..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check 1: iOS SDK files exist
if [ -f "ios_deployment/ReleAFSDK.swift" ]; then
    check_pass "ReleAFSDK.swift exists"
else
    check_fail "ReleAFSDK.swift missing"
fi

if [ -f "ios_deployment/ReleAFSDK+Network.swift" ]; then
    check_pass "ReleAFSDK+Network.swift exists"
else
    check_fail "ReleAFSDK+Network.swift missing"
fi

# Check 2: Documentation files exist
DOCS=(
    "README.md"
    "API_DOCUMENTATION.md"
    "FRONTEND_INTEGRATION_GUIDE.md"
    "FRONTEND_UPDATES_LIST.md"
    "PERFORMANCE_OPTIMIZATION_GUIDE.md"
    "BACKEND_MERGE_GUIDE.md"
    "BACKEND_INTEGRATION_UPDATES.md"
    "DEPLOYMENT_CHECKLIST.md"
    "DEPLOYMENT_SUMMARY.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "ios_deployment/$doc" ]; then
        check_pass "Documentation: $doc exists"
    else
        check_fail "Documentation: $doc missing"
    fi
done

# Check 3: Configuration files exist
if [ -f "ios_deployment/production_config.yaml" ]; then
    check_pass "production_config.yaml exists"
else
    check_fail "production_config.yaml missing"
fi

# Check 4: Simulation script exists
if [ -f "ios_deployment/ios_deployment_simulation.py" ]; then
    check_pass "ios_deployment_simulation.py exists"
else
    check_fail "ios_deployment_simulation.py missing"
fi

# Check 5: Validation script exists
if [ -f "ios_deployment/validate_ios_integration.py" ]; then
    check_pass "validate_ios_integration.py exists"
else
    check_fail "validate_ios_integration.py missing"
fi

echo ""
echo "🔧 Checking Backend Integration..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check 6: API Gateway exists
if [ -f "services/api_gateway/main.py" ]; then
    check_pass "API Gateway main.py exists"
    
    # Check CORS configuration
    if grep -q "CORS_ORIGINS" services/api_gateway/main.py; then
        check_pass "API Gateway has CORS_ORIGINS configuration"
    else
        check_warn "API Gateway CORS may need iOS origins update"
    fi
else
    check_fail "API Gateway main.py missing"
fi

# Check 7: Middleware exists
if [ -f "services/api_gateway/middleware/rate_limit.py" ]; then
    check_pass "Rate limiting middleware exists"
else
    check_fail "Rate limiting middleware missing"
fi

if [ -f "services/api_gateway/middleware/auth.py" ]; then
    check_pass "Auth middleware exists"
else
    check_fail "Auth middleware missing"
fi

# Check 8: Service files exist
SERVICES=(
    "services/orchestrator/main.py"
    "services/llm_service/server_v2.py"
    "services/rag_service/server.py"
    "services/vision_service/server_v2.py"
    "services/kg_service/server.py"
    "services/org_search_service/server.py"
)

for service in "${SERVICES[@]}"; do
    if [ -f "$service" ]; then
        check_pass "Service: $service exists"
    else
        check_fail "Service: $service missing"
    fi
done

echo ""
echo "☸️  Checking Kubernetes Manifests..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check 9: Kubernetes manifests exist
K8S_MANIFESTS=(
    "k8s/namespace.yaml"
    "k8s/configmaps/app-config.yaml"
    "k8s/services/api-gateway.yaml"
    "k8s/networking/ingress.yaml"
    "k8s/autoscaling/hpa.yaml"
)

for manifest in "${K8S_MANIFESTS[@]}"; do
    if [ -f "$manifest" ]; then
        check_pass "K8s manifest: $manifest exists"
    else
        check_fail "K8s manifest: $manifest missing"
    fi
done

echo ""
echo "🐍 Checking Python Dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check 10: Python dependencies
PYTHON_DEPS=(
    "fastapi"
    "aiohttp"
    "pydantic"
)

for dep in "${PYTHON_DEPS[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        check_pass "Python dependency: $dep installed"
    else
        check_warn "Python dependency: $dep not installed (may be optional)"
    fi
done

echo ""
echo "📊 SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Passed:   $PASSED"
echo "Failed:   $FAILED"
echo "Warnings: $WARNINGS"
echo ""

TOTAL=$((PASSED + FAILED + WARNINGS))
PASS_RATE=$((PASSED * 100 / TOTAL))

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL CHECKS PASSED! Ready for deployment.${NC}"
    exit 0
elif [ $PASS_RATE -ge 90 ]; then
    echo -e "${YELLOW}⚠️  MOSTLY READY. Review warnings and failures.${NC}"
    exit 1
else
    echo -e "${RED}❌ NOT READY. Fix critical failures before deployment.${NC}"
    exit 2
fi

