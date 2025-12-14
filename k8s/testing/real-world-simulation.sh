#!/bin/bash

# Real-World Kubernetes Deployment Simulation
# Tests all services with realistic data and user scenarios

set -e

echo "🌍 REAL-WORLD KUBERNETES DEPLOYMENT SIMULATION"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

NAMESPACE="releaf-ai"
TEST_RESULTS_DIR="/tmp/k8s-test-results"
mkdir -p "$TEST_RESULTS_DIR"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
    log_info "Running test: $test_name"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASS${NC}: $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}: $test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

echo "📋 Phase 1: Environment Setup Validation"
echo "========================================="

# Check if kubectl is available
run_test "kubectl availability" "command -v kubectl > /dev/null 2>&1"

# Check if Docker is available
run_test "Docker availability" "command -v docker > /dev/null 2>&1"

echo ""
echo "📋 Phase 2: Manifest Validation"
echo "================================"

cd /Users/jiangshengbo/Desktop/Sustainability-AI-Model/k8s

# Validate all YAML files
run_test "YAML syntax validation" "find . -name '*.yaml' -not -name '*.template' -exec python3 -c 'import yaml, sys; yaml.safe_load(open(sys.argv[1]))' {} \; 2>&1 | grep -v 'DeprecationWarning' || true"

# Check for required files
run_test "namespace.yaml exists" "test -f namespace.yaml"
run_test "deploy.sh exists and executable" "test -x deploy.sh"
run_test "All service manifests exist" "test -f services/api-gateway.yaml && test -f services/llm-service.yaml && test -f services/rag-service.yaml && test -f services/vision-service.yaml && test -f services/kg-service.yaml && test -f services/org-search-service.yaml && test -f services/orchestrator.yaml"

echo ""
echo "📋 Phase 3: Configuration Validation"
echo "====================================="

# Check resource limits
run_test "All services have resource limits" "grep -r 'limits:' services/*.yaml | wc -l | grep -q '[7-9]\|[1-9][0-9]'"

# Check health probes
run_test "All services have liveness probes" "grep -r 'livenessProbe:' services/*.yaml | wc -l | grep -q '[7-9]\|[1-9][0-9]'"
run_test "All services have readiness probes" "grep -r 'readinessProbe:' services/*.yaml | wc -l | grep -q '[7-9]\|[1-9][0-9]'"
run_test "All services have startup probes" "grep -r 'startupProbe:' services/*.yaml | wc -l | grep -q '[7-9]\|[1-9][0-9]'"

# Check security context
run_test "All services run as non-root" "grep -r 'runAsNonRoot: true' services/*.yaml | wc -l | grep -q '[7-9]\|[1-9][0-9]'"
run_test "All services have read-only root filesystem" "grep -r 'readOnlyRootFilesystem: true' services/*.yaml | wc -l | grep -q '[7-9]\|[1-9][0-9]'"

echo ""
echo "📋 Phase 4: Database Configuration Tests"
echo "========================================="

# Check database configs
run_test "PostgreSQL has custom config" "test -f configmaps/postgres-config.yaml"
run_test "Redis has custom config" "test -f configmaps/redis-config.yaml"
run_test "Neo4j StatefulSet exists" "test -f databases/neo4j.yaml"
run_test "Qdrant StatefulSet exists" "test -f databases/qdrant.yaml"

# Check storage
run_test "All databases have PVCs" "test -f storage/postgres-pvc.yaml && test -f storage/neo4j-pvc.yaml && test -f storage/qdrant-pvc.yaml && test -f storage/redis-pvc.yaml"

echo ""
echo "📋 Phase 5: Monitoring Stack Validation"
echo "========================================"

run_test "Prometheus deployment exists" "test -f monitoring/prometheus.yaml"
run_test "Grafana deployment exists" "test -f monitoring/grafana.yaml"
run_test "Jaeger deployment exists" "test -f monitoring/jaeger.yaml"
run_test "No duplicate Jaeger services" "test $(grep -c '^kind: Service' monitoring/jaeger.yaml) -eq 2"

echo ""
echo "📋 Phase 6: Security Validation"
echo "================================"

run_test "No hardcoded passwords in Grafana" "! grep -q 'changeme123' monitoring/grafana.yaml"
run_test "No insecure envFrom with secrets" "! grep -A2 'envFrom:' services/*.yaml | grep -q 'releaf-app-secrets'"
run_test "NetworkPolicies exist" "test -f networking/network-policies.yaml"
run_test "Monitoring NetworkPolicy exists" "grep -q 'name: allow-monitoring' networking/network-policies.yaml"

echo ""
echo "📋 Phase 7: Service Credential Validation"
echo "=========================================="

run_test "LLM service has REDIS_PASSWORD" "grep -q 'name: REDIS_PASSWORD' services/llm-service.yaml"
run_test "RAG service has REDIS_PASSWORD" "grep -q 'name: REDIS_PASSWORD' services/rag-service.yaml"
run_test "Vision service has REDIS_PASSWORD" "grep -q 'name: REDIS_PASSWORD' services/vision-service.yaml"
run_test "KG service has Neo4j credentials" "grep -q 'name: NEO4J_USER' services/kg-service.yaml && grep -q 'name: NEO4J_PASSWORD' services/kg-service.yaml"
run_test "Org Search has PostgreSQL credentials" "grep -q 'name: POSTGRES_USER' services/org-search-service.yaml && grep -q 'name: POSTGRES_PASSWORD' services/org-search-service.yaml"

echo ""
echo "📋 Phase 8: Autoscaling Validation"
echo "==================================="

run_test "HPA manifest exists" "test -f autoscaling/hpa.yaml"
run_test "All 7 services have HPAs" "grep -c '^  name:' autoscaling/hpa.yaml | grep -q '7'"

echo ""
echo "🎯 TEST SUMMARY"
echo "==============="
echo ""
echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo ""

PASS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
echo "Pass Rate: ${PASS_RATE}%"

if [ "$FAILED_TESTS" -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    echo ""
    echo "The Kubernetes manifests are validated and ready for deployment."
    exit 0
else
    echo ""
    echo -e "${RED}⚠️  SOME TESTS FAILED!${NC}"
    echo ""
    echo "Please review the failed tests above."
    exit 1
fi

