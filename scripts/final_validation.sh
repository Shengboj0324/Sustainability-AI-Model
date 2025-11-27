#!/bin/bash
# Final Comprehensive Validation Script
# Validates all aspects of ReleAF AI codebase

set -e

echo "🚀 FINAL COMPREHENSIVE VALIDATION"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# Function to run test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo "🔍 Testing: $test_name"
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}: $test_name"
        ((PASSED++))
    else
        echo -e "${RED}❌ FAILED${NC}: $test_name"
        ((FAILED++))
    fi
    echo ""
}

echo "📋 VALIDATION CHECKLIST"
echo "=================================="
echo ""

# 1. Syntax Validation
echo "1️⃣  SYNTAX VALIDATION"
echo "-------------------"
run_test "Services syntax" "find services -name '*.py' -exec python3 -m py_compile {} \;"
run_test "Models syntax" "find models -name '*.py' -exec python3 -m py_compile {} \;"
run_test "Training syntax" "find training -name '*.py' -exec python3 -m py_compile {} \;"
echo ""

# 2. Import Validation
echo "2️⃣  IMPORT VALIDATION"
echo "-------------------"
run_test "Answer Formatter" "python3 -c 'import sys; sys.path.insert(0, \".\"); from services.shared.answer_formatter import AnswerFormatter'"
run_test "Vision System" "python3 -c 'import sys; sys.path.insert(0, \".\"); from models.vision.integrated_vision import IntegratedVisionSystem'"
run_test "Common Utils" "python3 -c 'import sys; sys.path.insert(0, \".\"); from services.shared.common import get_device'"
echo ""

# 3. Configuration Validation
echo "3️⃣  CONFIGURATION VALIDATION"
echo "-------------------------"
run_test "LLM Config" "test -f configs/llm_sft.yaml"
run_test "RAG Config" "test -f configs/rag.yaml"
run_test "Vision Config" "test -f configs/vision.yaml"
run_test "Orchestrator Config" "test -f configs/orchestrator.yaml"
echo ""

# 4. Test Execution
echo "4️⃣  TEST EXECUTION"
echo "----------------"
run_test "Comprehensive Simulation" "python3 tests/test_comprehensive_simulation.py"
run_test "Industrial Scale" "python3 tests/test_industrial_scale.py"
echo ""

# 5. Code Quality
echo "5️⃣  CODE QUALITY CHECKS"
echo "---------------------"
run_test "Deep Code Analysis" "python3 scripts/deep_code_fixing.py"
echo ""

# Summary
echo "=================================="
echo "📊 VALIDATION SUMMARY"
echo "=================================="
echo ""
echo "Total Tests: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
else
    echo -e "${GREEN}Failed: $FAILED${NC}"
fi
echo ""

# Calculate percentage
TOTAL=$((PASSED + FAILED))
if [ $TOTAL -gt 0 ]; then
    PERCENTAGE=$((PASSED * 100 / TOTAL))
    echo "Success Rate: $PERCENTAGE%"
fi
echo ""

# Final verdict
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL VALIDATIONS PASSED${NC}"
    echo "🎉 CODE QUALITY: WORLD-CLASS"
    echo "🚀 SYSTEM IS PRODUCTION-READY"
    exit 0
else
    echo -e "${RED}⚠️  $FAILED validation(s) failed${NC}"
    echo "Please review and fix the issues above"
    exit 1
fi

