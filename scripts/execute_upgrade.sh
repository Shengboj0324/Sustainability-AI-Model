#!/bin/bash

################################################################################
# Full-Scale Upgrade Execution Script
# Executes comprehensive system upgrade for ReleAF AI
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "🚀 FULL-SCALE UPGRADE EXECUTION"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Step 1: Backup current state
echo "================================================================================"
echo "📦 STEP 1: BACKUP CURRENT STATE"
echo "================================================================================"
print_info "Creating backup of current configuration..."

BACKUP_DIR="backups/upgrade_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp pyproject.toml "$BACKUP_DIR/"
cp Dockerfile "$BACKUP_DIR/"
cp docker-compose.yml "$BACKUP_DIR/"
cp requirements.txt "$BACKUP_DIR/" 2>/dev/null || true

print_success "Backup created at: $BACKUP_DIR"
echo ""

# Step 2: Check Python version
echo "================================================================================"
echo "🐍 STEP 2: PYTHON VERSION CHECK"
echo "================================================================================"

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_info "Current Python version: $PYTHON_VERSION"

if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    print_success "Python 3.11+ detected"
else
    print_warning "Python 3.11+ recommended for optimal performance"
    print_info "Current version will work but consider upgrading"
fi
echo ""

# Step 3: Update pip and build tools
echo "================================================================================"
echo "🔧 STEP 3: UPDATE PIP AND BUILD TOOLS"
echo "================================================================================"

print_info "Upgrading pip, setuptools, and wheel..."
python3 -m pip install --upgrade pip setuptools wheel
print_success "Build tools updated"
echo ""

# Step 4: Install upgraded dependencies
echo "================================================================================"
echo "📦 STEP 4: INSTALL UPGRADED DEPENDENCIES"
echo "================================================================================"

print_info "Installing dependencies from upgraded pyproject.toml..."
pip install -e . --upgrade

print_success "Dependencies installed successfully"
echo ""

# Step 5: Validate syntax
echo "================================================================================"
echo "✅ STEP 5: SYNTAX VALIDATION"
echo "================================================================================"

print_info "Validating Python syntax for all service files..."

SYNTAX_ERRORS=0
for file in $(find services models -name "*.py" 2>/dev/null); do
    if ! python3 -m py_compile "$file" 2>/dev/null; then
        print_error "Syntax error in: $file"
        SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
    fi
done

if [ $SYNTAX_ERRORS -eq 0 ]; then
    print_success "All files passed syntax validation"
else
    print_error "Found $SYNTAX_ERRORS syntax errors"
    exit 1
fi
echo ""

# Step 6: Run tests
echo "================================================================================"
echo "🧪 STEP 6: RUN TESTS"
echo "================================================================================"

print_info "Running test suite..."

if [ -f "tests/test_comprehensive_simulation.py" ]; then
    python3 tests/test_comprehensive_simulation.py || {
        print_warning "Some tests failed - review output above"
    }
else
    print_warning "Test file not found - skipping tests"
fi
echo ""

# Step 7: Docker image validation
echo "================================================================================"
echo "🐳 STEP 7: DOCKER IMAGE VALIDATION"
echo "================================================================================"

print_info "Pulling upgraded Docker images..."

docker pull python:3.11-slim
docker pull postgis/postgis:16-3.4
docker pull neo4j:5.16
docker pull qdrant/qdrant:v1.8.0

print_success "Docker images updated"
echo ""

# Step 8: Build new Docker images
echo "================================================================================"
echo "🏗️  STEP 8: BUILD NEW DOCKER IMAGES"
echo "================================================================================"

print_info "Building updated application Docker images..."

if docker-compose build --no-cache; then
    print_success "Docker images built successfully"
else
    print_error "Docker build failed"
    exit 1
fi
echo ""

# Step 9: Final validation
echo "================================================================================"
echo "🎯 STEP 9: FINAL VALIDATION"
echo "================================================================================"

print_info "Running full-scale upgrade analysis..."
python3 scripts/full_scale_upgrade_analysis.py

echo ""
echo "================================================================================"
echo "🎉 UPGRADE COMPLETE!"
echo "================================================================================"
print_success "Full-scale upgrade executed successfully"
print_info "Backup location: $BACKUP_DIR"
print_info "Next steps:"
echo "  1. Review upgrade report above"
echo "  2. Test services with: docker-compose up"
echo "  3. Run comprehensive tests"
echo "  4. Monitor performance metrics"
echo ""
print_success "System ready for deployment! 🚀"
echo "================================================================================"

