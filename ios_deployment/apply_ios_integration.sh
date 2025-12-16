#!/bin/bash

# Apply iOS Integration Updates to Backend
# This script applies all the critical updates needed for iOS deployment

set -e

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║              🔧 APPLYING iOS INTEGRATION UPDATES                             ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}📁 Project Root: $PROJECT_ROOT${NC}"
echo ""

# Backup function
backup_file() {
    local file=$1
    if [ -f "$file" ]; then
        cp "$file" "$file.backup.$(date +%Y%m%d_%H%M%S)"
        echo -e "${GREEN}✅${NC} Backed up: $file"
    fi
}

# Update function
update_file() {
    local file=$1
    local description=$2
    echo -e "${BLUE}🔄 Updating: $description${NC}"
    backup_file "$file"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Updating Environment Variables"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Update .env.example
ENV_FILE="$PROJECT_ROOT/.env.example"
if [ -f "$ENV_FILE" ]; then
    update_file "$ENV_FILE" ".env.example"
    
    # Check if CORS_ORIGINS needs update
    if grep -q "CORS_ORIGINS=http://localhost" "$ENV_FILE"; then
        # Update CORS_ORIGINS
        sed -i.bak 's|CORS_ORIGINS=.*|CORS_ORIGINS=https://releaf.ai,https://www.releaf.ai,https://app.releaf.ai,capacitor://localhost,ionic://localhost,http://localhost:3000,http://localhost:8080|' "$ENV_FILE"
        echo -e "${GREEN}✅${NC} Updated CORS_ORIGINS in .env.example"
    else
        echo -e "${YELLOW}⚠️${NC}  CORS_ORIGINS already configured or not found"
    fi
else
    echo -e "${YELLOW}⚠️${NC}  .env.example not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  Updating Kubernetes ConfigMaps"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Update k8s/configmaps/app-config.yaml
K8S_CONFIG="$PROJECT_ROOT/k8s/configmaps/app-config.yaml"
if [ -f "$K8S_CONFIG" ]; then
    update_file "$K8S_CONFIG" "Kubernetes ConfigMap"
    
    # Update CORS_ORIGINS
    sed -i.bak 's|CORS_ORIGINS: "\*"|CORS_ORIGINS: "https://releaf.ai,https://www.releaf.ai,https://app.releaf.ai,capacitor://localhost,ionic://localhost"|' "$K8S_CONFIG"
    echo -e "${GREEN}✅${NC} Updated CORS_ORIGINS in Kubernetes ConfigMap"
else
    echo -e "${YELLOW}⚠️${NC}  Kubernetes ConfigMap not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  Updating Kubernetes Ingress"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Update k8s/networking/ingress.yaml
INGRESS_FILE="$PROJECT_ROOT/k8s/networking/ingress.yaml"
if [ -f "$INGRESS_FILE" ]; then
    update_file "$INGRESS_FILE" "Kubernetes Ingress"
    
    # Update CORS allow-origin
    sed -i.bak 's|nginx.ingress.kubernetes.io/cors-allow-origin: "\*"|nginx.ingress.kubernetes.io/cors-allow-origin: "https://releaf.ai,https://www.releaf.ai,https://app.releaf.ai,capacitor://localhost,ionic://localhost"|' "$INGRESS_FILE"
    echo -e "${GREEN}✅${NC} Updated CORS in Kubernetes Ingress"
else
    echo -e "${YELLOW}⚠️${NC}  Kubernetes Ingress not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  Creating iOS Production Config"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Copy production config to services/api_gateway/config/
CONFIG_DIR="$PROJECT_ROOT/services/api_gateway/config"
mkdir -p "$CONFIG_DIR"

if [ -f "$SCRIPT_DIR/production_config.yaml" ]; then
    cp "$SCRIPT_DIR/production_config.yaml" "$CONFIG_DIR/production_ios.yaml"
    echo -e "${GREEN}✅${NC} Copied production_config.yaml to services/api_gateway/config/"
else
    echo -e "${YELLOW}⚠️${NC}  production_config.yaml not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  Creating iOS SDK Directory"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Copy iOS SDK to sdk/ios/
SDK_DIR="$PROJECT_ROOT/sdk/ios"
mkdir -p "$SDK_DIR"

if [ -f "$SCRIPT_DIR/ReleAFSDK.swift" ]; then
    cp "$SCRIPT_DIR/ReleAFSDK.swift" "$SDK_DIR/"
    echo -e "${GREEN}✅${NC} Copied ReleAFSDK.swift to sdk/ios/"
fi

if [ -f "$SCRIPT_DIR/ReleAFSDK+Network.swift" ]; then
    cp "$SCRIPT_DIR/ReleAFSDK+Network.swift" "$SDK_DIR/"
    echo -e "${GREEN}✅${NC} Copied ReleAFSDK+Network.swift to sdk/ios/"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6️⃣  Creating Documentation Directory"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Copy documentation to docs/ios/
DOCS_DIR="$PROJECT_ROOT/docs/ios"
mkdir -p "$DOCS_DIR"

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
    if [ -f "$SCRIPT_DIR/$doc" ]; then
        cp "$SCRIPT_DIR/$doc" "$DOCS_DIR/"
        echo -e "${GREEN}✅${NC} Copied $doc to docs/ios/"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7️⃣  Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✅ iOS integration files applied successfully!${NC}"
echo ""
echo "📁 Files created/updated:"
echo "   • .env.example (CORS updated)"
echo "   • k8s/configmaps/app-config.yaml (CORS updated)"
echo "   • k8s/networking/ingress.yaml (CORS updated)"
echo "   • services/api_gateway/config/production_ios.yaml (created)"
echo "   • sdk/ios/ReleAFSDK.swift (created)"
echo "   • sdk/ios/ReleAFSDK+Network.swift (created)"
echo "   • docs/ios/*.md (created)"
echo ""
echo "⚠️  MANUAL UPDATES STILL REQUIRED:"
echo "   1. Update services/api_gateway/main.py CORS configuration (lines 65-72)"
echo "   2. Add User-Agent logging middleware to services/api_gateway/main.py"
echo "   3. Add Request ID middleware to services/api_gateway/main.py"
echo "   4. Update services/api_gateway/middleware/rate_limit.py for tier support"
echo "   5. Add iOS health check endpoint to services/api_gateway/main.py"
echo ""
echo "📖 See BACKEND_INTEGRATION_UPDATES.md for detailed instructions"
echo ""
echo -e "${BLUE}🚀 Next steps:${NC}"
echo "   1. Review and apply manual updates"
echo "   2. Run: python3 ios_deployment/validate_ios_integration.py"
echo "   3. Run: python3 ios_deployment/ios_deployment_simulation.py"
echo "   4. Deploy to staging environment"
echo "   5. Run production validation"
echo ""

