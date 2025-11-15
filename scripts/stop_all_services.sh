#!/bin/bash
# Stop all ReleAF AI services

set -e

echo "🛑 Stopping ReleAF AI Services..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to stop service by PID file
stop_service() {
    local service_name=$1
    local pid_file="logs/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping ${service_name} (PID: $pid)...${NC}"
            kill $pid
            rm "$pid_file"
            echo -e "${GREEN}✓ ${service_name} stopped${NC}"
        else
            echo -e "${YELLOW}${service_name} not running${NC}"
            rm "$pid_file"
        fi
    else
        echo -e "${YELLOW}No PID file for ${service_name}${NC}"
    fi
}

# Stop all services
stop_service "api_gateway"
stop_service "orchestrator"
stop_service "vision_service"
stop_service "llm_service"
stop_service "rag_service"
stop_service "kg_service"
stop_service "org_search_service"

# Stop Docker containers
echo -e "${YELLOW}Stopping Docker containers...${NC}"
docker-compose down
echo -e "${GREEN}✓ Docker containers stopped${NC}"

echo ""
echo -e "${GREEN}✓ All services stopped${NC}"

