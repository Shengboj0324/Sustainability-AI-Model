#!/bin/bash
# Start all ReleAF AI services

set -e

echo "🌱 Starting ReleAF AI Services..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found. Run setup.sh first.${NC}"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Start databases
echo -e "${YELLOW}Starting databases...${NC}"
docker-compose up -d postgres neo4j qdrant redis
echo -e "${GREEN}✓ Databases started${NC}"

# Wait for databases
echo -e "${YELLOW}Waiting for databases to be ready...${NC}"
sleep 10

# Check database health
echo -e "${YELLOW}Checking database health...${NC}"
docker-compose ps

# Start services in background
echo -e "${YELLOW}Starting microservices...${NC}"

# Activate virtual environment
source venv/bin/activate

# Start Vision Service
echo -e "${YELLOW}Starting Vision Service...${NC}"
cd services/vision_service
python server.py > ../../logs/vision_service.log 2>&1 &
VISION_PID=$!
cd ../..
echo -e "${GREEN}✓ Vision Service started (PID: $VISION_PID)${NC}"

# Start LLM Service
echo -e "${YELLOW}Starting LLM Service...${NC}"
cd services/llm_service
python server.py > ../../logs/llm_service.log 2>&1 &
LLM_PID=$!
cd ../..
echo -e "${GREEN}✓ LLM Service started (PID: $LLM_PID)${NC}"

# Start RAG Service
echo -e "${YELLOW}Starting RAG Service...${NC}"
cd services/rag_service
python server.py > ../../logs/rag_service.log 2>&1 &
RAG_PID=$!
cd ../..
echo -e "${GREEN}✓ RAG Service started (PID: $RAG_PID)${NC}"

# Start KG Service
echo -e "${YELLOW}Starting KG Service...${NC}"
cd services/kg_service
python server.py > ../../logs/kg_service.log 2>&1 &
KG_PID=$!
cd ../..
echo -e "${GREEN}✓ KG Service started (PID: $KG_PID)${NC}"

# Start Org Search Service
echo -e "${YELLOW}Starting Org Search Service...${NC}"
cd services/org_search_service
python server.py > ../../logs/org_search_service.log 2>&1 &
ORG_PID=$!
cd ../..
echo -e "${GREEN}✓ Org Search Service started (PID: $ORG_PID)${NC}"

# Wait for services to start
echo -e "${YELLOW}Waiting for services to initialize...${NC}"
sleep 5

# Start Orchestrator
echo -e "${YELLOW}Starting Orchestrator...${NC}"
cd services/orchestrator
python main.py > ../../logs/orchestrator.log 2>&1 &
ORCH_PID=$!
cd ../..
echo -e "${GREEN}✓ Orchestrator started (PID: $ORCH_PID)${NC}"

# Wait for orchestrator
sleep 3

# Start API Gateway
echo -e "${YELLOW}Starting API Gateway...${NC}"
cd services/api_gateway
python main.py > ../../logs/api_gateway.log 2>&1 &
API_PID=$!
cd ../..
echo -e "${GREEN}✓ API Gateway started (PID: $API_PID)${NC}"

# Save PIDs
echo "$VISION_PID" > logs/vision_service.pid
echo "$LLM_PID" > logs/llm_service.pid
echo "$RAG_PID" > logs/rag_service.pid
echo "$KG_PID" > logs/kg_service.pid
echo "$ORG_PID" > logs/org_search_service.pid
echo "$ORCH_PID" > logs/orchestrator.pid
echo "$API_PID" > logs/api_gateway.pid

echo ""
echo -e "${GREEN}✓ All services started successfully!${NC}"
echo ""
echo "Service URLs:"
echo "  API Gateway:    http://localhost:8080"
echo "  Orchestrator:   http://localhost:8000"
echo "  Vision Service: http://localhost:8001"
echo "  LLM Service:    http://localhost:8002"
echo "  RAG Service:    http://localhost:8003"
echo "  KG Service:     http://localhost:8004"
echo "  Org Search:     http://localhost:8005"
echo ""
echo "API Documentation: http://localhost:8080/docs"
echo ""
echo "Logs are available in: logs/"
echo "To stop services: bash scripts/stop_all_services.sh"

