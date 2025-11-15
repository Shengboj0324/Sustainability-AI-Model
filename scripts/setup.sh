#!/bin/bash
# Setup script for ReleAF AI platform

set -e

echo "🌱 Setting up ReleAF AI Platform..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10+ is required. Found: $python_version"
    exit 1
fi
echo -e "${GREEN}✓ Python version: $python_version${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -e .
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created. Please update with your configuration.${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p logs
mkdir -p models/{llm,vision,gnn}
mkdir -p data/{raw,processed,annotations}
echo -e "${GREEN}✓ Directories created${NC}"

# Download sample data (optional)
read -p "Download sample datasets? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Downloading sample data...${NC}"
    bash scripts/download_sample_data.sh
fi

# Setup databases with Docker
read -p "Start databases with Docker Compose? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Starting databases...${NC}"
    docker-compose up -d postgres neo4j qdrant redis
    echo -e "${GREEN}✓ Databases started${NC}"
    
    # Wait for databases to be ready
    echo -e "${YELLOW}Waiting for databases to be ready...${NC}"
    sleep 10
    
    # Run database migrations
    echo -e "${YELLOW}Running database migrations...${NC}"
    python scripts/init_databases.py
fi

echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Update .env file with your configuration"
echo "2. Download or prepare your datasets"
echo "3. Train models: python training/llm/train_sft.py"
echo "4. Start services: docker-compose up"
echo ""
echo "For more information, see README.md"

