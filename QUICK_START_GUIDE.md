# QUICK START GUIDE - Get Running in 1 Hour

**Goal:** Fix environment and start training/running services  
**Time:** 30-60 minutes  
**Difficulty:** Easy

---

## STEP 1: Fix Environment (30 min)

### Install ARM Python
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ARM Python
brew install python@3.11

# Verify ARM
python3.11 -c "import platform; print(platform.machine())"
# Should print: arm64
```

### Create ARM Virtual Environment
```bash
cd /Users/jiangshengbo/Desktop/Sustainability-AI-Model

# Create venv
python3.11 -m venv venv-arm

# Activate
source venv-arm/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Install Dependencies
```bash
# Install PyTorch for Apple Silicon
pip install torch torchvision torchaudio

# Install transformers ecosystem
pip install transformers accelerate peft datasets sentence-transformers

# Install JAX for Apple Silicon
pip install jax-metal

# Install all project dependencies
pip install -r requirements.txt
```

### Verify Installation
```bash
# Check transformers works
python -c "from transformers import Trainer; print('✅ Transformers works!')"

# Check all services import
python -c "from services.llm_service import server_v2; print('✅ LLM service works!')"
python -c "from services.rag_service import server; print('✅ RAG service works!')"
python -c "from services.vision_service import server_v2; print('✅ Vision service works!')"
python -c "from services.kg_service import server; print('✅ KG service works!')"

# Check training scripts import
python -c "from training.llm import train_sft; print('✅ LLM training works!')"
python -c "from training.vision import train_classifier; print('✅ Vision training works!')"
python -c "from training.gnn import train_gnn; print('✅ GNN training works!')"
```

**Expected:** All ✅ checks pass

---

## STEP 2: Set Up Environment Variables (5 min)

### Create .env File
```bash
# Copy example
cp .env.example .env

# Edit with your values
nano .env
```

### Minimum Required Variables
```bash
# .env
ENV=development

# Database passwords (for local dev, use simple passwords)
POSTGRES_PASSWORD=dev_password_123
NEO4J_PASSWORD=dev_password_123

# API keys (for local dev, optional)
VALID_API_KEYS=dev-key-12345

# Service URLs (for local dev)
LLM_SERVICE_URL=http://localhost:8001
RAG_SERVICE_URL=http://localhost:8002
VISION_SERVICE_URL=http://localhost:8003
KG_SERVICE_URL=http://localhost:8004
ORG_SEARCH_SERVICE_URL=http://localhost:8005
```

---

## STEP 3: Start Databases (5 min)

### Option A: Docker Compose (Recommended)
```bash
# Start only databases
docker-compose up -d postgres neo4j qdrant

# Verify
docker ps
# Should see 3 containers running
```

### Option B: Local Installation
```bash
# PostgreSQL
brew install postgresql@16
brew services start postgresql@16

# Neo4j
brew install neo4j
neo4j start

# Qdrant
docker run -d -p 6333:6333 qdrant/qdrant:v1.8.0
```

---

## STEP 4: Test Services (10 min)

### Start Vision Service
```bash
# Terminal 1
source venv-arm/bin/activate
uvicorn services.vision_service.server_v2:app --port 8003 --reload

# Test (in another terminal)
curl http://localhost:8003/health
# Should return: {"status": "healthy"}
```

### Start KG Service
```bash
# Terminal 2
source venv-arm/bin/activate
uvicorn services.kg_service.server:app --port 8004 --reload

# Test
curl http://localhost:8004/health
# Should return: {"status": "healthy"}
```

### Start Org Search Service
```bash
# Terminal 3
source venv-arm/bin/activate
uvicorn services.org_search_service.server:app --port 8005 --reload

# Test
curl http://localhost:8005/health
# Should return: {"status": "healthy"}
```

---

## STEP 5: Prepare Sample Data (Optional)

### Create Minimal Test Dataset
```bash
# Create directories
mkdir -p data/processed/llm_sft
mkdir -p data/processed/vision_cls
mkdir -p data/processed/gnn

# Create sample LLM data
cat > data/processed/llm_sft/sustainability_qa_train.jsonl << 'EOF'
{"messages": [{"role": "user", "content": "What is recycling?"}, {"role": "assistant", "content": "Recycling is the process of converting waste materials into new materials and objects."}]}
{"messages": [{"role": "user", "content": "How do I recycle plastic?"}, {"role": "assistant", "content": "Check the recycling number on the plastic. Most communities accept #1 and #2 plastics."}]}
EOF

# Create validation set
cp data/processed/llm_sft/sustainability_qa_train.jsonl data/processed/llm_sft/sustainability_qa_val.jsonl
```

---

## STEP 6: Test Training (Optional)

### Test Vision Training
```bash
# Dry run (won't actually train without data)
python training/vision/train_classifier.py --help

# Should show usage without errors
```

### Test LLM Training
```bash
# Dry run
python training/llm/train_sft.py --help

# Should show usage without errors
```

---

## VERIFICATION CHECKLIST

After completing all steps:

- [ ] ARM Python installed (`python -c "import platform; print(platform.machine())"` → arm64)
- [ ] All dependencies installed (`pip list | grep transformers`)
- [ ] Transformers works (`python -c "from transformers import Trainer"`)
- [ ] All services import (`python -c "from services.llm_service import server_v2"`)
- [ ] Databases running (`docker ps` shows postgres, neo4j, qdrant)
- [ ] .env file created with passwords
- [ ] At least 1 service starts (`curl http://localhost:8003/health`)

---

## TROUBLESHOOTING

### Issue: "Architecture: x86_64"
**Fix:** You're using wrong Python. Use `python3.11` explicitly:
```bash
python3.11 -m venv venv-arm
```

### Issue: "jaxlib AVX error"
**Fix:** Install JAX for Apple Silicon:
```bash
pip uninstall jax jaxlib
pip install jax-metal
```

### Issue: "Connection refused" for databases
**Fix:** Start databases:
```bash
docker-compose up -d postgres neo4j qdrant
```

### Issue: "POSTGRES_PASSWORD must be set"
**Fix:** Create .env file with passwords (see Step 2)

---

## NEXT STEPS

After quick start:

1. **Prepare Real Datasets**
   - Run data collection scripts
   - Annotate vision data
   - Build knowledge graph

2. **Start Training**
   - Vision classifier: `python training/vision/train_classifier.py`
   - GNN: `python training/gnn/train_gnn.py`
   - LLM: `python training/llm/train_sft.py`

3. **Deploy Services**
   - Create Dockerfiles
   - Test docker-compose
   - Deploy to staging

4. **Integration Testing**
   - Test service-to-service calls
   - Verify end-to-end flows
   - Load testing

---

## EXPECTED RESULTS

After 1 hour:
- ✅ ARM Python environment working
- ✅ All dependencies installed
- ✅ All services can import
- ✅ Databases running
- ✅ At least 3 services running
- ✅ Health checks passing

**You're ready to start development!**

---

## SUPPORT

If you encounter issues:

1. Check `ENVIRONMENT_FIX_GUIDE.md` for detailed troubleshooting
2. Check `EXTREME_SKEPTICISM_AUDIT_RESULTS.md` for known issues
3. Verify all steps in this guide were completed

**Most common issue:** Using x86 Python instead of ARM Python  
**Quick fix:** Use `python3.11` explicitly everywhere

