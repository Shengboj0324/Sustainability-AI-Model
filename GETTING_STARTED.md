# 🚀 ReleAF AI - Getting Started Guide

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Training Models](#training-models)
4. [Preparing RAG System](#preparing-rag-system)
5. [Running Services](#running-services)
6. [Production Deployment](#production-deployment)

---

## 🎯 System Overview

**ReleAF AI** is a comprehensive sustainability and waste management AI platform with:
- **LLM Service**: Llama-3-8B with LoRA fine-tuning
- **Vision Service**: ViT classifier + YOLOv8 detector
- **RAG Service**: BGE-large embeddings + hybrid retrieval
- **Knowledge Graph**: Neo4j for material relationships
- **GNN Service**: GraphSAGE/GAT for upcycling recommendations
- **Organization Search**: Location-based charity/recycling center finder

**Architecture**: 6 microservices on ports 8000-8005
**Deployment Target**: Digital Ocean (Web + iOS backend)

---

## ✅ Prerequisites

### System Requirements
- **Python**: 3.8+
- **GPU**: CUDA-capable (recommended) or Apple Silicon (MPS)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB for models and data

### Install Dependencies
```bash
# Install core dependencies
pip install torch torchvision torchaudio
pip install transformers sentence-transformers
pip install fastapi uvicorn pydantic httpx
pip install chromadb faiss-cpu
pip install pyyaml pillow numpy pandas
pip install peft accelerate bitsandbytes

# For vision models
pip install ultralytics opencv-python

# For GNN
pip install torch-geometric networkx

# For Neo4j (optional)
pip install neo4j
```

### Verify Installation
```bash
python3 scripts/production_readiness_test.py
```

Expected output: **100% PRODUCTION READY!**

---

## 🎓 Training Models

### 1. LLM Training (Llama-3-8B with LoRA)

**Data**: `data/llm_training_ultra_expanded.json` (295+ examples)

**Configuration**: `configs/llm_sft.yaml`

**Train the model**:
```bash
# Basic training
python3 training/llm/train_sft.py

# With custom config
python3 training/llm/train_sft.py --config configs/llm_sft.yaml

# Advanced options
python3 training/llm/train_sft.py \
  --model_name meta-llama/Llama-3-8B \
  --output_dir models/llm/llama3-sustainability-lora \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32
```

**Expected Training Time**:
- GPU (RTX 3090): ~2-3 hours
- Apple M1/M2: ~4-6 hours
- CPU: Not recommended (20+ hours)

**Output**:
- Model checkpoint: `models/llm/llama3-sustainability-lora/`
- Training logs: `models/llm/llama3-sustainability-lora/training_log.txt`
- LoRA adapters: `models/llm/llama3-sustainability-lora/adapter_model.bin`

**Verify Training**:
```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3-8B')
model = PeftModel.from_pretrained(base_model, 'models/llm/llama3-sustainability-lora')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3-8B')

prompt = 'How do I recycle plastic bottles?'
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
"
```

---

### 2. Vision Model Training

#### A. Classifier Training (ViT Multi-Head)

**Data**: `data/raw/images/` + `data/vision_dataset_metadata.json`

**Configuration**: `configs/vision_cls.yaml`

**Train the classifier**:
```bash
# Prepare dataset first
python3 scripts/data/prepare_all_datasets.py

# Train multi-head classifier
python3 training/vision/train_multihead.py

# With custom config
python3 training/vision/train_multihead.py --config configs/vision_cls.yaml
```

**Expected Training Time**:
- GPU: ~1-2 hours
- Apple M1/M2: ~3-4 hours

**Output**:
- Model: `models/vision/multihead_classifier.pth`
- Metrics: `models/vision/training_metrics.json`

#### B. Detector Training (YOLOv8)



### 3. Test RAG Retrieval

```bash
python3 -c "
import asyncio
from services.rag_service.server import RAGService

async def test_rag():
    rag = RAGService()
    await rag.initialize()

    # Test query
    query = 'How do I recycle plastic bottles?'
    results = await rag.retrieve(query, top_k=3, mode='hybrid')

    print(f'Query: {query}')
    print(f'Found {len(results)} results:')
    for i, result in enumerate(results, 1):
        print(f'{i}. {result[\"title\"]} (score: {result[\"score\"]:.3f})')

asyncio.run(test_rag())
"
```

### 4. Advanced RAG Features

The RAG system includes advanced retrieval capabilities:

**Hybrid Retrieval** (Dense + Sparse):
```python
results = await rag.retrieve(query, mode='hybrid', top_k=5)
```

**Query Expansion**:
```python
results = await rag.retrieve_with_expansion(query, top_k=5)
```

**Multi-Query Retrieval**:
```python
results = await rag.retrieve_multi_query(query, top_k=5)
```

**Contextual Compression**:
```python
results = await rag.retrieve_with_compression(query, top_k=5)
```

---

## 🚀 Running Services

### 1. Start Individual Services

Each service runs on a specific port:

```bash
# LLM Service (Port 8001)
cd services/llm_service
uvicorn server_v2:app --host 0.0.0.0 --port 8001

# Vision Service (Port 8002)
cd services/vision_service
uvicorn server_v2:app --host 0.0.0.0 --port 8002

# RAG Service (Port 8003)
cd services/rag_service
uvicorn server:app --host 0.0.0.0 --port 8003

# Knowledge Graph Service (Port 8004)
cd services/kg_service
uvicorn server:app --host 0.0.0.0 --port 8004

# Organization Search Service (Port 8005)
cd services/org_search_service
uvicorn server:app --host 0.0.0.0 --port 8005

# Orchestrator (Port 8000)
cd services/orchestrator
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Start All Services (Production)

Create a startup script `start_all_services.sh`:

```bash
#!/bin/bash

# Start all ReleAF AI services

echo "🚀 Starting ReleAF AI Services..."

# Start services in background
cd services/llm_service && uvicorn server_v2:app --host 0.0.0.0 --port 8001 &
cd services/vision_service && uvicorn server_v2:app --host 0.0.0.0 --port 8002 &
cd services/rag_service && uvicorn server:app --host 0.0.0.0 --port 8003 &
cd services/kg_service && uvicorn server:app --host 0.0.0.0 --port 8004 &
cd services/org_search_service && uvicorn server:app --host 0.0.0.0 --port 8005 &
cd services/orchestrator && uvicorn main:app --host 0.0.0.0 --port 8000 &

echo "✅ All services started!"
echo "Orchestrator: http://localhost:8000"
echo "LLM: http://localhost:8001"
echo "Vision: http://localhost:8002"
echo "RAG: http://localhost:8003"
echo "KG: http://localhost:8004"
echo "Org Search: http://localhost:8005"
```

Make it executable:
```bash
chmod +x start_all_services.sh
./start_all_services.sh
```

### 3. Test Services

```bash
# Test orchestrator health
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "How do I recycle plastic?"}],
    "max_tokens": 512
  }'

# Test vision endpoint
curl -X POST http://localhost:8002/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/plastic-bottle.jpg",
    "enable_detection": true,
    "enable_classification": true
  }'
```

---

## 🌐 Production Deployment (Digital Ocean)

### 1. Prepare for Deployment

**Create requirements.txt**:
```bash
pip freeze > requirements.txt
```

**Create Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8000 8001 8002 8003 8004 8005

# Start services
CMD ["./start_all_services.sh"]
```

### 2. Deploy to Digital Ocean

**Option A: Docker Deployment**
```bash
# Build image
docker build -t releaf-ai .

# Run container
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -p 8003:8003 -p 8004:8004 -p 8005:8005 \
  releaf-ai
```

**Option B: App Platform**
1. Push code to GitHub
2. Create new App in Digital Ocean App Platform
3. Connect GitHub repository
4. Configure build settings:
   - Build Command: `pip install -r requirements.txt`
   - Run Command: `./start_all_services.sh`
5. Set environment variables
6. Deploy!

### 3. Environment Variables

Set these in production:

```bash
# Model paths
LLM_MODEL_PATH=/app/models/llm/llama3-sustainability-lora
VISION_MODEL_PATH=/app/models/vision/multihead_classifier.pth
GNN_MODEL_PATH=/app/models/gnn/graphsage_model.pth

# Service URLs (internal)
ORCHESTRATOR_URL=http://localhost:8000
LLM_SERVICE_URL=http://localhost:8001
VISION_SERVICE_URL=http://localhost:8002
RAG_SERVICE_URL=http://localhost:8003
KG_SERVICE_URL=http://localhost:8004
ORG_SEARCH_SERVICE_URL=http://localhost:8005

# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# API Keys (if needed)
OPENAI_API_KEY=your_key_here
```

### 4. Production Optimizations

**Enable caching**:
- All services use `RequestCache` and `RateLimiter` from `services/shared/utils.py`
- Already configured for production

**Connection pooling**:
- HTTP clients use connection pooling via `httpx.AsyncClient`
- Already implemented in all services

**Resource cleanup**:
- All services use `cleanup_resources()` from `services/shared/common.py`
- Automatic GPU memory management

**Rate limiting**:
- 100 requests per 60 seconds per IP (configurable)
- Already enabled in all services

---

## 📊 Monitoring & Maintenance

### Health Checks

All services expose `/health` endpoint:
```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
# etc.
```

### Metrics

Services expose `/stats` endpoint:
```bash
curl http://localhost:8001/stats
# Returns: cache hits, rate limits, request counts, etc.
```

### Logs

Services log to stdout/stderr. In production:
```bash
# View logs
docker logs -f <container_id>

# Or with Digital Ocean
doctl apps logs <app_id>
```

---

## 🎯 Next Steps

1. **Train Models**: Start with LLM training (most important)
2. **Prepare RAG**: Build vector database with your knowledge base
3. **Test Locally**: Run all services and test end-to-end
4. **Deploy**: Push to Digital Ocean
5. **Monitor**: Set up health checks and logging
6. **Iterate**: Expand datasets and retrain models

---

## 🆘 Troubleshooting

**Issue**: Out of memory during training
- **Solution**: Reduce batch size, use gradient accumulation, or use smaller model

**Issue**: Services can't communicate
- **Solution**: Check service URLs in environment variables

**Issue**: RAG returns no results
- **Solution**: Verify vector database is initialized, check document count

**Issue**: Vision model errors
- **Solution**: Ensure images are in correct format (RGB, proper size)

---

## 📚 Additional Resources

- **Code Quality**: Run `python3 scripts/production_readiness_test.py`
- **Data Annotation**: See `scripts/holistic_data_annotation.py`
- **Advanced RAG**: See `services/rag_service/advanced_retrieval.py`
- **Deduplication**: See `scripts/comprehensive_deduplication.py`

---

**🎉 You're ready to build the future of sustainability AI!**
python3 training/vision/train_detector.py --config configs/vision_det.yaml
```

**Expected Training Time**: ~2-3 hours on GPU

**Output**:
- Model: `models/vision/yolov8_detector.pt`
- Metrics: `models/vision/detector_metrics.json`

---

### 3. GNN Training (GraphSAGE/GAT)

**Data**: `data/gnn_training_fully_annotated.json` (20+ nodes, 100% annotated)

**Configuration**: `configs/gnn.yaml`

**Train the GNN**:
```bash
python3 training/gnn/train_gnn.py --config configs/gnn.yaml
```

**Expected Training Time**: ~30 minutes

**Output**:
- Model: `models/gnn/graphsage_model.pth`
- Node embeddings: `models/gnn/node_embeddings.npy`

---

## 📚 Preparing RAG System

### 1. Prepare Knowledge Base

**Data**: `data/rag_knowledge_base_expanded.json` (13+ documents)

**Expand the knowledge base** (optional):
```bash
# Add more documents to data/rag_knowledge_base_expanded.json
# Format:
# {
#   "doc_id": "unique_id",
#   "title": "Document Title",
#   "content": "Full document content...",
#   "doc_type": "guide|regulation|faq|tutorial",
#   "metadata": {"source": "...", "date": "..."}
# }
```

### 2. Build Vector Database

**Configuration**: `configs/rag.yaml`

**Initialize RAG system**:
```bash
# This will:
# 1. Load documents from data/rag_knowledge_base_expanded.json
# 2. Generate embeddings using BGE-large
# 3. Build ChromaDB vector database
# 4. Create hybrid search index

python3 -c "
import asyncio
from services.rag_service.server import RAGService

async def init_rag():
    rag = RAGService()
    await rag.initialize()
    print('✅ RAG system initialized')
    print(f'Documents indexed: {len(rag.documents)}')
    print(f'Vector DB ready: {rag.vector_db is not None}')

asyncio.run(init_rag())
"
```


