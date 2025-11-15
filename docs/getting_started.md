# Getting Started with ReleAF AI

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
- **Docker & Docker Compose**: [Install Docker](https://docs.docker.com/get-docker/)
- **CUDA 11.8+** (for GPU support): [Install CUDA](https://developer.nvidia.com/cuda-downloads)
- **Git**: [Install Git](https://git-scm.com/downloads)

### Hardware Requirements

**Minimum**:
- CPU: 8 cores
- RAM: 32 GB
- GPU: NVIDIA GPU with 16GB VRAM (e.g., RTX 4080, A4000)
- Storage: 100 GB SSD

**Recommended**:
- CPU: 16+ cores
- RAM: 64 GB
- GPU: NVIDIA GPU with 24GB+ VRAM (e.g., RTX 4090, A5000, A6000)
- Storage: 500 GB NVMe SSD

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Sustainability-AI-Model
```

### 2. Run Setup Script

```bash
bash scripts/setup.sh
```

This script will:
- Create a virtual environment
- Install dependencies
- Create `.env` file
- Set up directory structure
- Optionally start databases

### 3. Configure Environment

Edit `.env` file with your configuration:

```bash
nano .env
```

Key configurations:
- Database credentials
- API keys (Hugging Face, Weights & Biases)
- Model paths
- Service URLs

### 4. Prepare Data

#### Download Sample Datasets

For vision models:
```bash
# TrashNet
wget https://github.com/garythung/trashnet/archive/master.zip
unzip master.zip -d data/raw/images/

# TACO (requires registration)
# Visit: http://tacodataset.org/
```

For LLM training:
```bash
# Create your own sustainability Q&A dataset
# See docs/data_preparation.md for guidelines
```

### 5. Train Models

#### Train Vision Classifier

```bash
python training/vision/train_classifier.py --config configs/vision_cls.yaml
```

#### Train Object Detector

```bash
python training/vision/train_detector.py --config configs/vision_det.yaml
```

#### Fine-tune LLM

```bash
python training/llm/train_sft.py --config configs/llm_sft.yaml
```

### 6. Start Services

#### Option A: Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

#### Option B: Manual Start

```bash
bash scripts/start_all_services.sh
```

### 7. Verify Installation

Check service health:

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "services": {
    "orchestrator": {"healthy": true},
    "vision": {"healthy": true},
    "llm": {"healthy": true},
    ...
  }
}
```

## Using the API

### Example 1: Classify Waste Image

```bash
curl -X POST http://localhost:8080/api/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/bottle.jpg",
    "return_probabilities": true,
    "top_k": 3
  }'
```

### Example 2: Chat with AI

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "How do I recycle plastic bottles?"}
    ],
    "location": {"lat": 37.7749, "lon": -122.4194}
  }'
```

### Example 3: Find Recycling Centers

```bash
curl -X POST http://localhost:8080/api/v1/organizations/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "recycling center",
    "location": {"lat": 37.7749, "lon": -122.4194},
    "radius_km": 10
  }'
```

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Edit code, add features, fix bugs...

### 3. Run Tests

```bash
pytest tests/
```

### 4. Format Code

```bash
black .
isort .
```

### 5. Commit and Push

```bash
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name
```

## Monitoring

### View Logs

```bash
# All services
tail -f logs/*.log

# Specific service
tail -f logs/llm_service.log
```

### Weights & Biases

Training metrics are logged to W&B:
https://wandb.ai/your-username/releaf-ai

### Prometheus & Grafana (Optional)

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

Access Grafana: http://localhost:3000

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

Reduce batch sizes in configs:
- `configs/llm_sft.yaml`: `per_device_train_batch_size`
- `configs/vision_cls.yaml`: `batch_size`

### Service Won't Start

Check logs:
```bash
tail -f logs/service_name.log
```

Check ports:
```bash
lsof -i :8080
```

### Database Connection Issues

Restart databases:
```bash
docker-compose restart postgres neo4j qdrant
```

## Next Steps

- [Architecture Overview](architecture.md)
- [Data Preparation Guide](data_preparation.md)
- [Training Guide](training_guide.md)
- [Deployment Guide](deployment.md)
- [API Reference](api_reference.md)

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Documentation: [Full docs](https://docs.releaf-ai.com)
- Community: [Discord](https://discord.gg/releaf-ai)

