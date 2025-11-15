# ReleAF AI - Sustainability & Waste Intelligence Platform

## Vision

ReleAF AI is a comprehensive AI-powered platform designed to revolutionize waste management, recycling, and upcycling through advanced machine learning. The system combines computer vision, natural language processing, knowledge graphs, and retrieval-augmented generation to provide:

- **Intelligent Waste Recognition**: Advanced image classification and object detection for identifying waste materials
- **Sustainability Expertise**: Domain-specialized LLM trained on recycling, upcycling, and circular economy knowledge
- **Creative Upcycling**: Scientific and innovative suggestions for transforming waste into art and usable objects
- **Organization Discovery**: Real-time connection to charities, clubs, and recycling facilities
- **Material Science**: Deep understanding of waste components, chemistry, and safe handling practices

## Architecture Overview

ReleAF AI uses a modular, microservices-based architecture with specialized AI models:

### Core Components

1. **Text Brain** - Domain LLM (8-14B parameter model with LoRA fine-tuning)
   - Reasoning and explanation for sustainability questions
   - Upcycling ideation with safety constraints
   - Tool orchestration and decision-making

2. **Vision Brain** - Dual computer vision system
   - Image Classifier: ViT-based for single-object waste/material classification
   - Object Detector: YOLO-based for multi-object scene understanding

3. **Retrieval Brain** - RAG stack
   - Hybrid retrieval (BM25 + dense vectors)
   - Knowledge base of recycling rules, upcycling guides, material properties

4. **Knowledge Graph** - Neo4j-based relationship modeling
   - Material properties and relationships
   - Upcycling possibilities
   - Organization capabilities and locations

5. **Organization Search** - Real-time discovery service
   - Charities and environmental clubs
   - Recycling facilities and drop-off centers
   - Location-based recommendations

6. **Orchestrator** - Intelligent routing and coordination
   - Request classification and routing
   - Multi-modal integration
   - Tool calling and response synthesis

## Project Structure

```
releaf-ai/
├── configs/           # Training and service configurations
├── data/             # Raw, processed, and annotated datasets
├── docs/             # Architecture and API documentation
├── services/         # Runtime microservices
├── models/           # Model checkpoints and adapters
├── training/         # Training pipelines
├── scripts/          # Utility and deployment scripts
└── tests/            # Unit, integration, and e2e tests
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training/inference)
- Docker & Docker Compose
- Node.js 18+ (for some services)
- Neo4j 5.0+
- PostgreSQL 15+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Sustainability-AI-Model

# Install Python dependencies
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d
```

### Training Models

```bash
# Train vision classifier
python training/vision/train_classifier.py --config configs/vision_cls.yaml

# Train object detector
python training/vision/train_detector.py --config configs/vision_det.yaml

# Fine-tune LLM
python training/llm/train_sft.py --config configs/llm_sft.yaml
```

### Running Services

```bash
# Start all services
./scripts/start_all_services.sh

# Or start individually
python services/api_gateway/main.py
python services/orchestrator/main.py
python services/llm_service/server.py
python services/vision_service/server.py
python services/rag_service/server.py
```

## Development Roadmap

### Phase 1: Foundation (Current)
- [x] Project structure and architecture design
- [ ] Data collection and preprocessing pipelines
- [ ] Vision model training (classifier + detector)
- [ ] LLM fine-tuning on sustainability domain
- [ ] Basic RAG implementation

### Phase 2: Integration
- [ ] Microservices implementation
- [ ] Orchestrator and routing logic
- [ ] Knowledge graph construction
- [ ] Organization database and search
- [ ] API gateway and authentication

### Phase 3: Enhancement
- [ ] Multimodal VLM integration
- [ ] GNN for recommendation
- [ ] Advanced safety filters
- [ ] Performance optimization
- [ ] Comprehensive evaluation suite

### Phase 4: Production
- [ ] Deployment infrastructure
- [ ] Monitoring and logging
- [ ] User feedback loop
- [ ] Continuous learning pipeline
- [ ] Mobile and web interfaces

## Key Technologies

- **ML Frameworks**: PyTorch, Transformers, PEFT (LoRA)
- **Vision**: timm, ultralytics (YOLO), torchvision
- **LLM**: Llama-3, Qwen-2.5, vLLM for serving
- **Embeddings**: BGE-large, GTE-large
- **Vector DB**: Qdrant, FAISS
- **Graph DB**: Neo4j
- **Backend**: FastAPI, PostgreSQL
- **Orchestration**: Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana

## Data Sources

### Vision Datasets
- TrashNet
- TACO (Trash Annotations in Context)
- Kaggle Garbage Classification
- Humans in the Loop Recycling Dataset
- Custom user-contributed data (with consent)

### Knowledge Sources
- Government recycling guidelines (US, EU, etc.)
- Environmental NGO documentation
- Material science databases
- Upcycling and DIY project repositories
- Charity and organization directories

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

See [LICENSE](LICENSE) file for details.

## Contact

For questions and support, please open an issue or contact the development team.

---

**Note**: This is an active research and development project. Models and APIs are subject to change.