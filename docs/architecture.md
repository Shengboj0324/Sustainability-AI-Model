# ReleAF AI - System Architecture

## Overview

ReleAF AI is a multi-modal AI platform for sustainability, waste management, and upcycling. The system uses a microservices architecture with specialized AI models working in concert.

## Architecture Principles

1. **Separation of Concerns**: Each model has a crisp mandate - don't let one model do everything
2. **Modularity**: Services can be developed, deployed, and scaled independently
3. **Specialization**: Use the right tool for the job (LLM for reasoning, vision for recognition, etc.)
4. **Scalability**: Horizontal scaling of individual services based on load
5. **Maintainability**: Clear interfaces and well-defined contracts between services

## System Components

### 1. Text Brain - Domain LLM

**Purpose**: Reasoning, explanation, and decision-making for sustainability questions

**Architecture**:
- Base: Llama-3-8B-Instruct or Qwen-2.5-7B-Instruct
- Fine-tuning: LoRA/QLoRA SFT (no full model fine-tune)
- Specialization: 50k-150k domain-specific examples

**Capabilities**:
- Understand user questions (text + image captions)
- Chain-of-thought reasoning about recycling, upcycling, materials, chemistry
- Generate detailed but safe instructions and creative ideas
- Decide when to call RAG, org search APIs, or request more information
- Tool/function calling for orchestration

**Why not train from scratch?**
- Training LLMs from scratch requires massive compute (millions of dollars)
- Pre-trained models already have strong language understanding
- Fine-tuning specializes existing knowledge efficiently

### 2. Retrieval Brain - RAG Stack

**Purpose**: Ground truth knowledge retrieval

**Architecture**:
- Embedding: BGE-large or GTE-large for dense retrieval
- Retriever: Hybrid (BM25 + dense vector) with top-k fusion
- Vector Store: Qdrant or FAISS
- Re-ranker: Cross-encoder for final ranking

**Knowledge Sources**:
- Government recycling guidelines (city, state, country)
- NGO and environmental organization documentation
- Material property databases
- Upcycling project descriptions and DIY guides

**Why hybrid retrieval?**
- Dense retrieval: Semantic similarity, handles synonyms
- Sparse retrieval (BM25): Exact keyword matching, handles rare terms
- Fusion: Best of both worlds

### 3. Vision Brain - Waste Recognition

**Purpose**: Identify and classify waste materials from images

**Architecture**: Two-component system

#### A) Image Classifier
- Backbone: ViT-B/16 or ConvNeXt-Base
- Multi-head: Item type + Material + Bin type
- Use case: Clean/cropped images, single objects

**Classes**:
- Item types: ~20 classes (bottle, can, box, bag, textile, e-waste, etc.)
- Materials: ~15 classes (PET, HDPE, glass, aluminum, paper, etc.)
- Bin types: 4 classes (recycle, compost, landfill, hazardous)

#### B) Object Detector
- Model: YOLOv8/YOLOv11 or DETR variant
- Use case: Real-world messy scenes with multiple items
- Output: Bounding boxes + classifications for each item

**Why two models?**
- Classifier: High accuracy for single objects
- Detector: Handles complex scenes with multiple items
- Detector → Classifier pipeline: Detect boxes, crop, refine classification

### 4. Multimodal Bridge

**Purpose**: Connect vision and language understanding

**Architecture**:
- Custom VLM (LLaVA-style) fine-tuned on waste images + explanations
- Converts image understanding to natural language descriptions
- Enables LLM to reason about visual content

### 5. Knowledge Graph + GNN

**Purpose**: Model relationships and enable reasoning

**Architecture**:
- Graph DB: Neo4j (property graph)
- GNN: GraphSAGE or GAT (v2+ feature)

**Relationships**:
```
ItemType —[MADE_OF]→ Material
Material —[CAN_BE_UPCYCLED_TO]→ ProductIdea
Material —[HAS_HAZARD]→ Hazard
Material —[HAS_PROPERTY]→ Property
Organization —[ACCEPTS]→ ItemType|Material
Organization —[LOCATED_IN]→ Location
ProductIdea —[REQUIRES_TOOL]→ Tool
ProductIdea —[REQUIRES_SKILL]→ Skill
```

**GNN Use Cases** (v2+):
- Recommend new upcycling edges
- Find similar materials with known upcycling paths
- Predict feasibility of novel material combinations

**MVP**: Use graph queries and rule-based logic. GNN is an upgrade.

### 6. Organization Search Service

**Purpose**: Connect users with recycling facilities, charities, and environmental organizations

**Architecture**:
- Backend: FastAPI + PostgreSQL
- Data sources: Cached open data, charity directories, facility databases
- Geospatial: PostGIS for location-based queries

**Tool Interfaces**:
```python
search_orgs(query, lat, lon, radius_km, type)
get_recycling_rules(lat, lon)
get_material_properties(material_id)
```

**Why not raw web browsing?**
- LLMs hallucinate when browsing
- Cleaned APIs provide reliable, structured data
- Faster and more cost-effective

### 7. Orchestrator - Agent Layer

**Purpose**: Route requests and coordinate services

**Architecture**:
- Request classifier: Determines request type and task
- State machine: Manages workflow execution
- Tool caller: Invokes appropriate services

**Request Types**:
- IMAGE_ONLY, TEXT_ONLY, MULTIMODAL, BATCH

**Task Types**:
- BIN_DECISION: Which bin?
- UPCYCLING_IDEA: How to reuse?
- ORG_SEARCH: Where to recycle/donate?
- THEORY_QA: General questions
- MATERIAL_INFO: Chemistry and properties
- SAFETY_CHECK: Safe handling

**Workflows**:
```
BIN_DECISION:
  Vision → RAG (local rules) → LLM (decision)

UPCYCLING_IDEA:
  Vision → KG (paths) → RAG (examples) → LLM (ideas)

ORG_SEARCH:
  Org Search → LLM (rank & explain)

THEORY_QA:
  RAG → KG (optional) → LLM (answer)
```

## Data Flow

### Example: User uploads image asking "Can I recycle this?"

1. **API Gateway** receives request
2. **Orchestrator** classifies as IMAGE_ONLY + BIN_DECISION
3. **Vision Service** detects/classifies object → "PET plastic bottle"
4. **RAG Service** retrieves local recycling rules for user's location
5. **LLM Service** synthesizes decision with explanation
6. **API Gateway** returns response to user

### Example: User asks "How can I upcycle old jeans?"

1. **API Gateway** receives request
2. **Orchestrator** classifies as TEXT_ONLY + UPCYCLING_IDEA
3. **KG Service** queries upcycling paths for denim/cotton
4. **RAG Service** retrieves example projects
5. **LLM Service** generates creative, detailed ideas
6. **API Gateway** returns response to user

## Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM | Llama-3-8B, Qwen-2.5-7B, vLLM |
| Vision | PyTorch, timm, ultralytics |
| Embeddings | sentence-transformers, BGE |
| Vector DB | Qdrant, FAISS |
| Graph DB | Neo4j |
| GNN | PyTorch Geometric |
| Backend | FastAPI, PostgreSQL |
| Orchestration | Docker, Kubernetes |
| Monitoring | Prometheus, Grafana, Weights & Biases |

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Load Balancer                        │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     API Gateway                          │
│                  (Authentication, Rate Limiting)         │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     Orchestrator                         │
│              (Request Routing, Workflow)                 │
└─────────────────────────────────────────────────────────┘
          │           │           │           │
    ┌─────┴─────┬─────┴─────┬─────┴─────┬─────┴─────┐
    ▼           ▼           ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Vision │ │  LLM   │ │  RAG   │ │   KG   │ │  Org   │
│Service │ │Service │ │Service │ │Service │ │ Search │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

## Scaling Strategy

- **Vision Service**: GPU-based, horizontal scaling with load balancer
- **LLM Service**: vLLM for efficient batching, multiple replicas
- **RAG Service**: Cache frequently accessed documents, read replicas
- **KG Service**: Neo4j clustering for high availability
- **Org Search**: Stateless, easy horizontal scaling

## Security & Safety

1. **Input Validation**: Sanitize all user inputs
2. **Content Filtering**: Check for harmful/illegal content
3. **Rate Limiting**: Prevent abuse
4. **Safety Filters**: Ensure LLM doesn't suggest dangerous activities
5. **Data Privacy**: User data encryption, GDPR compliance
6. **Authentication**: API keys, OAuth for user accounts

## Future Enhancements

1. **Active Learning**: User feedback loop for continuous improvement
2. **Federated Learning**: Privacy-preserving model updates
3. **Edge Deployment**: On-device inference for mobile apps
4. **Multi-language**: Support for non-English languages
5. **AR Integration**: Augmented reality for real-time waste identification

