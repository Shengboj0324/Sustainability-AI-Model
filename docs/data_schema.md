# Data Schemas and Formats

## Overview

This document defines the data schemas used throughout the ReleAF AI platform.

## 1. Vision Data

### Image Classification Dataset

**Directory Structure**:
```
data/processed/vision_cls/
├── train/
│   ├── plastic_bottle/
│   ├── glass_bottle/
│   ├── aluminum_can/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

**Metadata Format** (JSON):
```json
{
  "image_id": "img_001",
  "filename": "plastic_bottle_001.jpg",
  "item_type": "plastic_bottle",
  "material": "PET",
  "bin_type": "recycle",
  "source": "user_upload",
  "location": {"lat": 37.7749, "lon": -122.4194},
  "verified": true,
  "annotations": {
    "item_confidence": 0.95,
    "material_confidence": 0.88
  }
}
```

### Object Detection Dataset

**Format**: YOLO or COCO format

**YOLO Format** (txt):
```
# Each line: class_id center_x center_y width height
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

**COCO Format** (JSON):
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "scene_001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [100, 120, 200, 180],
      "area": 36000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "plastic_bottle",
      "supercategory": "recyclable"
    }
  ]
}
```

## 2. LLM Training Data

### Chat Format (JSONL)

**Sustainability Q&A**:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are ReleAF AI, an expert sustainability assistant..."
    },
    {
      "role": "user",
      "content": "Can I recycle pizza boxes with grease stains?"
    },
    {
      "role": "assistant",
      "content": "Pizza boxes with heavy grease stains should not be recycled..."
    }
  ],
  "metadata": {
    "category": "recycling_rules",
    "difficulty": "medium",
    "location": "US",
    "verified": true
  }
}
```

**Upcycling Ideas**:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are ReleAF AI, an expert sustainability assistant..."
    },
    {
      "role": "user",
      "content": "I have 5 glass jars and some old fabric. What can I make?"
    },
    {
      "role": "assistant",
      "content": "Here are some creative upcycling ideas:\n\n1. **Fabric-Covered Storage Jars**..."
    }
  ],
  "metadata": {
    "category": "upcycling",
    "materials": ["glass", "fabric"],
    "difficulty": "easy",
    "time_estimate": "30-60 minutes"
  }
}
```

**Tool Calling Examples**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Where can I recycle batteries near San Francisco?"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "search_orgs",
            "arguments": "{\"query\": \"battery recycling\", \"lat\": 37.7749, \"lon\": -122.4194, \"radius_km\": 10}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": "[{\"name\": \"SF Hazardous Waste Facility\", \"address\": \"...\"}]"
    },
    {
      "role": "assistant",
      "content": "I found several battery recycling locations near you..."
    }
  ]
}
```

## 3. RAG Knowledge Base

### Document Format

**Recycling Guidelines**:
```json
{
  "doc_id": "recycle_guide_sf_2024",
  "title": "San Francisco Recycling Guidelines 2024",
  "content": "Full text content...",
  "chunks": [
    {
      "chunk_id": "chunk_001",
      "text": "Plastic bottles labeled #1 (PET) and #2 (HDPE)...",
      "embedding": [0.1, 0.2, ...],
      "metadata": {
        "section": "plastics",
        "page": 3
      }
    }
  ],
  "metadata": {
    "source": "sfgov.org",
    "authority": "San Francisco Department of Environment",
    "location": "San Francisco, CA",
    "date": "2024-01-01",
    "category": "recycling_rules",
    "language": "en"
  }
}
```

**Material Properties**:
```json
{
  "material_id": "PET",
  "name": "Polyethylene Terephthalate",
  "common_names": ["PET", "PETE", "Plastic #1"],
  "properties": {
    "density": 1.38,
    "melting_point": 260,
    "glass_transition": 70,
    "recyclability": "high",
    "food_safe": true
  },
  "uses": ["bottles", "containers", "fibers"],
  "recycling_info": {
    "curbside_accepted": true,
    "recycling_code": 1,
    "process": "mechanical recycling"
  },
  "hazards": [],
  "upcycling_potential": "high"
}
```

## 4. Knowledge Graph Schema

### Node Types

**Material Node**:
```json
{
  "node_id": "mat_PET",
  "type": "Material",
  "properties": {
    "name": "PET",
    "density": 1.38,
    "melting_point": 260,
    "recyclability_score": 0.9,
    "toxicity_level": 0.1,
    "embedding": [...]
  }
}
```

**ItemType Node**:
```json
{
  "node_id": "item_plastic_bottle",
  "type": "ItemType",
  "properties": {
    "name": "plastic_bottle",
    "commonality": 0.95,
    "embedding": [...]
  }
}
```

**ProductIdea Node**:
```json
{
  "node_id": "prod_planter",
  "type": "ProductIdea",
  "properties": {
    "name": "Hanging Planter",
    "difficulty": 2,
    "time_required": 30,
    "popularity": 0.8,
    "embedding": [...]
  }
}
```

### Edge Types

**Edges** (CSV/Parquet):
```csv
source_id,target_id,edge_type,weight,properties
item_plastic_bottle,mat_PET,MADE_OF,1.0,"{}"
mat_PET,prod_planter,CAN_BE_UPCYCLED_TO,0.85,"{\"tools\": [\"scissors\", \"drill\"]}"
mat_PET,mat_HDPE,SIMILAR_TO,0.7,"{\"similarity_type\": \"chemical\"}"
```

## 5. Organization Database

### Organizations Table (PostgreSQL)

```sql
CREATE TABLE organizations (
    org_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL, -- 'charity', 'recycling_facility', 'club', etc.
    description TEXT,
    url VARCHAR(500),
    email VARCHAR(255),
    phone VARCHAR(50),
    location GEOGRAPHY(POINT, 4326),
    address JSONB,
    accepted_materials TEXT[],
    services TEXT[],
    hours JSONB,
    verified BOOLEAN DEFAULT FALSE,
    rating DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_org_location ON organizations USING GIST(location);
CREATE INDEX idx_org_type ON organizations(type);
CREATE INDEX idx_org_materials ON organizations USING GIN(accepted_materials);
```

**Example Row** (JSON representation):
```json
{
  "org_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "SF Recycling Center",
  "type": "recycling_facility",
  "description": "Full-service recycling facility accepting all materials",
  "url": "https://sfrecycling.org",
  "location": {"lat": 37.7749, "lon": -122.4194},
  "address": {
    "street": "123 Green St",
    "city": "San Francisco",
    "state": "CA",
    "zip": "94102"
  },
  "accepted_materials": ["plastic", "glass", "metal", "paper", "e-waste"],
  "services": ["drop-off", "pickup", "education"],
  "hours": {
    "monday": "8:00-17:00",
    "tuesday": "8:00-17:00",
    "saturday": "9:00-14:00"
  },
  "verified": true,
  "rating": 4.5
}
```

## 6. API Request/Response Formats

### Vision Classification Request

```json
{
  "image": "base64_encoded_image_data",
  "image_url": "https://example.com/image.jpg",
  "return_probabilities": true,
  "top_k": 3
}
```

### Vision Classification Response

```json
{
  "predictions": {
    "item_type": {
      "class": "plastic_bottle",
      "confidence": 0.95,
      "top_k": [
        {"class": "plastic_bottle", "confidence": 0.95},
        {"class": "glass_bottle", "confidence": 0.03}
      ]
    },
    "material": {
      "class": "PET",
      "confidence": 0.88
    },
    "bin_type": {
      "class": "recycle",
      "confidence": 0.92
    }
  },
  "processing_time_ms": 45
}
```

### Chat Request

```json
{
  "messages": [
    {"role": "user", "content": "How do I recycle this?"}
  ],
  "image": "base64_encoded_image_data",
  "location": {"lat": 37.7749, "lon": -122.4194},
  "max_tokens": 512,
  "temperature": 0.7
}
```

### Chat Response

```json
{
  "response": "Based on the image, this appears to be a PET plastic bottle...",
  "sources": [
    {"title": "SF Recycling Guide", "url": "..."}
  ],
  "suggestions": [
    "Find nearby recycling centers",
    "Learn about upcycling ideas"
  ],
  "processing_time_ms": 1250
}
```

