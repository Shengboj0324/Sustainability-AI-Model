# Dataset Guide for ReleAF AI

## Overview

This guide covers all datasets needed for training the ReleAF AI models.

## Vision Datasets

### 1. TrashNet

**Description**: 6-class waste classification dataset with 2,527 images.

**Classes**:
- Glass
- Paper
- Cardboard
- Plastic
- Metal
- Trash

**Download**:
```bash
wget https://github.com/garythung/trashnet/archive/master.zip
unzip master.zip -d data/raw/images/trashnet/
```

**Citation**:
```
@misc{trashnet,
  author = {Gary Thung and Mindy Yang},
  title = {TrashNet},
  year = {2016},
  publisher = {GitHub},
  url = {https://github.com/garythung/trashnet}
}
```

### 2. TACO (Trash Annotations in Context)

**Description**: 1,500+ images with 60 waste categories, annotated with bounding boxes.

**Download**: Requires registration at http://tacodataset.org/

**Usage**: Object detection training

**Citation**:
```
@article{taco2020,
  title={TACO: Trash Annotations in Context for Litter Detection},
  author={Proença, Pedro F and Simões, Pedro},
  journal={arXiv preprint arXiv:2003.06975},
  year={2020}
}
```

### 3. Kaggle Garbage Classification

**Description**: 15,000+ images across 12 categories.

**Download**:
```bash
kaggle datasets download -d asdasdasasdas/garbage-classification
```

### 4. Humans in the Loop - Recycling Dataset

**Description**: High-quality annotated recycling images.

**Download**: Available on Roboflow Universe

### 5. Custom Dataset Collection

**Guidelines for collecting your own data**:

1. **Image Quality**:
   - Resolution: Minimum 640x640
   - Lighting: Well-lit, avoid extreme shadows
   - Background: Varied backgrounds for robustness

2. **Diversity**:
   - Different angles and orientations
   - Various lighting conditions
   - Multiple brands and variations
   - Clean and dirty items

3. **Annotation**:
   - Use tools like LabelImg, CVAT, or Roboflow
   - Follow COCO or YOLO format
   - Include material type labels

## Text Datasets for LLM

### 1. Recycling Guidelines

**Sources**:
- EPA (Environmental Protection Agency) - https://www.epa.gov/recycle
- Local government websites
- Waste management company guidelines

**Collection**:
```bash
python scripts/scrape_recycling_guidelines.py \
  --sources data/sources/recycling_sources.txt \
  --output data/raw/text/recycling_guidelines/
```

### 2. Upcycling Projects

**Sources**:
- Instructables.com
- Pinterest DIY boards
- YouTube transcripts
- Craft blogs

**Format**:
```json
{
  "title": "Plastic Bottle Planter",
  "materials": ["plastic bottle", "soil", "seeds"],
  "tools": ["scissors", "marker"],
  "difficulty": "easy",
  "time": "30 minutes",
  "steps": ["Step 1...", "Step 2..."],
  "safety_notes": ["Wash bottle thoroughly..."]
}
```

### 3. Material Properties

**Sources**:
- Material safety data sheets (MSDS)
- Chemistry databases
- Recycling industry publications

**Example**:
```json
{
  "material": "PET",
  "full_name": "Polyethylene Terephthalate",
  "recycling_code": 1,
  "properties": {
    "density": 1.38,
    "melting_point": 260,
    "recyclable": true
  },
  "common_uses": ["bottles", "containers"],
  "recycling_process": "mechanical recycling"
}
```

### 4. Sustainability Q&A

**Creation Methods**:

1. **Manual Curation**: Expert-written Q&A pairs
2. **Synthetic Generation**: Use GPT-4 to generate from guidelines
3. **Community Sourcing**: Reddit, StackExchange

**Format**:
```json
{
  "question": "Can I recycle pizza boxes?",
  "answer": "Pizza boxes can be recycled if they're clean...",
  "category": "recycling_rules",
  "verified": true,
  "source": "EPA Guidelines"
}
```

### 5. Organization Data

**Sources**:
- Charity Navigator
- Earth911 database
- Local environmental organization directories
- Google Places API

**Schema**:
```sql
CREATE TABLE organizations (
    org_id UUID PRIMARY KEY,
    name VARCHAR(255),
    type VARCHAR(50),
    location GEOGRAPHY(POINT),
    accepted_materials TEXT[],
    services TEXT[],
    verified BOOLEAN
);
```

## Knowledge Graph Data

### Node Types and Relationships

**Materials**:
```csv
material_id,name,type,recyclable,properties
mat_001,PET,plastic,true,"{""density"": 1.38}"
mat_002,HDPE,plastic,true,"{""density"": 0.95}"
```

**Upcycling Paths**:
```csv
source_material,target_product,difficulty,tools_required
PET,planter,easy,"[""scissors""]"
glass_jar,candle_holder,easy,"[""wax"", ""wick""]"
```

## Data Preparation Pipeline

### 1. Vision Data Preparation

```bash
# Organize images
python scripts/organize_vision_data.py \
  --input data/raw/images/ \
  --output data/processed/vision_cls/ \
  --split 0.7 0.15 0.15

# Create YOLO annotations
python scripts/convert_to_yolo.py \
  --input data/annotations/vision/coco.json \
  --output data/processed/vision_det/
```

### 2. LLM Data Preparation

```bash
# Convert to chat format
python training/llm/data_prep.py \
  --input data/raw/text/ \
  --output data/processed/llm_sft/ \
  --format chat
```

### 3. RAG Index Building

```bash
# Build vector index
python scripts/build_rag_index.sh
```

### 4. Knowledge Graph Construction

```bash
# Build graph from data
python services/kg_service/build_graph.py \
  --materials data/raw/text/material_properties.json \
  --upcycling data/raw/text/upcycling_projects/ \
  --output data/processed/kg/
```

## Data Quality Guidelines

### Vision Data

✅ **Good**:
- Clear, well-lit images
- Single object in focus (for classifier)
- Multiple objects in natural scenes (for detector)
- Varied backgrounds and angles

❌ **Avoid**:
- Blurry or low-resolution images
- Extreme lighting conditions
- Heavily occluded objects
- Mislabeled data

### Text Data

✅ **Good**:
- Factually accurate information
- Clear, concise language
- Properly cited sources
- Diverse examples

❌ **Avoid**:
- Outdated information
- Contradictory guidelines
- Unsafe recommendations
- Biased or incomplete data

## Data Augmentation

### Vision Augmentation

Applied during training (see `configs/vision_cls.yaml`):
- Random crops and resizing
- Horizontal flips
- Color jittering
- Rotation
- Mixup and CutMix

### Text Augmentation

```python
# Paraphrasing
# Back-translation
# Synonym replacement
# Question generation from documents
```

## Dataset Statistics

Track your dataset composition:

```bash
python scripts/dataset_stats.py --data data/processed/
```

Expected output:
```
Vision Classifier:
  Train: 10,000 images
  Val: 2,000 images
  Test: 2,000 images
  Classes: 20

Object Detector:
  Train: 5,000 images
  Val: 1,000 images
  Annotations: 25,000 boxes

LLM Training:
  Total examples: 50,000
  Sustainability Q&A: 20,000
  Upcycling: 15,000
  Org routing: 10,000
  Safety: 5,000
```

## Continuous Data Collection

Set up pipelines for ongoing data collection:

1. **User Feedback Loop**: Collect user-uploaded images (with consent)
2. **Web Scraping**: Regular updates from recycling guidelines
3. **Community Contributions**: Crowdsourced upcycling ideas
4. **Active Learning**: Identify and label uncertain predictions

## Data Privacy and Ethics

- ✅ Obtain user consent for data collection
- ✅ Anonymize personal information
- ✅ Comply with GDPR and data protection laws
- ✅ Provide opt-out mechanisms
- ✅ Regular data audits for bias

## References

- [TACO Dataset](http://tacodataset.org/)
- [TrashNet](https://github.com/garythung/trashnet)
- [EPA Recycling Guidelines](https://www.epa.gov/recycle)
- [Material Properties Database](https://www.matweb.com/)

