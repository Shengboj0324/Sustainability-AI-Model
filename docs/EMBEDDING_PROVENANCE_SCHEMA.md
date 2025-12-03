# Enhanced Embedding Provenance Schema

**Version**: 1.0.0  
**Date**: 2025-12-03  
**Status**: Phase 1 Implementation

---

## Overview

This document defines the comprehensive embedding provenance schema for ReleAF AI's RAG system. Every embedding stored in Qdrant will include full provenance metadata for transparency, trust, and regulatory compliance.

---

## Schema Design

### 1. Core Document Fields (Existing)
```python
{
    "content": str,          # Document text content
    "doc_id": str,           # Unique document identifier (UUID)
    "doc_type": str,         # Document type (recycling_guideline, upcycling_project, etc.)
    "source": str,           # Original source (reddit, youtube, synthetic, manual)
}
```

### 2. Enhanced Metadata Fields (NEW)
```python
{
    "metadata": {
        # Basic metadata (existing)
        "category": str,                    # Content category
        "location": Optional[str],          # Geographic location
        "authority": Optional[str],         # Authoritative source
        
        # NEW: Enhanced provenance
        "created_at": str,                  # ISO 8601 timestamp
        "updated_at": str,                  # ISO 8601 timestamp
        "update_count": int,                # Number of updates
        "quality_score": float,             # 0.0-1.0 quality assessment
        "validation_status": str,           # "verified", "pending", "flagged"
        "language": str,                    # ISO 639-1 code (en, es, fr, etc.)
    }
}
```

### 3. Embedding Metadata (NEW)
```python
{
    "embedding_metadata": {
        # Model information
        "model_name": str,                  # "BAAI/bge-large-en-v1.5"
        "model_version": str,               # "1.5.0"
        "model_checksum": str,              # SHA-256 hash of model weights
        
        # Embedding properties
        "embedding_dim": int,               # 1024
        "normalization": bool,              # True/False
        "pooling_strategy": str,            # "mean", "cls", "max"
        
        # Generation metadata
        "embedding_created_at": str,        # ISO 8601 timestamp
        "embedding_generation_time_ms": float,  # Time to generate embedding
        "content_checksum": str,            # SHA-256 hash of content
        
        # Version control
        "schema_version": str,              # "1.0.0"
        "migration_history": List[Dict],    # List of migrations
    }
}
```

### 4. Data Lineage (NEW)
```python
{
    "lineage": {
        # Source tracking
        "original_source": str,             # "reddit", "youtube", "synthetic", "manual"
        "source_url": Optional[str],        # Original URL if applicable
        "source_id": Optional[str],         # Source-specific ID
        
        # Collection metadata
        "collection_date": str,             # ISO 8601 timestamp
        "collection_method": str,           # "scraping", "api", "manual", "synthetic"
        "collector_version": str,           # Version of collection script
        
        # Processing history
        "processing_pipeline": List[str],   # ["deduplication", "quality_check", "validation"]
        "transformations": List[Dict],      # List of transformations applied
        
        # Update tracking
        "last_updated": str,                # ISO 8601 timestamp
        "update_reason": Optional[str],     # "content_change", "model_upgrade", "quality_improvement"
        "previous_versions": List[str],     # List of previous doc_id versions
    }
}
```

### 5. Trust Indicators (NEW)
```python
{
    "trust_indicators": {
        # Trust scoring
        "trust_score": float,               # 0.0-1.0 overall trust score
        "source_reliability": float,        # 0.0-1.0 source reliability
        "content_quality": float,           # 0.0-1.0 content quality
        "freshness_score": float,           # 0.0-1.0 based on age
        
        # Validation
        "human_verified": bool,             # Has human reviewed this?
        "verification_date": Optional[str], # ISO 8601 timestamp
        "verifier_id": Optional[str],       # ID of human verifier
        
        # Usage statistics
        "retrieval_count": int,             # How many times retrieved
        "positive_feedback_count": int,     # Positive user feedback
        "negative_feedback_count": int,     # Negative user feedback
        "avg_relevance_score": float,       # Average relevance from reranker
    }
}
```

---

## Complete Qdrant Payload Structure

```python
{
    # Core fields
    "content": "How to recycle plastic bottles...",
    "doc_id": "550e8400-e29b-41d4-a716-446655440000",
    "doc_type": "recycling_guideline",
    "source": "epa_gov",
    
    # Enhanced metadata
    "metadata": {
        "category": "plastic_recycling",
        "location": "USA",
        "authority": "EPA",
        "created_at": "2024-11-15T10:30:00Z",
        "updated_at": "2024-12-03T14:20:00Z",
        "update_count": 2,
        "quality_score": 0.95,
        "validation_status": "verified",
        "language": "en"
    },
    
    # Embedding metadata
    "embedding_metadata": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "model_version": "1.5.0",
        "model_checksum": "sha256:abc123...",
        "embedding_dim": 1024,
        "normalization": true,
        "pooling_strategy": "mean",
        "embedding_created_at": "2024-12-03T14:20:15Z",
        "embedding_generation_time_ms": 45.2,
        "content_checksum": "sha256:def456...",
        "schema_version": "1.0.0",
        "migration_history": []
    },
    
    # Data lineage
    "lineage": {
        "original_source": "epa_gov",
        "source_url": "https://www.epa.gov/recycle/...",
        "source_id": "epa_plastic_guide_2024",
        "collection_date": "2024-11-15T10:30:00Z",
        "collection_method": "scraping",
        "collector_version": "1.2.0",
        "processing_pipeline": ["deduplication", "quality_check", "validation"],
        "transformations": [
            {"type": "text_cleaning", "timestamp": "2024-11-15T10:31:00Z"},
            {"type": "chunking", "timestamp": "2024-11-15T10:32:00Z"}
        ],
        "last_updated": "2024-12-03T14:20:00Z",
        "update_reason": "model_upgrade",
        "previous_versions": ["550e8400-e29b-41d4-a716-446655440001"]
    },
    
    # Trust indicators
    "trust_indicators": {
        "trust_score": 0.98,
        "source_reliability": 1.0,
        "content_quality": 0.95,
        "freshness_score": 0.99,
        "human_verified": true,
        "verification_date": "2024-11-20T09:00:00Z",
        "verifier_id": "admin_001",
        "retrieval_count": 1247,
        "positive_feedback_count": 98,
        "negative_feedback_count": 2,
        "avg_relevance_score": 0.87
    }
}
```

---

## Implementation Notes

1. **Backward Compatibility**: All new fields are optional with defaults
2. **Migration Strategy**: Existing documents will be migrated with default values
3. **Performance**: Metadata stored in Qdrant payload (no additional lookups)
4. **Validation**: Pydantic models enforce schema compliance
5. **Versioning**: `schema_version` allows future schema evolution

---

## Next Steps (Phase 1)

1. ✅ Define schema (this document)
2. [ ] Implement enhanced `RetrievedDocument` dataclass
3. [ ] Implement `EmbeddingVersionTracker` class
4. [ ] Update `embed_query` method with provenance
5. [ ] Update `retrieve` method to extract provenance
6. [ ] Update API response schemas
7. [ ] Create migration script for existing data
8. [ ] Comprehensive testing and validation

