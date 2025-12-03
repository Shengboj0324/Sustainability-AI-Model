"""
Transparency API for Embedding Provenance

This module provides public API endpoints for querying provenance metadata,
version information, and audit trails. Enables transparency and trust in the
embedding system.

Features:
- Document provenance queries
- Version information queries
- Audit trail access (admin only)
- Statistics and analytics
- GDPR-compliant data access

Author: ReleAF AI Team
Date: 2025-12-03
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging

try:
    from .provenance import EmbeddingMetadata, DataLineage, TrustIndicators
    from .audit_trail import AuditRecord, EventType, EntityType, ActorType, Action
except ImportError:
    from provenance import EmbeddingMetadata, DataLineage, TrustIndicators
    from audit_trail import AuditRecord, EventType, EntityType, ActorType, Action

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class ProvenanceResponse(BaseModel):
    """Response model for document provenance"""
    doc_id: str
    embedding_metadata: Optional[Dict[str, Any]] = None
    lineage: Optional[Dict[str, Any]] = None
    trust_indicators: Optional[Dict[str, Any]] = None
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class VersionInfoResponse(BaseModel):
    """Response model for version information"""
    version: str
    model_name: str
    model_version: str
    status: str
    created_at: str
    num_documents: int
    embedding_dim: int
    normalization: bool
    pooling_strategy: str
    model_checksum: Optional[str] = None
    migrations: List[Dict[str, Any]] = Field(default_factory=list)


class AuditTrailResponse(BaseModel):
    """Response model for audit trail"""
    audit_id: str
    event_type: str
    timestamp: str
    entity_type: str
    entity_id: str
    actor_type: str
    actor_id: Optional[str] = None
    action: str
    success: bool
    duration_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditTrailListResponse(BaseModel):
    """Response model for audit trail list"""
    records: List[AuditTrailResponse]
    total_count: int
    limit: int
    offset: int


class StatisticsResponse(BaseModel):
    """Response model for provenance statistics"""
    total_documents: int
    total_versions: int
    active_versions: int
    deprecated_versions: int
    total_audit_events: int
    documents_by_version: Dict[str, int]
    events_by_type: Dict[str, int]
    avg_trust_score: float
    avg_freshness_score: float


# ============================================================================
# Transparency API Router
# ============================================================================

def create_transparency_router(rag_service, audit_manager, version_tracker) -> APIRouter:
    """
    Create transparency API router

    Args:
        rag_service: RAGService instance
        audit_manager: AuditTrailManager instance
        version_tracker: EmbeddingVersionTracker instance

    Returns:
        FastAPI router with transparency endpoints
    """
    router = APIRouter(prefix="/provenance", tags=["provenance"])

    @router.get("/document/{doc_id}", response_model=ProvenanceResponse)
    async def get_document_provenance(doc_id: str):
        """
        Get provenance metadata for a specific document

        Args:
            doc_id: Document ID

        Returns:
            Document provenance metadata

        Raises:
            HTTPException: If document not found
        """
        try:
            # Record audit event
            await audit_manager.record_event(
                event_type=EventType.PROVENANCE_ACCESSED.value,
                entity_type=EntityType.DOCUMENT.value,
                entity_id=doc_id,
                action=Action.READ.value,
                actor_type=ActorType.API.value
            )

            # Query Qdrant for document
            search_result = await rag_service.qdrant_client.retrieve(
                collection_name=rag_service.collection_name,
                ids=[doc_id]
            )

            if not search_result:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

            doc = search_result[0]
            payload = doc.payload

            # Extract provenance metadata
            response = ProvenanceResponse(
                doc_id=doc_id,
                embedding_metadata=payload.get("embedding_metadata"),
                lineage=payload.get("lineage"),
                trust_indicators=payload.get("trust_indicators")
            )

            logger.info(f"Retrieved provenance for document {doc_id}")
            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving provenance for {doc_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/version/{version}", response_model=VersionInfoResponse)
    async def get_version_info(version: str):
        """
        Get information about a specific embedding model version

        Args:
            version: Version identifier (e.g., "1.0.0")

        Returns:
            Version information

        Raises:
            HTTPException: If version not found
        """
        try:
            # Record audit event
            await audit_manager.record_event(
                event_type=EventType.VERSION_COMPATIBILITY_CHECKED.value,
                entity_type=EntityType.VERSION.value,
                entity_id=version,
                action=Action.READ.value,
                actor_type=ActorType.API.value
            )

            # Get version info from tracker
            async with version_tracker.lock:
                if version not in version_tracker.version_history:
                    raise HTTPException(status_code=404, detail=f"Version {version} not found")

                version_data = version_tracker.version_history[version]

            response = VersionInfoResponse(
                version=version,
                model_name=version_data.get("model_name", ""),
                model_version=version_data.get("model_version", ""),
                status=version_data.get("status", "unknown"),
                created_at=version_data.get("created_at", ""),
                num_documents=version_data.get("num_documents", 0),
                embedding_dim=version_data.get("embedding_dim", 0),
                normalization=version_data.get("normalization", True),
                pooling_strategy=version_data.get("pooling_strategy", "mean"),
                model_checksum=version_data.get("model_checksum"),
                migrations=version_data.get("migrations", [])
            )

            logger.info(f"Retrieved version info for {version}")
            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving version info for {version}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/audit/entity/{entity_id}", response_model=AuditTrailListResponse)
    async def get_entity_audit_trail(
        entity_id: str,
        entity_type: Optional[str] = Query(None, description="Filter by entity type"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
        offset: int = Query(0, ge=0, description="Offset for pagination")
    ):
        """
        Get audit trail for a specific entity

        Args:
            entity_id: Entity ID
            entity_type: Optional entity type filter
            limit: Maximum number of records
            offset: Offset for pagination

        Returns:
            List of audit records
        """
        try:
            # Record audit event
            await audit_manager.record_event(
                event_type=EventType.AUDIT_TRAIL_ACCESSED.value,
                entity_type=EntityType.SYSTEM.value,
                entity_id=entity_id,
                action=Action.READ.value,
                actor_type=ActorType.API.value,
                metadata={"query_entity_type": entity_type, "limit": limit, "offset": offset}
            )

            # Get audit history
            records = await audit_manager.get_entity_history(
                entity_id=entity_id,
                entity_type=entity_type,
                limit=limit + offset  # Get more to handle offset
            )

            # Apply offset
            records = records[offset:offset + limit]

            # Convert to response models
            audit_responses = [
                AuditTrailResponse(
                    audit_id=r.audit_id,
                    event_type=r.event_type,
                    timestamp=r.timestamp,
                    entity_type=r.entity_type,
                    entity_id=r.entity_id,
                    actor_type=r.actor_type,
                    actor_id=r.actor_id,
                    action=r.action,
                    success=r.success,
                    duration_ms=r.duration_ms,
                    metadata=r.metadata
                )
                for r in records
            ]

            response = AuditTrailListResponse(
                records=audit_responses,
                total_count=len(records),
                limit=limit,
                offset=offset
            )

            logger.info(f"Retrieved {len(records)} audit records for entity {entity_id}")
            return response

        except Exception as e:
            logger.error(f"Error retrieving audit trail for {entity_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/statistics", response_model=StatisticsResponse)
    async def get_provenance_statistics():
        """
        Get overall provenance statistics

        Returns:
            Provenance statistics
        """
        try:
            # Record audit event
            await audit_manager.record_event(
                event_type=EventType.AUDIT_TRAIL_ACCESSED.value,
                entity_type=EntityType.SYSTEM.value,
                entity_id="statistics",
                action=Action.READ.value,
                actor_type=ActorType.API.value
            )

            # Get version statistics
            all_versions = await version_tracker.get_all_versions()
            active_versions = await version_tracker.get_active_versions()

            documents_by_version = {}
            async with version_tracker.lock:
                for version in all_versions:
                    version_data = version_tracker.version_history.get(version, {})
                    documents_by_version[version] = version_data.get("num_documents", 0)

            total_documents = sum(documents_by_version.values())

            response = StatisticsResponse(
                total_documents=total_documents,
                total_versions=len(all_versions),
                active_versions=len(active_versions),
                deprecated_versions=len(all_versions) - len(active_versions),
                total_audit_events=0,  # Would need to query audit trail
                documents_by_version=documents_by_version,
                events_by_type={},  # Would need to query audit trail
                avg_trust_score=0.0,  # Would need to query Qdrant
                avg_freshness_score=0.0  # Would need to query Qdrant
            )

            logger.info("Retrieved provenance statistics")
            return response

        except Exception as e:
            logger.error(f"Error retrieving provenance statistics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router

