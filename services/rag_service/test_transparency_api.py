"""
Comprehensive Test Suite for Transparency API

Tests all API endpoints in transparency_api.py:
- GET /provenance/document/{doc_id}
- GET /provenance/version/{version}
- GET /provenance/audit/entity/{entity_id}
- GET /provenance/statistics

Author: ReleAF AI Team
Date: 2025-12-03
"""

import asyncio
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from transparency_api import create_transparency_router
from audit_trail import AuditTrailManager, EventType, EntityType, ActorType, Action
from version_tracker import EmbeddingVersionTracker


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_rag_service():
    """Create mock RAG service"""
    service = Mock()
    service.qdrant_client = AsyncMock()
    return service


@pytest.fixture
def mock_audit_manager():
    """Create mock audit manager"""
    manager = AsyncMock(spec=AuditTrailManager)
    manager.record_event = AsyncMock(return_value="audit_123")
    manager.get_entity_history = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def mock_version_tracker():
    """Create mock version tracker"""
    tracker = Mock(spec=EmbeddingVersionTracker)
    tracker.lock = asyncio.Lock()
    tracker.version_history = {
        "v1.0": {
            "model_name": "BAAI/bge-large-en-v1.5",
            "model_version": "1.5.0",
            "status": "active",
            "created_at": "2025-12-03T00:00:00Z",
            "embedding_dim": 1024,
            "normalization": True,
            "pooling_strategy": "mean",
            "model_checksum": "abc123",
            "migrations": []
        }
    }
    tracker.get_all_versions = AsyncMock(return_value=["v1.0"])
    tracker.get_active_versions = AsyncMock(return_value=["v1.0"])
    return tracker


@pytest.fixture
def test_app(mock_rag_service, mock_audit_manager, mock_version_tracker):
    """Create test FastAPI app with transparency router"""
    app = FastAPI()
    router = create_transparency_router(
        mock_rag_service,
        mock_audit_manager,
        mock_version_tracker
    )
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Create test client"""
    return TestClient(test_app)


# ============================================================================
# Test GET /provenance/document/{doc_id}
# ============================================================================

class TestGetDocumentProvenance:
    """Test document provenance endpoint"""
    
    def test_get_document_provenance_success(self, client, mock_rag_service):
        """Test successful document provenance retrieval"""
        # Mock Qdrant response
        mock_rag_service.qdrant_client.retrieve.return_value = [
            Mock(
                id="doc_123",
                payload={
                    "content": "test content",
                    "embedding_metadata": {"model_name": "test"},
                    "lineage": {"source": "test"},
                    "trust_indicators": {"trust_score": 0.9}
                }
            )
        ]
        
        response = client.get("/provenance/document/doc_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc_123"
        assert "embedding_metadata" in data
        assert "lineage" in data
        assert "trust_indicators" in data
    
    def test_get_document_provenance_not_found(self, client, mock_rag_service):
        """Test document not found"""
        mock_rag_service.qdrant_client.retrieve.return_value = []
        
        response = client.get("/provenance/document/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_document_provenance_no_metadata(self, client, mock_rag_service):
        """Test document without provenance metadata"""
        mock_rag_service.qdrant_client.retrieve.return_value = [
            Mock(id="doc_123", payload={"content": "test"})
        ]
        
        response = client.get("/provenance/document/doc_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["embedding_metadata"] is None
        assert data["lineage"] is None
        assert data["trust_indicators"] is None


# ============================================================================
# Test GET /provenance/version/{version}
# ============================================================================

class TestGetVersionInfo:
    """Test version info endpoint"""
    
    def test_get_version_info_success(self, client):
        """Test successful version info retrieval"""
        response = client.get("/provenance/version/v1.0")
        
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "v1.0"
        assert data["model_name"] == "BAAI/bge-large-en-v1.5"
        assert data["status"] == "active"
    
    def test_get_version_info_not_found(self, client):
        """Test version not found"""
        response = client.get("/provenance/version/nonexistent")

        assert response.status_code == 404


# ============================================================================
# Test GET /provenance/audit/entity/{entity_id}
# ============================================================================

class TestGetEntityAuditTrail:
    """Test entity audit trail endpoint"""

    def test_get_audit_trail_success(self, client, mock_audit_manager):
        """Test successful audit trail retrieval"""
        from audit_trail import AuditRecord
        from datetime import datetime, timezone

        # Mock audit records
        mock_audit_manager.get_entity_history.return_value = [
            AuditRecord(
                audit_id="audit_1",
                event_type=EventType.DOCUMENT_CREATED.value,
                timestamp=datetime.now(timezone.utc).isoformat(),
                entity_type=EntityType.DOCUMENT.value,
                entity_id="doc_123",
                actor_type=ActorType.SYSTEM.value,
                action=Action.CREATE.value,
                success=True,
                duration_ms=10.0
            )
        ]

        response = client.get("/provenance/audit/entity/doc_123")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert len(data["records"]) == 1
        assert data["records"][0]["entity_id"] == "doc_123"

    def test_get_audit_trail_with_filters(self, client, mock_audit_manager):
        """Test audit trail with filters"""
        mock_audit_manager.get_entity_history.return_value = []

        response = client.get(
            "/provenance/audit/entity/doc_123",
            params={"entity_type": "document", "limit": 50, "offset": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 50
        assert data["offset"] == 10

    def test_get_audit_trail_empty(self, client, mock_audit_manager):
        """Test empty audit trail"""
        mock_audit_manager.get_entity_history.return_value = []

        response = client.get("/provenance/audit/entity/nonexistent")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert len(data["records"]) == 0


# ============================================================================
# Test GET /provenance/statistics
# ============================================================================

class TestGetStatistics:
    """Test statistics endpoint"""

    def test_get_statistics_success(self, client, mock_version_tracker):
        """Test successful statistics retrieval"""
        response = client.get("/provenance/statistics")

        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_versions" in data
        assert "active_versions" in data
        assert data["total_versions"] == 1
        assert data["active_versions"] == 1

