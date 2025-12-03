

import asyncio
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from server import rag_service, app
from audit_trail import EventType, EntityType
from fastapi.testclient import TestClient


class TestPhase3Integration:
    """Test complete Phase 1+2+3 integration"""
    
    def test_imports_successful(self):
        """Test that all Phase 2 components are imported"""
        assert hasattr(rag_service, 'audit_manager')
        assert hasattr(rag_service, 'version_tracker')
        assert rag_service.audit_manager is not None
        assert rag_service.version_tracker is not None
    
    def test_transparency_router_mounted(self):
        """Test that transparency API router is mounted"""
        client = TestClient(app)
        
        # Test that provenance endpoints exist
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Check OpenAPI schema includes provenance endpoints
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_schema = response.json()
        
        # Verify provenance endpoints are registered
        paths = openapi_schema.get("paths", {})
        assert "/provenance/document/{doc_id}" in paths
        assert "/provenance/version/{version}" in paths
        assert "/provenance/audit/entity/{entity_id}" in paths
        assert "/provenance/statistics" in paths
    
    @pytest.mark.asyncio
    async def test_audit_manager_initialization(self):
        """Test that audit manager initializes correctly"""
        # Initialize service
        await rag_service.initialize()
        
        try:
            # Verify audit manager is initialized
            assert rag_service.audit_manager is not None
            assert rag_service.audit_manager.json_dir.exists()
            
            # Test recording an event
            audit_id = await rag_service.audit_manager.record_event(
                event_type=EventType.DOCUMENT_ACCESSED.value,
                entity_type=EntityType.DOCUMENT.value,
                entity_id="test_doc_123",
                action="READ",
                actor_type="SYSTEM",
                success=True,
                metadata={"test": "integration"}
            )
            
            assert audit_id is not None
            assert isinstance(audit_id, str)
            
        finally:
            await rag_service.close()
    
    @pytest.mark.asyncio
    async def test_version_tracker_integration(self):
        """Test that version tracker works correctly"""
        await rag_service.initialize()
        
        try:
            # Register a version
            await rag_service.version_tracker.register_version(
                model_name="BAAI/bge-large-en-v1.5",
                model_version="1.5.0",
                embedding_dim=1024,
                normalization=True,
                pooling_strategy="mean",
                model_checksum="test_checksum_123"
            )
            
            # Get all versions
            versions = await rag_service.version_tracker.get_all_versions()
            assert len(versions) > 0
            
        finally:
            await rag_service.close()
    
    def test_health_endpoint(self):
        """Test health endpoint still works after integration"""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert data["service"] == "rag"
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint still works"""
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200


if __name__ == "__main__":
    print("Running Phase 3 Integration Tests...")
    pytest.main([__file__, "-v", "-s"])

