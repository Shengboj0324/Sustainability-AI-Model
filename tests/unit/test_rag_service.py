"""
Unit tests for RAG Service

Tests the retrieval, re-ranking, and embedding functionality
of the RAG service with proper mocking.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import numpy as np

# Import service components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rag_service.server import (
    RAGService,
    RetrievalRequest,
    RetrievalMode,
    DocumentType,
    RetrievedDocument
)


@pytest.fixture
def mock_config():
    """Mock RAG configuration"""
    return {
        "embedding": {
            "model_name": "test-model",
            "dimension": 384
        },
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_collection"
        },
        "retrieval": {
            "dense_top_k": 10,
            "sparse_top_k": 10,
            "fusion_weights": {"dense": 0.6, "sparse": 0.4}
        },
        "reranking": {
            "model_name": "test-reranker",
            "enabled": True
        }
    }


@pytest.fixture
def rag_service(mock_config):
    """Create RAG service instance with mocked config"""
    with patch.object(RAGService, '_load_config', return_value=mock_config):
        service = RAGService()
        return service


@pytest.fixture
def sample_documents():
    """Sample retrieved documents"""
    return [
        RetrievedDocument(
            content="Plastic bottles can be recycled in most curbside programs.",
            score=0.95,
            doc_id="doc_1",
            doc_type="recycling_guideline",
            metadata={"location": "US"},
            source="EPA Guidelines"
        ),
        RetrievedDocument(
            content="PET plastic is recyclable and commonly used for bottles.",
            score=0.87,
            doc_id="doc_2",
            doc_type="material_property",
            metadata={"material": "PET"},
            source="Material Database"
        ),
        RetrievedDocument(
            content="Clean and dry bottles before recycling.",
            score=0.82,
            doc_id="doc_3",
            doc_type="recycling_guideline",
            metadata={"location": "US"},
            source="Local Guidelines"
        )
    ]


class TestRAGService:
    """Test RAG service functionality"""
    
    def test_initialization(self, rag_service, mock_config):
        """Test service initialization"""
        assert rag_service.config == mock_config
        assert rag_service.collection_name == "test_collection"
        assert rag_service.embedding_dim == 384
    
    def test_default_config(self):
        """Test default configuration fallback"""
        with patch('pathlib.Path.exists', return_value=False):
            service = RAGService()
            config = service._get_default_config()
            
            assert "embedding" in config
            assert "qdrant" in config
            assert "retrieval" in config
            assert "reranking" in config
    
    @pytest.mark.asyncio
    async def test_embed_query(self, rag_service):
        """Test query embedding"""
        # Mock embedding model
        mock_model = Mock()
        mock_embedding = np.random.rand(384)
        mock_model.encode.return_value = mock_embedding
        rag_service.embedding_model = mock_model
        
        query = "How to recycle plastic?"
        embedding = await rag_service.embed_query(query)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        mock_model.encode.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rerank_documents(self, rag_service, sample_documents):
        """Test document re-ranking"""
        # Mock reranker
        mock_reranker = Mock()
        mock_scores = np.array([0.9, 0.95, 0.85])  # Reordered scores
        mock_reranker.predict.return_value = mock_scores
        rag_service.reranker = mock_reranker
        
        query = "recycling plastic bottles"
        reranked = await rag_service.rerank_documents(query, sample_documents, top_k=3)
        
        assert len(reranked) == 3
        # Check that documents are reordered by new scores
        assert reranked[0].score == 0.95
        assert reranked[1].score == 0.9
        assert reranked[2].score == 0.85
    
    @pytest.mark.asyncio
    async def test_rerank_without_reranker(self, rag_service, sample_documents):
        """Test re-ranking fallback when reranker is None"""
        rag_service.reranker = None
        
        query = "test query"
        result = await rag_service.rerank_documents(query, sample_documents, top_k=2)
        
        # Should return original documents limited to top_k
        assert len(result) == 2
        assert result == sample_documents[:2]


class TestRetrievalRequest:
    """Test request validation"""
    
    def test_valid_request(self):
        """Test valid retrieval request"""
        request = RetrievalRequest(
            query="How to recycle plastic?",
            top_k=5,
            mode=RetrievalMode.HYBRID
        )
        
        assert request.query == "How to recycle plastic?"
        assert request.top_k == 5
        assert request.mode == RetrievalMode.HYBRID
    
    def test_location_validation(self):
        """Test location coordinate validation"""
        # Valid location
        request = RetrievalRequest(
            query="test",
            location={"lat": 37.7749, "lon": -122.4194}
        )
        assert request.location["lat"] == 37.7749
        
        # Invalid latitude
        with pytest.raises(ValueError, match="Latitude must be between"):
            RetrievalRequest(
                query="test",
                location={"lat": 100, "lon": 0}
            )
        
        # Missing coordinates
        with pytest.raises(ValueError, match="must contain 'lat' and 'lon'"):
            RetrievalRequest(
                query="test",
                location={"lat": 37.7749}
            )

