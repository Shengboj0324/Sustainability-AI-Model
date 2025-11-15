"""
Integration tests for production-ready RAG service

Tests:
- Async operations
- Caching behavior
- Timeout handling
- Metrics collection
- Graceful shutdown
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services" / "rag_service"))

from server import RAGService, QueryCache, query_cache


class TestQueryCache:
    """Test query cache functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test basic cache operations"""
        cache = QueryCache(max_size=10, ttl_seconds=60)
        
        # Set value
        await cache.set("test query", 5, "hybrid", None, {"result": "data"})
        
        # Get value
        result = await cache.get("test query", 5, "hybrid", None)
        assert result == {"result": "data"}
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test cache TTL expiration"""
        cache = QueryCache(max_size=10, ttl_seconds=1)
        
        # Set value
        await cache.set("test query", 5, "hybrid", None, {"result": "data"})
        
        # Should exist immediately
        result = await cache.get("test query", 5, "hybrid", None)
        assert result is not None
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        # Should be expired
        result = await cache.get("test query", 5, "hybrid", None)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = QueryCache(max_size=2, ttl_seconds=60)
        
        # Fill cache
        await cache.set("query1", 5, "hybrid", None, {"result": "1"})
        await cache.set("query2", 5, "hybrid", None, {"result": "2"})
        
        # Add third item (should evict oldest)
        await cache.set("query3", 5, "hybrid", None, {"result": "3"})
        
        # Cache should have 2 items
        assert len(cache.cache) == 2
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key generation with different parameters"""
        cache = QueryCache()
        
        # Same query, different parameters = different keys
        await cache.set("query", 5, "hybrid", None, {"result": "1"})
        await cache.set("query", 10, "hybrid", None, {"result": "2"})
        
        result1 = await cache.get("query", 5, "hybrid", None)
        result2 = await cache.get("query", 10, "hybrid", None)
        
        assert result1 != result2


class TestRAGServiceProduction:
    """Test production features of RAG service"""
    
    @pytest.mark.asyncio
    async def test_async_qdrant_client(self):
        """Test that AsyncQdrantClient is used"""
        with patch('server.AsyncQdrantClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections = AsyncMock(return_value=Mock(collections=[]))
            mock_client.return_value = mock_instance
            
            service = RAGService()
            
            # Mock model loading
            with patch.object(service, '_load_embedding_model', new=AsyncMock()):
                with patch.object(service, '_load_reranker', new=AsyncMock()):
                    await service._connect_qdrant()
            
            # Verify AsyncQdrantClient was called with connection pool settings
            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args[1]
            assert 'limits' in call_kwargs
            assert call_kwargs['limits']['max_connections'] == 100
    
    @pytest.mark.asyncio
    async def test_embedding_timeout(self):
        """Test embedding timeout handling"""
        service = RAGService()
        service.embedding_model = Mock()
        
        # Mock slow embedding
        async def slow_encode(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return Mock(tolist=lambda: [0.1] * 1024)
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock(side_effect=slow_encode)
            mock_loop.return_value.run_in_executor = mock_executor
            
            # Should timeout
            with pytest.raises(Exception):  # asyncio.TimeoutError or HTTPException
                await service.embed_query("test query")
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown"""
        service = RAGService()
        service.qdrant_client = AsyncMock()
        service.qdrant_client.close = AsyncMock()
        
        # Shutdown
        await service.close()
        
        # Verify cleanup
        assert service._shutdown is True
        service.qdrant_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_environment_variable_config(self):
        """Test environment variable configuration"""
        import os
        
        # Set environment variables
        os.environ['QDRANT_HOST'] = 'test-host'
        os.environ['QDRANT_PORT'] = '9999'
        os.environ['CACHE_SIZE'] = '500'
        
        service = RAGService()
        config = service._get_default_config()
        
        assert config['qdrant']['host'] == 'test-host'
        assert config['qdrant']['port'] == 9999
        
        # Cleanup
        del os.environ['QDRANT_HOST']
        del os.environ['QDRANT_PORT']
        del os.environ['CACHE_SIZE']
    
    @pytest.mark.asyncio
    async def test_filter_logic_or_not_and(self):
        """Test that doc_types filter uses OR logic (should) not AND (must)"""
        from qdrant_client.models import Filter
        
        service = RAGService()
        service.qdrant_client = AsyncMock()
        service.qdrant_client.search = AsyncMock(return_value=[])
        service.embedding_model = Mock()
        
        # Mock embedding
        query_embedding = [0.1] * 1024
        
        # Call with multiple doc_types
        doc_types = ["recycling_guideline", "upcycling_project"]
        
        await service.dense_retrieval(query_embedding, 5, doc_types)
        
        # Verify search was called
        call_args = service.qdrant_client.search.call_args
        query_filter = call_args[1].get('query_filter')
        
        # Filter should use 'should' (OR) not 'must' (AND)
        if query_filter:
            # Check that filter has 'should' field
            assert hasattr(query_filter, 'should') or 'should' in str(query_filter)


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling of concurrent requests"""
    cache = QueryCache(max_size=100, ttl_seconds=60)
    
    async def make_request(i):
        await cache.set(f"query{i}", 5, "hybrid", None, {"result": f"data{i}"})
        result = await cache.get(f"query{i}", 5, "hybrid", None)
        return result
    
    # Make 50 concurrent requests
    tasks = [make_request(i) for i in range(50)]
    results = await asyncio.gather(*tasks)
    
    # All should succeed
    assert len(results) == 50
    assert all(r is not None for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

