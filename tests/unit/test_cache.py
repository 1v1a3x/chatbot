"""Tests for cache service."""

import pytest
import pytest_asyncio

from rag_chatbot.services.cache import MemoryCache, CacheService


class TestMemoryCache:
    """Test memory cache backend."""
    
    @pytest_asyncio.fixture
    async def cache(self):
        """Create fresh cache instance."""
        cache = MemoryCache()
        yield cache
        await cache.clear()
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test setting and getting values."""
        await cache.set("key1", "value1", ttl=300)
        result = await cache.get("key1")
        assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, cache):
        """Test getting non-existent key returns None."""
        result = await cache.get("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test deleting values."""
        await cache.set("key1", "value1", ttl=300)
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_expired_value(self, cache):
        """Test expired values are not returned."""
        await cache.set("key1", "value1", ttl=0)  # Already expired
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clearing all values."""
        await cache.set("key1", "value1", ttl=300)
        await cache.set("key2", "value2", ttl=300)
        await cache.clear()
        
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_complex_values(self, cache):
        """Test caching complex data types."""
        complex_data = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "string": "test",
        }
        await cache.set("complex", complex_data, ttl=300)
        result = await cache.get("complex")
        assert result == complex_data


class TestCacheService:
    """Test cache service."""
    
    @pytest_asyncio.fixture
    async def service(self):
        """Create fresh service instance."""
        service = CacheService()
        yield service
        await service.clear()
    
    @pytest.mark.asyncio
    async def test_search_results_caching(self, service):
        """Test search results caching."""
        query = "machine learning"
        results = [{"_id": "1", "_score": 1.0}]
        
        # Initially not cached
        cached = await service.get_search_results(query)
        assert cached is None
        
        # Set cache
        await service.set_search_results(query, results)
        
        # Should be cached now
        cached = await service.get_search_results(query)
        assert cached == results
    
    @pytest.mark.asyncio
    async def test_completion_caching(self, service):
        """Test completion caching."""
        query = "What is AI?"
        context_hash = "abc123"
        completion = "AI is artificial intelligence."
        
        # Initially not cached
        cached = await service.get_completion(query, context_hash)
        assert cached is None
        
        # Set cache
        await service.set_completion(query, context_hash, completion)
        
        # Should be cached now
        cached = await service.get_completion(query, context_hash)
        assert cached == completion
    
    @pytest.mark.asyncio
    async def test_different_queries_different_cache(self, service):
        """Test different queries have separate cache entries."""
        query1 = "machine learning"
        query2 = "deep learning"
        results1 = [{"id": "1"}]
        results2 = [{"id": "2"}]
        
        await service.set_search_results(query1, results1)
        await service.set_search_results(query2, results2)
        
        assert await service.get_search_results(query1) == results1
        assert await service.get_search_results(query2) == results2
