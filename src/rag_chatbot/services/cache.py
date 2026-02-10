"""Cache implementations for search results and responses."""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime, timedelta

from rag_chatbot.core.config import get_settings
from rag_chatbot.core.logging_config import get_logger
from rag_chatbot.core.exceptions import CacheError

logger = get_logger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache with TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache implementation."""
    
    def __init__(self):
        self._cache: dict = {}
        self._expiry: dict = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        now = datetime.utcnow()
        
        # Check if expired
        if key in self._expiry and self._expiry[key] < now:
            await self.delete(key)
            return None
        
        value = self._cache.get(key)
        if value is not None:
            logger.debug(f"Cache hit for key: {key[:16]}...")
        
        return value
    
    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in memory cache."""
        self._cache[key] = value
        self._expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)
        logger.debug(f"Cache set for key: {key[:16]}...")
    
    async def delete(self, key: str) -> None:
        """Delete value from memory cache."""
        self._cache.pop(key, None)
        self._expiry.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._expiry.clear()
        logger.info("Memory cache cleared")


class CacheService:
    """High-level cache service."""
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        """Initialize cache service.
        
        Args:
            backend: Cache backend (defaults to memory)
        """
        settings = get_settings()
        self.backend = backend or MemoryCache()
        self.enabled = settings.cache_enabled
        self.default_ttl = settings.cache_ttl
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        key_data = json.dumps(data, sort_keys=True)
        hash_obj = hashlib.sha256(key_data.encode())
        return f"{prefix}:{hash_obj.hexdigest()[:32]}"
    
    async def get_search_results(self, query: str) -> Optional[list]:
        """Get cached search results."""
        if not self.enabled:
            return None
        
        key = self._generate_key("search", query)
        return await self.backend.get(key)
    
    async def set_search_results(self, query: str, results: list) -> None:
        """Cache search results."""
        if not self.enabled:
            return
        
        key = self._generate_key("search", query)
        await self.backend.set(key, results, self.default_ttl)
    
    async def get_completion(self, query: str, context_hash: str) -> Optional[str]:
        """Get cached completion."""
        if not self.enabled:
            return None
        
        key = self._generate_key("completion", {"query": query, "context": context_hash})
        return await self.backend.get(key)
    
    async def set_completion(self, query: str, context_hash: str, completion: str) -> None:
        """Cache completion."""
        if not self.enabled:
            return
        
        key = self._generate_key("completion", {"query": query, "context": context_hash})
        await self.backend.set(key, completion, self.default_ttl)
    
    async def clear(self) -> None:
        """Clear all cache."""
        await self.backend.clear()


# Global cache service instance
_cache_service: Optional[CacheService] = None


async def get_cache_service() -> CacheService:
    """Get or create global cache service."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
