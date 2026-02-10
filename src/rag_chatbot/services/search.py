"""Async Elasticsearch search service."""

from typing import Any, Dict, List

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from rag_chatbot.core.config import get_settings
from rag_chatbot.core.logging_config import get_logger, sanitize_for_logging
from rag_chatbot.core.exceptions import SearchError
from rag_chatbot.services.cache import get_cache_service

logger = get_logger(__name__)


class SearchService:
    """Elasticsearch search service with caching and retry logic."""
    
    def __init__(self):
        """Initialize search service."""
        settings = get_settings()
        self.settings = settings
        self.client = AsyncElasticsearch(
            hosts=[settings.es_url],
            api_key=settings.es_api_key,
            verify_certs=settings.es_verify_ssl,
            request_timeout=settings.es_timeout,
            retry_on_timeout=True,
            max_retries=settings.es_max_retries,
        )
        self.index_source_fields = {settings.es_index: ["content"]}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ESConnectionError, ConnectionError)),
        reraise=True,
    )
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Perform semantic search with caching.
        
        Args:
            query: Search query string
            
        Returns:
            List of search results
            
        Raises:
            SearchError: If search fails
        """
        logger.info(f"Searching for: {sanitize_for_logging(query)}...")
        
        # Check cache
        cache = await get_cache_service()
        cached_results = await cache.get_search_results(query)
        if cached_results is not None:
            logger.info(f"Returning {len(cached_results)} cached results")
            return cached_results
        
        # Build query
        es_query = {
            "retriever": {
                "standard": {
                    "query": {
                        "semantic": {
                            "field": "content",
                            "query": query
                        }
                    }
                }
            },
            "highlight": {
                "fields": {
                    "content": {
                        "type": "semantic",
                        "number_of_fragments": 2,
                        "order": "score",
                    }
                }
            },
            "size": self.settings.es_results_size,
        }
        
        try:
            result = await self.client.search(
                index=self.settings.es_index,
                body=es_query,
                request_timeout=self.settings.es_timeout,
            )
            hits = result["hits"]["hits"]
            logger.info(f"Found {len(hits)} results")
            
            # Cache results
            await cache.set_search_results(query, hits)
            
            return hits
            
        except Exception as e:
            logger.error(f"Search failed: {sanitize_for_logging(str(e), 200)}")
            raise SearchError("Failed to search documents. Please try again later.") from e
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Elasticsearch health.
        
        Returns:
            Health status dictionary
        """
        try:
            health = await self.client.cluster.health()
            return {
                "status": health["status"],
                "available": True,
                "cluster_name": health.get("cluster_name", "unknown"),
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unavailable",
                "available": False,
                "error": str(e),
            }
    
    async def close(self) -> None:
        """Close Elasticsearch connection."""
        await self.client.close()


# Global search service instance
_search_service: Optional[SearchService] = None


async def get_search_service() -> SearchService:
    """Get or create global search service."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
