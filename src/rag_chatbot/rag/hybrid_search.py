"""Hybrid search combining keyword and semantic search."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio

from rag_chatbot.core.config import get_settings
from rag_chatbot.core.logging_config import get_logger
from rag_chatbot.core.exceptions import SearchError

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Search result with relevance scores."""

    id: str
    content: str
    score: float
    semantic_score: float
    keyword_score: float
    metadata: Dict[str, Any]


class HybridSearcher:
    """Hybrid search combining BM25 keyword search and semantic search.

    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """

    def __init__(
        self,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        rrf_k: int = 60,
    ):
        """Initialize hybrid searcher.

        Args:
            keyword_weight: Weight for keyword search (0-1)
            semantic_weight: Weight for semantic search (0-1)
            rrf_k: RRF constant (higher = more weight to top results)
        """
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.rrf_k = rrf_k
        self.settings = get_settings()

    async def search(
        self,
        query: str,
        es_client,
        index: str,
        size: int = 10,
    ) -> List[SearchResult]:
        """Perform hybrid search.

        Args:
            query: Search query
            es_client: Elasticsearch client
            index: Index to search
            size: Number of results

        Returns:
            List of fused search results
        """
        # Run both searches in parallel
        keyword_task = self._keyword_search(query, es_client, index, size * 2)
        semantic_task = self._semantic_search(query, es_client, index, size * 2)

        keyword_results, semantic_results = await asyncio.gather(
            keyword_task,
            semantic_task,
            return_exceptions=True,
        )

        # Handle errors
        if isinstance(keyword_results, Exception):
            logger.error(f"Keyword search failed: {keyword_results}")
            keyword_results = []

        if isinstance(semantic_results, Exception):
            logger.error(f"Semantic search failed: {semantic_results}")
            semantic_results = []

        # Fuse results
        fused = self._reciprocal_rank_fusion(
            keyword_results,
            semantic_results,
            size,
        )

        logger.info(f"Hybrid search found {len(fused)} results")
        return fused

    async def _keyword_search(
        self,
        query: str,
        es_client,
        index: str,
        size: int,
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "title", "metadata.*"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            },
            "size": size,
        }

        response = await es_client.search(index=index, body=search_body)
        return response["hits"]["hits"]

    async def _semantic_search(
        self,
        query: str,
        es_client,
        index: str,
        size: int,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search."""
        search_body = {
            "retriever": {
                "standard": {
                    "query": {
                        "semantic": {
                            "field": "content",
                            "query": query,
                        }
                    }
                }
            },
            "size": size,
        }

        response = await es_client.search(index=index, body=search_body)
        return response["hits"]["hits"]

    def _reciprocal_rank_fusion(
        self,
        keyword_results: List[Dict],
        semantic_results: List[Dict],
        size: int,
    ) -> List[SearchResult]:
        """Fuse results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each list
        """
        # Create rank dictionaries
        keyword_ranks = {hit["_id"]: rank for rank, hit in enumerate(keyword_results)}
        semantic_ranks = {hit["_id"]: rank for rank, hit in enumerate(semantic_results)}

        # Get all unique IDs
        all_ids = set(keyword_ranks.keys()) | set(semantic_ranks.keys())

        # Calculate RRF scores
        fused_scores = {}
        for doc_id in all_ids:
            score = 0.0
            keyword_rank = keyword_ranks.get(doc_id)
            semantic_rank = semantic_ranks.get(doc_id)

            if keyword_rank is not None:
                score += self.keyword_weight * (1 / (self.rrf_k + keyword_rank))

            if semantic_rank is not None:
                score += self.semantic_weight * (1 / (self.rrf_k + semantic_rank))

            # Get content from either result
            content = ""
            metadata = {}
            keyword_score = 0.0
            semantic_score = 0.0

            for hit in keyword_results:
                if hit["_id"] == doc_id:
                    content = hit.get("_source", {}).get("content", "")
                    metadata = hit.get("_source", {}).get("metadata", {})
                    keyword_score = hit.get("_score", 0)
                    break

            for hit in semantic_results:
                if hit["_id"] == doc_id:
                    if not content:
                        content = hit.get("_source", {}).get("content", "")
                        metadata = hit.get("_source", {}).get("metadata", {})
                    semantic_score = hit.get("_score", 0)
                    break

            fused_scores[doc_id] = SearchResult(
                id=doc_id,
                content=content,
                score=score,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                metadata=metadata,
            )

        # Sort by RRF score and return top results
        sorted_results = sorted(
            fused_scores.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return sorted_results[:size]


class QueryRewriter:
    """Rewrite queries for better search results."""

    def __init__(self):
        """Initialize query rewriter."""
        self.expansion_templates = {
            "what is": ["define", "explain", "description of"],
            "how to": ["steps to", "guide for", "tutorial on"],
            "best": ["top", "recommended", "optimal"],
        }

    def rewrite(self, query: str) -> List[str]:
        """Rewrite query into multiple variations.

        Args:
            query: Original query

        Returns:
            List of query variations
        """
        variations = [query]
        query_lower = query.lower()

        # Add expansions
        for key, expansions in self.expansion_templates.items():
            if key in query_lower:
                for expansion in expansions:
                    new_query = query_lower.replace(key, expansion)
                    if new_query not in variations:
                        variations.append(new_query)

        # Add quoted phrases for exact matching
        words = query.split()
        if len(words) > 3:
            # Add version with quoted key terms
            key_terms = " ".join(words[:3])
            variations.append(f'"{key_terms}" {" ".join(words[3:])}')

        return variations[:5]  # Limit variations


class Reranker:
    """Rerank search results using cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed, skipping reranking")
                return None
        return self._model

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Rerank results by relevance.

        Args:
            query: Original query
            results: Search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        model = self._load_model()
        if model is None or not results:
            return results[:top_k]

        # Prepare pairs
        pairs = [[query, result.content] for result in results]

        # Score pairs
        scores = model.predict(pairs)

        # Sort by new scores
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [r for r, s in scored_results[:top_k]]
