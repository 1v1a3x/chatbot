"""Main chatbot orchestrator service."""

from typing import Dict, Any

from rag_chatbot.core.config import get_settings
from rag_chatbot.core.logging_config import get_logger, set_correlation_id, sanitize_for_logging
from rag_chatbot.core.security import (
    validate_input,
    detect_prompt_injection,
    get_rate_limiter,
)
from rag_chatbot.core.exceptions import (
    SecurityError,
    InputValidationError,
    PromptInjectionError,
    RateLimitExceededError,
)
from rag_chatbot.services.search import get_search_service
from rag_chatbot.services.llm import get_llm_service

logger = get_logger(__name__)


class ChatbotService:
    """Main chatbot orchestrator."""
    
    async def process_question(
        self,
        question: str,
        correlation_id: str = None
    ) -> Dict[str, Any]:
        """Process a user question end-to-end.
        
        Args:
            question: User's question
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            Response dictionary with answer and metadata
            
        Raises:
            SecurityError: If security checks fail
            ServiceError: If service operations fail
        """
        # Set correlation ID for tracing
        cid = set_correlation_id(correlation_id)
        logger.info(f"Processing question: {sanitize_for_logging(question)}")
        
        try:
            # 1. Rate limiting
            rate_limiter = get_rate_limiter()
            if not rate_limiter.is_allowed():
                wait_time = rate_limiter.get_wait_time()
                raise RateLimitExceededError(retry_after=int(wait_time))
            
            # 2. Validate input
            try:
                validated_question = validate_input(question)
            except InputValidationError as e:
                logger.warning(f"Input validation failed: {e}")
                raise
            
            # 3. Check for prompt injection
            injection_pattern = detect_prompt_injection(validated_question)
            if injection_pattern:
                logger.warning(f"Prompt injection detected: {injection_pattern}")
                raise PromptInjectionError()
            
            # 4. Search for context
            search_service = await get_search_service()
            search_results = await search_service.search(validated_question)
            
            if not search_results:
                logger.info("No results found")
                return {
                    "success": True,
                    "answer": "I don't have any relevant information to answer your question.",
                    "sources": [],
                    "cached": False,
                    "correlation_id": cid,
                }
            
            # 5. Generate response
            llm_service = await get_llm_service()
            answer = await llm_service.generate(validated_question, search_results)
            
            # Extract source information
            sources = [
                {
                    "id": hit.get("_id"),
                    "score": hit.get("_score"),
                    "index": hit.get("_index"),
                }
                for hit in search_results
            ]
            
            logger.info("Successfully processed question")
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "correlation_id": cid,
            }
            
        except SecurityError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error processing question: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check all service health.
        
        Returns:
            Health status for all services
        """
        search_service = await get_search_service()
        llm_service = await get_llm_service()
        
        es_health = await search_service.health_check()
        llm_health = await llm_service.health_check()
        
        all_healthy = es_health["available"] and llm_health["available"]
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": {
                "elasticsearch": es_health,
                "llm": llm_health,
            },
            "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        }


# Global chatbot service instance
_chatbot_service = None


async def get_chatbot_service() -> ChatbotService:
    """Get or create global chatbot service."""
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service
