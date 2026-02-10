"""Security utilities for input validation and sanitization."""

import html
import re
from typing import Optional, List

import bleach

from rag_chatbot.core.config import get_settings
from rag_chatbot.core.exceptions import InputValidationError, PromptInjectionError

# Prompt injection patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(?:previous|above|all)\s+(?:instructions?|context)",
    r"disregard\s+(?:previous|above|all)\s+(?:instructions?|context)",
    r"forget\s+(?:previous|above|all)\s+(?:instructions?|context)",
    r"system\s*:\s*",
    r"user\s*:\s*",
    r"assistant\s*:\s*",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"you are\s+(?:now\s+)?(?:an?\s+)?(?:helpful|friendly|ai|assistant)",
    r"pretend\s+(?:to\s+)?be",
    r"act\s+as\s+(?:if\s+)?you\s+are",
    r"new\s+instructions\s*:",
    r"override\s+(?:previous|all)\s+instructions",
    r"\[\s*SYSTEM\s*\]",
    r"\[\s*INSTRUCTION\s*\]",
    r"###\s*(?:system|user|assistant|instructions)",
    r"<!--\s*system",
    r"\{\s*\"role\"\s*:\s*\"system\"",
]


def validate_input(text: str, max_length: Optional[int] = None) -> str:
    """Validate and sanitize user input.
    
    Args:
        text: Input text to validate
        max_length: Maximum allowed length (uses config default if None)
        
    Returns:
        Sanitized text
        
    Raises:
        InputValidationError: If validation fails
    """
    settings = get_settings()
    max_length = max_length or settings.max_query_length
    
    if not text:
        raise InputValidationError("Input cannot be empty")
    
    if len(text) > max_length:
        raise InputValidationError(
            f"Input too long: {len(text)} characters (max: {max_length})"
        )
    
    # Check for control characters
    if any(ord(char) < 32 and char not in '\t\n\r' for char in text):
        raise InputValidationError("Input contains invalid control characters")
    
    # Sanitize HTML
    text = bleach.clean(text, tags=[], strip=True)
    
    # Escape HTML entities
    text = html.escape(text)
    
    return text.strip()


def detect_prompt_injection(text: str) -> Optional[str]:
    """Detect prompt injection attempts.
    
    Args:
        text: Input text to check
        
    Returns:
        Matched pattern if injection detected, None otherwise
    """
    text_lower = text.lower()
    
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return pattern
    
    return None


def sanitize_for_logging(text: str, max_length: int = 100) -> str:
    """Sanitize text for safe logging.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length to log
        
    Returns:
        Sanitized text safe for logging
    """
    if not text:
        return ""
    
    # Truncate
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # Remove newlines
    text = text.replace('\n', ' ').replace('\r', '')
    
    return text


def validate_index_name(name: str) -> bool:
    """Validate Elasticsearch index name.
    
    Args:
        name: Index name to validate
        
    Returns:
        True if valid, False otherwise
    """
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', name))


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict = {}
    
    def is_allowed(self, identifier: str = "default") -> bool:
        """Check if request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier for rate limiting
            
        Returns:
            True if allowed, False otherwise
        """
        import time
        
        now = time.time()
        
        # Get or create request list for identifier
        if identifier not in self._requests:
            self._requests[identifier] = []
        
        # Remove old requests
        self._requests[identifier] = [
            req_time for req_time in self._requests[identifier]
            if req_time > now - self.window_seconds
        ]
        
        # Check limit
        if len(self._requests[identifier]) >= self.max_requests:
            return False
        
        # Record request
        self._requests[identifier].append(now)
        return True
    
    def get_wait_time(self, identifier: str = "default") -> float:
        """Get seconds until next request is allowed.
        
        Args:
            identifier: Unique identifier
            
        Returns:
            Seconds to wait (0 if allowed)
        """
        import time
        
        now = time.time()
        
        if identifier not in self._requests:
            return 0.0
        
        relevant_requests = [
            req_time for req_time in self._requests[identifier]
            if req_time > now - self.window_seconds
        ]
        
        if len(relevant_requests) < self.max_requests:
            return 0.0
        
        oldest = min(relevant_requests)
        return max(0.0, self.window_seconds - (now - oldest))


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        _rate_limiter = RateLimiter(
            settings.rate_limit_requests,
            settings.rate_limit_window
        )
    return _rate_limiter
