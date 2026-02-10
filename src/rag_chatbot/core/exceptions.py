"""Custom exceptions for the chatbot application."""


class ChatbotException(Exception):
    """Base exception for chatbot errors."""
    
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ConfigurationError(ChatbotException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=500, details=details)


class SecurityError(ChatbotException):
    """Raised when a security violation is detected."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=403, details=details)


class InputValidationError(SecurityError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, details=details)


class PromptInjectionError(SecurityError):
    """Raised when prompt injection is detected."""
    
    def __init__(self, message: str = "Potentially harmful input detected", details: dict = None):
        super().__init__(message, details=details)


class RateLimitExceededError(SecurityError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60, details: dict = None):
        super().__init__(message, status_code=429, details=details)
        self.retry_after = retry_after


class ServiceError(ChatbotException):
    """Raised when a service operation fails."""
    
    def __init__(self, message: str, status_code: int = 502, details: dict = None):
        super().__init__(message, status_code=status_code, details=details)


class SearchError(ServiceError):
    """Raised when search operation fails."""
    
    def __init__(self, message: str = "Search operation failed", details: dict = None):
        super().__init__(message, status_code=502, details=details)


class LLMError(ServiceError):
    """Raised when LLM operation fails."""
    
    def __init__(self, message: str = "LLM operation failed", details: dict = None):
        super().__init__(message, status_code=502, details=details)


class CacheError(ServiceError):
    """Raised when cache operation fails."""
    
    def __init__(self, message: str = "Cache operation failed", details: dict = None):
        super().__init__(message, status_code=500, details=details)
