"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    app_name: str = Field(default="RAG Chatbot", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    
    # Elasticsearch
    es_url: str = Field(..., description="Elasticsearch URL")
    es_api_key: str = Field(..., description="Elasticsearch API key")
    es_index: str = Field(default="ttintegration", description="Elasticsearch index name")
    es_verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    es_timeout: int = Field(default=30, ge=1, le=300, description="Elasticsearch timeout in seconds")
    es_max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    es_results_size: int = Field(default=3, ge=1, le=20, description="Number of search results")
    
    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    openai_timeout: int = Field(default=30, ge=1, le=300, description="OpenAI timeout in seconds")
    openai_max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    openai_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    openai_max_tokens: int = Field(default=2000, ge=100, le=4000, description="Maximum tokens in response")
    
    # Security
    max_query_length: int = Field(default=1000, ge=100, le=5000, description="Maximum query length")
    max_context_length: int = Field(default=8000, ge=1000, le=20000, description="Maximum context length")
    rate_limit_requests: int = Field(default=10, ge=1, le=100, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=60, ge=10, le=3600, description="Rate limit window in seconds")
    allowed_hosts: List[str] = Field(default=["*"], description="Allowed CORS hosts")
    
    # Cache
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=300, ge=60, le=3600, description="Cache TTL in seconds")
    cache_backend: str = Field(default="memory", description="Cache backend (memory, redis)")
    redis_url: Optional[str] = Field(default=None, description="Redis URL for caching")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    @field_validator("es_url")
    @classmethod
    def validate_es_url(cls, v: str) -> str:
        """Validate Elasticsearch URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("ES_URL must start with http:// or https://")
        return v
    
    @field_validator("es_index")
    @classmethod
    def validate_es_index(cls, v: str) -> str:
        """Validate index name to prevent path traversal."""
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("ES_INDEX contains invalid characters")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
