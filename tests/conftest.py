"""Test configuration and fixtures."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock

from rag_chatbot.core.config import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(
        es_url="https://test.elasticsearch.com",
        es_api_key="test_es_key",
        openai_api_key="test_openai_key",
        es_index="test_index",
        environment="testing",
    )


@pytest_asyncio.fixture
async def mock_es_client():
    """Create mock Elasticsearch client."""
    client = AsyncMock()
    client.search = AsyncMock(return_value={
        "hits": {
            "hits": [
                {"_id": "1", "_score": 1.0, "_source": {"content": "Test content"}},
            ]
        }
    })
    client.cluster = AsyncMock()
    client.cluster.health = AsyncMock(return_value={
        "status": "green",
        "cluster_name": "test-cluster",
    })
    return client


@pytest_asyncio.fixture
async def mock_openai_client():
    """Create mock OpenAI client."""
    client = AsyncMock()
    
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    client.models = AsyncMock()
    client.models.list = AsyncMock(return_value=Mock(data=[]))
    
    return client
