"""
RAG Chatbot using OpenAI and Elasticsearch.

This module provides a simple CLI-based chatbot that retrieves context
from Elasticsearch and generates answers using OpenAI's GPT model.

Usage:
    python streamlit_app.py
"""
import logging
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, ConnectionError as ESConnectionError
from openai import OpenAI, RateLimitError, APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""
    pass


class ChatbotError(Exception):
    """Base exception for chatbot errors."""
    pass


class SearchError(ChatbotError):
    """Raised when search operation fails."""
    pass


class LLMError(ChatbotError):
    """Raised when LLM operation fails."""
    pass


# Configuration with validation
def get_config() -> Dict[str, str]:
    """Load and validate configuration from environment variables."""
    required_vars = ["ES_API_KEY", "OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Please copy .env.example to .env and fill in your credentials."
        )
    
    return {
        "ES_URL": os.environ.get("ES_URL", "https://b85c-176-76-226-102.ngrok-free.app"),
        "ES_API_KEY": os.environ["ES_API_KEY"],
        "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        "ES_INDEX": os.environ.get("ES_INDEX", "ttintegration"),
        "MODEL_NAME": os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
        "RESULTS_SIZE": int(os.environ.get("RESULTS_SIZE", "3")),
    }


# Initialize configuration
config = get_config()

# Initialize clients
es_client = Elasticsearch(config["ES_URL"], api_key=config["ES_API_KEY"])
openai_client = OpenAI(api_key=config["OPENAI_API_KEY"])

index_source_fields = {config["ES_INDEX"]: ["content"]}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ESConnectionError, ConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def get_elasticsearch_results(query: str) -> List[Dict[str, Any]]:
    """
    Perform semantic search on Elasticsearch index.

    Args:
        query: The search query string.

    Returns:
        List of hit documents from Elasticsearch.

    Raises:
        SearchError: If the Elasticsearch query fails after retries.
    """
    logger.info(f"Searching Elasticsearch for: {query[:50]}...")
    
    es_query = {
        "retriever": {
            "standard": {
                "query": {
                    "semantic": {"field": "content", "query": query}
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
        "size": config["RESULTS_SIZE"],
    }

    try:
        result = es_client.search(index=config["ES_INDEX"], body=es_query)
        hits = result["hits"]["hits"]
        logger.info(f"Found {len(hits)} results")
        return hits
    except Exception as e:
        logger.error(f"Elasticsearch search failed: {e}")
        raise SearchError(f"Failed to search Elasticsearch: {e}") from e


def create_openai_prompt(results: List[Dict[str, Any]]) -> str:
    """
    Create a prompt for OpenAI based on search results.

    Args:
        results: List of Elasticsearch hit documents.

    Returns:
        Formatted prompt string for OpenAI.
    """
    context_parts = []

    for hit in results:
        if "highlight" in hit:
            highlighted_texts = []
            for values in hit["highlight"].values():
                highlighted_texts.extend(values)
            context_parts.append("\n --- \n".join(highlighted_texts))
        else:
            source_field = index_source_fields.get(hit["_index"], ["content"])[0]
            hit_context = hit["_source"].get(source_field, "")
            context_parts.append(hit_context)

    context = "\n".join(context_parts)

    prompt = f"""
Instructions:

- You are an assistant for question-answering tasks.
- Answer questions truthfully and factually using only the context presented.
- If you don't know the answer, just say that you don't know, don't make up an answer.
- You must always cite the document where the answer was extracted using inline academic citation style [], using the position.
- Use markdown format for code examples.
- You are correct, factual, precise, and reliable.

Context:
{context}

"""
    return prompt


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def generate_openai_completion(user_prompt: str, question: str) -> str:
    """
    Generate completion from OpenAI based on prompt and question.

    Args:
        user_prompt: The system prompt with context.
        question: The user's question.

    Returns:
        The generated response text.

    Raises:
        LLMError: If the OpenAI API call fails after retries.
    """
    logger.info("Generating completion with OpenAI...")
    
    try:
        response = openai_client.chat.completions.create(
            model=config["MODEL_NAME"],
            messages=[
                {"role": "system", "content": user_prompt},
                {"role": "user", "content": question},
            ],
        )
        content = response.choices[0].message.content
        logger.info("Successfully generated completion")
        return content
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise LLMError(f"Failed to generate completion: {e}") from e


def main():
    """Main entry point for the CLI chatbot."""
    try:
        question = input("Enter your question: ").strip()

        if not question:
            logger.error("Question cannot be empty")
            print("Error: Question cannot be empty")
            return 1

        logger.info(f"Processing question: {question[:50]}...")

        elasticsearch_results = get_elasticsearch_results(question)

        if not elasticsearch_results:
            logger.warning("No results found for query")
            print("No relevant documents found.")
            return 0

        context_prompt = create_openai_prompt(elasticsearch_results)
        openai_completion = generate_openai_completion(context_prompt, question)

        print("\n" + "=" * 50)
        print("Answer:")
        print("=" * 50)
        print(openai_completion)
        
        return 0

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration Error: {e}")
        return 1
    except SearchError as e:
        logger.error(f"Search error: {e}")
        print(f"Search Error: Failed to retrieve documents. Please try again later.")
        return 1
    except LLMError as e:
        logger.error(f"LLM error: {e}")
        print(f"Generation Error: Failed to generate response. Please try again later.")
        return 1
    except KeyboardInterrupt:
        logger.info("User interrupted")
        print("\nOperation cancelled.")
        return 0
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"Unexpected Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
