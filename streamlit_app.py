"""
RAG Chatbot using OpenAI and Elasticsearch.

This module provides a simple CLI-based chatbot that retrieves context
from Elasticsearch and generates answers using OpenAI's GPT model.

Usage:
    python streamlit_app.py
"""
import os
from typing import Any, Dict, List
from elasticsearch import Elasticsearch
from openai import OpenAI

# Configuration
ES_URL = os.environ.get("ES_URL", "https://b85c-176-76-226-102.ngrok-free.app")
ES_API_KEY = os.environ["ES_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ES_INDEX = "ttintegration"
MODEL_NAME = "gpt-3.5-turbo"
RESULTS_SIZE = 3

# Initialize clients
es_client = Elasticsearch(ES_URL, api_key=ES_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

index_source_fields = {ES_INDEX: ["content"]}


def get_elasticsearch_results(query: str) -> List[Dict[str, Any]]:
    """
    Perform semantic search on Elasticsearch index.

    Args:
        query: The search query string.

    Returns:
        List of hit documents from Elasticsearch.

    Raises:
        Exception: If the Elasticsearch query fails.
    """
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
        "size": RESULTS_SIZE,
    }

    try:
        result = es_client.search(index=ES_INDEX, body=es_query)
        return result["hits"]["hits"]
    except Exception as e:
        print(f"Error querying Elasticsearch: {e}")
        raise


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


def generate_openai_completion(user_prompt: str, question: str) -> str:
    """
    Generate completion from OpenAI based on prompt and question.

    Args:
        user_prompt: The system prompt with context.
        question: The user's question.

    Returns:
        The generated response text.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": user_prompt},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise


def main():
    """Main entry point for the CLI chatbot."""
    question = input("Enter your question: ").strip()

    if not question:
        print("Error: Question cannot be empty")
        return

    print("Searching...")

    try:
        elasticsearch_results = get_elasticsearch_results(question)

        if not elasticsearch_results:
            print("No relevant documents found.")
            return

        context_prompt = create_openai_prompt(elasticsearch_results)
        openai_completion = generate_openai_completion(context_prompt, question)

        print("\n" + "=" * 50)
        print("Answer:")
        print("=" * 50)
        print(openai_completion)

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()
