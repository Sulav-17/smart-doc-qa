"""
Embeddings module.

This module handles:
- creating vector embeddings for text
- attaching embeddings to document chunks
- preparing chunks for future vector database storage

Embeddings are the third step in the RAG pipeline.
"""

import os
from typing import Any

from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding_model() -> str:
    """
    Get the embedding model name from environment variables.

    Returns:
        The embedding model name.
    """
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_openai_client(api_key: str | None = None) -> OpenAI:
    """
    Create an OpenAI client.

    Args:
        api_key: Optional API key. If not provided, OpenAI will use OPENAI_API_KEY
                 from the environment.

    Returns:
        OpenAI client.
    """
    if api_key:
        return OpenAI(api_key=api_key)

    return OpenAI()


def create_embedding(
    text: str,
    client: Any | None = None,
    model: str | None = None,
) -> list[float]:
    """
    Create an embedding for a single piece of text.

    Args:
        text: Text to embed.
        client: Optional OpenAI-compatible client.
        model: Optional embedding model name.

    Returns:
        A list of floats representing the text embedding.

    Raises:
        ValueError: If the text is empty.
    """
    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("Cannot create an embedding for empty text.")

    embedding_client = client or get_openai_client()
    embedding_model = model or get_embedding_model()

    response = embedding_client.embeddings.create(
        model=embedding_model,
        input=cleaned_text,
    )

    return response.data[0].embedding


def embed_chunks(
    chunks: list[dict[str, Any]],
    client: Any | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Create embeddings for a list of document chunks.

    Args:
        chunks: List of chunk dictionaries.
        client: Optional OpenAI-compatible client.
        model: Optional embedding model name.

    Returns:
        The original chunks with an added embedding field.
    """
    embedded_chunks = []

    for chunk in chunks:
        embedding = create_embedding(
            text=chunk["text"],
            client=client,
            model=model,
        )

        embedded_chunk = {
            **chunk,
            "embedding": embedding,
            "embedding_model": model or get_embedding_model(),
        }

        embedded_chunks.append(embedded_chunk)

    return embedded_chunks