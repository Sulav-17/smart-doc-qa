"""
Embeddings module.

This module supports:
- local embeddings with SentenceTransformers
- optional OpenAI embeddings
- document embeddings
- query embeddings
- attaching embeddings to document chunks
"""

import os
from functools import lru_cache
from typing import Any

from openai import OpenAI


DEFAULT_EMBEDDING_PROVIDER = "local"

DEFAULT_LOCAL_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_OPENAI_EMBEDDING_MODEL = (
    "text-embedding-3-small"
)

VALID_EMBEDDING_PROVIDERS = {
    "local",
    "openai",
}


def get_embedding_provider() -> str:
    """
    Get the configured embedding provider.

    Returns:
        Either "local" or "openai".

    Raises:
        ValueError: If the provider is unsupported.
    """
    provider = os.getenv(
        "EMBEDDING_PROVIDER",
        DEFAULT_EMBEDDING_PROVIDER,
    ).strip().lower()

    if provider not in VALID_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )

    return provider


def get_embedding_model(
    provider: str | None = None,
) -> str:
    """
    Get the embedding model for the selected provider.

    Args:
        provider: Optional provider override.

    Returns:
        Embedding model name.
    """
    selected_provider = (
        provider or get_embedding_provider()
    )

    if selected_provider == "local":
        return os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            DEFAULT_LOCAL_EMBEDDING_MODEL,
        )

    return os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        DEFAULT_OPENAI_EMBEDDING_MODEL,
    )


def get_openai_client(
    api_key: str | None = None,
) -> OpenAI:
    """
    Create an OpenAI client.

    Args:
        api_key: Optional OpenAI API key.

    Returns:
        OpenAI client.
    """
    if api_key:
        return OpenAI(api_key=api_key)

    return OpenAI()


@lru_cache(maxsize=2)
def get_local_embedding_model(
    model_name: str | None = None,
):
    """
    Load and cache a SentenceTransformer model.

    The cache prevents the model from being loaded again
    for every document chunk.

    Args:
        model_name: Optional local model name.

    Returns:
        Loaded SentenceTransformer model.
    """
    from sentence_transformers import SentenceTransformer

    selected_model = (
        model_name
        or get_embedding_model(provider="local")
    )

    return SentenceTransformer(selected_model)


def convert_vector_to_list(
    vector: Any,
) -> list[float]:
    """
    Convert a model vector into a normal Python list.

    Args:
        vector: NumPy array, tensor, or list.

    Returns:
        List of floating-point numbers.
    """
    if hasattr(vector, "tolist"):
        vector = vector.tolist()

    return [float(value) for value in vector]


def create_local_embedding(
    text: str,
    model: Any | None = None,
    model_name: str | None = None,
    task: str = "document",
) -> list[float]:
    """
    Create a local embedding.

    Args:
        text: Text to embed.
        model: Optional loaded SentenceTransformer model.
        model_name: Optional model name.
        task: Either "document" or "query".

    Returns:
        Local embedding vector.

    Raises:
        ValueError: If the task is unsupported.
    """
    if task not in {"document", "query"}:
        raise ValueError(
            "Embedding task must be 'document' or 'query'."
        )

    embedding_model = (
        model
        or get_local_embedding_model(model_name)
    )

    if (
        task == "query"
        and hasattr(embedding_model, "encode_query")
    ):
        vector = embedding_model.encode_query(
            text,
            normalize_embeddings=True,
        )

    elif (
        task == "document"
        and hasattr(embedding_model, "encode_document")
    ):
        vector = embedding_model.encode_document(
            text,
            normalize_embeddings=True,
        )

    else:
        vector = embedding_model.encode(
            text,
            normalize_embeddings=True,
        )

    return convert_vector_to_list(vector)


def create_openai_embedding(
    text: str,
    client: Any | None = None,
    model: str | None = None,
) -> list[float]:
    """
    Create an embedding using OpenAI.

    Args:
        text: Text to embed.
        client: Optional OpenAI-compatible client.
        model: Optional OpenAI embedding model.

    Returns:
        OpenAI embedding vector.
    """
    embedding_client = (
        client or get_openai_client()
    )

    embedding_model = (
        model
        or get_embedding_model(provider="openai")
    )

    response = embedding_client.embeddings.create(
        model=embedding_model,
        input=text,
    )

    return convert_vector_to_list(
        response.data[0].embedding
    )


def create_embedding(
    text: str,
    client: Any | None = None,
    model: str | None = None,
    provider: str | None = None,
    local_model: Any | None = None,
    task: str = "document",
) -> list[float]:
    """
    Create an embedding using the selected provider.

    Args:
        text: Text to embed.
        client: Optional OpenAI-compatible client.
        model: Optional embedding model name.
        provider: Optional provider override.
        local_model: Optional loaded local model.
        task: Either "document" or "query".

    Returns:
        Embedding vector.

    Raises:
        ValueError: If text is empty.
    """
    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError(
            "Cannot create an embedding for empty text."
        )

    # Existing tests may pass a fake OpenAI client.
    # In that situation, automatically use OpenAI mode.
    selected_provider = provider

    if selected_provider is None:
        selected_provider = (
            "openai"
            if client is not None
            else get_embedding_provider()
        )

    selected_model = (
        model
        or get_embedding_model(selected_provider)
    )

    if selected_provider == "local":
        return create_local_embedding(
            text=cleaned_text,
            model=local_model,
            model_name=selected_model,
            task=task,
        )

    if selected_provider == "openai":
        return create_openai_embedding(
            text=cleaned_text,
            client=client,
            model=selected_model,
        )

    raise ValueError(
        f"Unsupported embedding provider: "
        f"{selected_provider}"
    )


def embed_chunks(
    chunks: list[dict[str, Any]],
    client: Any | None = None,
    model: str | None = None,
    provider: str | None = None,
    local_model: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Create embeddings for document chunks.

    Args:
        chunks: Document chunks.
        client: Optional OpenAI-compatible client.
        model: Optional embedding model.
        provider: Optional provider override.
        local_model: Optional local model instance.

    Returns:
        Chunks containing embeddings.
    """
    selected_provider = provider

    if selected_provider is None:
        selected_provider = (
            "openai"
            if client is not None
            else get_embedding_provider()
        )

    selected_model = (
        model
        or get_embedding_model(selected_provider)
    )

    shared_local_model = local_model

    if (
        selected_provider == "local"
        and shared_local_model is None
    ):
        shared_local_model = get_local_embedding_model(
            selected_model
        )

    embedded_chunks = []

    for chunk in chunks:
        embedding = create_embedding(
            text=chunk["text"],
            client=client,
            model=selected_model,
            provider=selected_provider,
            local_model=shared_local_model,
            task="document",
        )

        embedded_chunks.append(
            {
                **chunk,
                "embedding": embedding,
                "embedding_model": selected_model,
                "embedding_provider": selected_provider,
            }
        )

    return embedded_chunks