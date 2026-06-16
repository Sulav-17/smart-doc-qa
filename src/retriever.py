"""
Retriever module.

This module handles:
- creating a local ChromaDB client
- storing embedded chunks in a vector database
- previewing stored chunks

This is the fourth step in the RAG pipeline.
"""

from typing import Any

import chromadb


DEFAULT_CHROMA_PATH = "chroma_db"
DEFAULT_COLLECTION_NAME = "document_chunks"


def get_chroma_client(persist_directory: str = DEFAULT_CHROMA_PATH):
    """
    Create a persistent ChromaDB client.

    Args:
        persist_directory: Local folder where ChromaDB data will be stored.

    Returns:
        A ChromaDB persistent client.
    """
    return chromadb.PersistentClient(path=persist_directory)


def get_or_create_collection(
    client: Any,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """
    Get an existing ChromaDB collection or create it if it does not exist.

    Args:
        client: ChromaDB client.
        collection_name: Name of the collection.

    Returns:
        ChromaDB collection.
    """
    return client.get_or_create_collection(name=collection_name)


def reset_collection(
    client: Any,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """
    Delete and recreate a ChromaDB collection.

    This is useful during development when we want a clean database.

    Args:
        client: ChromaDB client.
        collection_name: Name of the collection.

    Returns:
        A fresh ChromaDB collection.
    """
    existing_collections = [collection.name for collection in client.list_collections()]

    if collection_name in existing_collections:
        client.delete_collection(name=collection_name)

    return client.create_collection(name=collection_name)


def build_record_id(chunk: dict[str, Any]) -> str:
    """
    Build a unique ChromaDB record ID for a chunk.

    Args:
        chunk: Embedded chunk dictionary.

    Returns:
        Unique string ID.
    """
    return f"page-{chunk['page_number']}-chunk-{chunk['chunk_id']}"


def prepare_chroma_records(
    embedded_chunks: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[list[float]], list[dict[str, Any]]]:
    """
    Convert embedded chunks into ChromaDB add() inputs.

    Args:
        embedded_chunks: Chunks that already contain embeddings.

    Returns:
        Tuple containing ids, documents, embeddings, and metadatas.
    """
    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in embedded_chunks:
        ids.append(build_record_id(chunk))
        documents.append(chunk["text"])
        embeddings.append(chunk["embedding"])
        metadatas.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "character_count": chunk["character_count"],
                "embedding_model": chunk.get("embedding_model", "unknown"),
            }
        )

    return ids, documents, embeddings, metadatas


def store_embedded_chunks(
    embedded_chunks: list[dict[str, Any]],
    persist_directory: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset: bool = True,
) -> int:
    """
    Store embedded chunks in ChromaDB.

    Args:
        embedded_chunks: Chunks that already contain embeddings.
        persist_directory: Local folder where ChromaDB data will be stored.
        collection_name: Name of the ChromaDB collection.
        reset: Whether to clear the collection before storing.

    Returns:
        Number of chunks stored.
    """
    if not embedded_chunks:
        return 0

    client = get_chroma_client(persist_directory=persist_directory)

    if reset:
        collection = reset_collection(
            client=client,
            collection_name=collection_name,
        )
    else:
        collection = get_or_create_collection(
            client=client,
            collection_name=collection_name,
        )

    ids, documents, embeddings, metadatas = prepare_chroma_records(embedded_chunks)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(ids)


def get_collection_count(
    persist_directory: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> int:
    """
    Count records in a ChromaDB collection.

    Args:
        persist_directory: Local folder where ChromaDB data is stored.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Number of stored records.
    """
    client = get_chroma_client(persist_directory=persist_directory)
    collection = get_or_create_collection(
        client=client,
        collection_name=collection_name,
    )

    return collection.count()


def preview_collection(
    persist_directory: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    limit: int = 3,
) -> dict[str, Any]:
    """
    Preview stored ChromaDB records.

    Args:
        persist_directory: Local folder where ChromaDB data is stored.
        collection_name: Name of the ChromaDB collection.
        limit: Number of records to preview.

    Returns:
        ChromaDB get() result.
    """
    client = get_chroma_client(persist_directory=persist_directory)
    collection = get_or_create_collection(
        client=client,
        collection_name=collection_name,
    )

    return collection.get(
        limit=limit,
        include=["documents", "metadatas"],
    )