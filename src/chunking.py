"""
Text chunking module.

This module handles:
- splitting extracted PDF text into smaller chunks
- preserving page numbers for citations
- preparing document chunks for future embeddings

Chunking is the second step in the RAG pipeline.
"""

from typing import Any


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """
    Validate chunking settings before splitting text.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Raises:
        ValueError: If the chunking settings are invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def normalize_text(text: str) -> str:
    """
    Normalize text before chunking.

    Args:
        text: Raw page text.

    Returns:
        Text with extra whitespace collapsed.
    """
    return " ".join(text.split())


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    Args:
        text: Text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters repeated between chunks.

    Returns:
        A list of text chunks.
    """
    validate_chunk_settings(chunk_size, chunk_overlap)

    normalized_text = normalize_text(text)

    if not normalized_text:
        return []

    chunks = []
    start_index = 0

    while start_index < len(normalized_text):
        end_index = start_index + chunk_size
        chunk = normalized_text[start_index:end_index].strip()

        if chunk:
            chunks.append(chunk)

        start_index += chunk_size - chunk_overlap

    return chunks


def chunk_document(
    pages: list[dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """
    Chunk all extracted PDF pages.

    Args:
        pages: List of page dictionaries from PDF extraction.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters repeated between chunks.

    Returns:
        A list of chunk dictionaries.
    """
    validate_chunk_settings(chunk_size, chunk_overlap)

    document_chunks = []
    chunk_counter = 1

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = split_text_into_chunks(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_text in page_chunks:
            document_chunks.append(
                {
                    "chunk_id": chunk_counter,
                    "page_number": page_number,
                    "text": chunk_text,
                    "character_count": len(chunk_text),
                }
            )

            chunk_counter += 1

    return document_chunks