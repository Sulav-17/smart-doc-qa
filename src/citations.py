"""
Citations module.

This module handles:
- formatting retrieved chunks into readable source passages
- creating source labels for page and chunk references
- preparing evidence for grounded RAG answers
"""

from typing import Any


def build_source_label(result: dict[str, Any]) -> str:
    """
    Build a readable source label for a retrieved chunk.

    Args:
        result: Retrieved chunk result.

    Returns:
        Source label such as "Page 2, Chunk 5".
    """
    return f"Page {result['page_number']}, Chunk {result['chunk_id']}"


def format_source_passage(result: dict[str, Any], max_length: int = 700) -> dict[str, Any]:
    """
    Format one retrieved chunk as a source passage.

    Args:
        result: Retrieved chunk result.
        max_length: Maximum number of characters to show.

    Returns:
        Formatted source passage dictionary.
    """
    text = result["text"].strip()

    if len(text) > max_length:
        text = text[:max_length].rstrip() + "..."

    return {
        "source": build_source_label(result),
        "page_number": result["page_number"],
        "chunk_id": result["chunk_id"],
        "distance": result.get("distance"),
        "text": text,
    }


def format_source_passages(
    search_results: list[dict[str, Any]],
    max_length: int = 700,
) -> list[dict[str, Any]]:
    """
    Format multiple retrieved chunks as source passages.

    Args:
        search_results: Retrieved chunks from vector search.
        max_length: Maximum passage length.

    Returns:
        List of formatted source passage dictionaries.
    """
    return [
        format_source_passage(result, max_length=max_length)
        for result in search_results
    ]


def build_context_block(search_results: list[dict[str, Any]]) -> str:
    """
    Build a context block for the LLM from retrieved chunks.

    Args:
        search_results: Retrieved chunks from vector search.

    Returns:
        Context string containing source labels and passages.
    """
    context_sections = []

    for result in search_results:
        source_label = build_source_label(result)

        context_sections.append(
            f"[{source_label}]\n{result['text']}"
        )

    return "\n\n".join(context_sections)