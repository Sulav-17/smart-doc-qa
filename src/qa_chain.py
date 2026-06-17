"""
Question-answering module.

This module handles:
- building a grounded RAG prompt
- sending retrieved context to the LLM
- generating an answer using only document sources
"""

import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.citations import build_context_block, format_source_passages


DEFAULT_ANSWER_MODEL = "gpt-5.5"
DEFAULT_PROMPT_PATH = "prompts/rag_answer.txt"


def get_answer_model() -> str:
    """
    Get the answer generation model from environment variables.

    Returns:
        Model name.
    """
    return os.getenv("OPENAI_ANSWER_MODEL", DEFAULT_ANSWER_MODEL)


def get_openai_client(api_key: str | None = None) -> OpenAI:
    """
    Create an OpenAI client.

    Args:
        api_key: Optional API key.

    Returns:
        OpenAI client.
    """
    if api_key:
        return OpenAI(api_key=api_key)

    return OpenAI()


def load_prompt_template(prompt_path: str = DEFAULT_PROMPT_PATH) -> str:
    """
    Load the RAG answer prompt template from disk.

    Args:
        prompt_path: Path to prompt template.

    Returns:
        Prompt template string.
    """
    path = Path(prompt_path)

    return path.read_text(encoding="utf-8")


def build_rag_prompt(
    question: str,
    search_results: list[dict[str, Any]],
    prompt_template: str,
) -> str:
    """
    Build the final RAG prompt using retrieved chunks.

    Args:
        question: User question.
        search_results: Retrieved chunks from vector search.
        prompt_template: Prompt template with {context} and {question}.

    Returns:
        Completed prompt.
    """
    context = build_context_block(search_results)

    return prompt_template.format(
        context=context,
        question=question,
    )


def generate_answer(
    question: str,
    search_results: list[dict[str, Any]],
    client: Any | None = None,
    model: str | None = None,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    """
    Generate a grounded answer from retrieved chunks.

    Args:
        question: User question.
        search_results: Retrieved chunks from vector search.
        client: Optional OpenAI-compatible client.
        model: Optional answer model.
        prompt_template: Optional prompt template.

    Returns:
        Dictionary with answer and sources.
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    if not search_results:
        return {
            "answer": "I do not know based on the provided document.",
            "sources": [],
        }

    answer_client = client or get_openai_client()
    answer_model = model or get_answer_model()
    template = prompt_template or load_prompt_template()

    final_prompt = build_rag_prompt(
        question=cleaned_question,
        search_results=search_results,
        prompt_template=template,
    )

    response = answer_client.responses.create(
        model=answer_model,
        input=final_prompt,
    )

    return {
        "answer": response.output_text,
        "sources": format_source_passages(search_results),
    }