"""
Question-answering module.

This module supports:
- local answer generation with Ollama
- optional OpenAI answer generation
- grounded RAG prompts
- answers with source passages
"""

import os
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from src.citations import (
    build_context_block,
    format_source_passages,
)


DEFAULT_AI_PROVIDER = "local"
DEFAULT_LOCAL_ANSWER_MODEL = "qwen3:4b"
DEFAULT_OPENAI_ANSWER_MODEL = "gpt-5.5"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_PROMPT_PATH = "prompts/rag_answer.txt"

VALID_AI_PROVIDERS = {
    "local",
    "openai",
}


def get_ai_provider() -> str:
    """
    Get the configured answer-generation provider.

    Returns:
        Either "local" or "openai".

    Raises:
        ValueError: If the provider is unsupported.
    """
    provider = os.getenv(
        "AI_PROVIDER",
        DEFAULT_AI_PROVIDER,
    ).strip().lower()

    if provider not in VALID_AI_PROVIDERS:
        raise ValueError(
            f"Unsupported AI provider: {provider}"
        )

    return provider


def get_answer_model(
    provider: str | None = None,
) -> str:
    """
    Get the answer model for the selected provider.

    Args:
        provider: Optional provider override.

    Returns:
        Answer model name.
    """
    selected_provider = (
        provider or get_ai_provider()
    )

    if selected_provider == "local":
        return os.getenv(
            "OLLAMA_MODEL",
            DEFAULT_LOCAL_ANSWER_MODEL,
        )

    return os.getenv(
        "OPENAI_ANSWER_MODEL",
        DEFAULT_OPENAI_ANSWER_MODEL,
    )


def get_ollama_base_url() -> str:
    """
    Get the Ollama server address.

    Returns:
        Ollama base URL.
    """
    return os.getenv(
        "OLLAMA_BASE_URL",
        DEFAULT_OLLAMA_BASE_URL,
    ).rstrip("/")


def get_openai_client(
    api_key: str | None = None,
) -> OpenAI:
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


def load_prompt_template(
    prompt_path: str = DEFAULT_PROMPT_PATH,
) -> str:
    """
    Load the RAG prompt template.

    Args:
        prompt_path: Prompt template path.

    Returns:
        Prompt template text.
    """
    path = Path(prompt_path)

    return path.read_text(
        encoding="utf-8"
    )


def build_rag_prompt(
    question: str,
    search_results: list[dict[str, Any]],
    prompt_template: str,
) -> str:
    """
    Build a grounded RAG prompt.

    Args:
        question: User question.
        search_results: Retrieved chunks.
        prompt_template: Prompt template.

    Returns:
        Completed prompt.
    """
    context = build_context_block(
        search_results
    )

    return prompt_template.format(
        context=context,
        question=question,
    )


def generate_openai_text(
    prompt: str,
    client: Any | None = None,
    model: str | None = None,
) -> str:
    """
    Generate answer text using OpenAI.

    Args:
        prompt: Completed RAG prompt.
        client: Optional OpenAI-compatible client.
        model: Optional model name.

    Returns:
        Generated answer.
    """
    answer_client = (
        client or get_openai_client()
    )

    answer_model = (
        model
        or get_answer_model(provider="openai")
    )

    response = answer_client.responses.create(
        model=answer_model,
        input=prompt,
    )

    answer = response.output_text.strip()

    if not answer:
        raise ValueError(
            "OpenAI returned an empty answer."
        )

    return answer


def generate_ollama_text(
    prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    http_client: Any | None = None,
) -> str:
    """
    Generate answer text using local Ollama.

    Args:
        prompt: Completed RAG prompt.
        model: Optional Ollama model.
        base_url: Optional Ollama server address.
        http_client: Optional requests-compatible client.

    Returns:
        Generated answer.
    """
    answer_model = (
        model
        or get_answer_model(provider="local")
    )

    ollama_url = (
        base_url or get_ollama_base_url()
    ).rstrip("/")

    request_client = (
        http_client or requests
    )

    response = request_client.post(
        f"{ollama_url}/api/generate",
        json={
            "model": answer_model,
            "prompt": prompt,
            "stream": False,
            "think": False,
        },
        timeout=180,
    )

    response.raise_for_status()

    response_data = response.json()

    answer = str(
        response_data.get("response", "")
    ).strip()

    if not answer:
        raise ValueError(
            "Ollama returned an empty answer."
        )

    return answer


def generate_answer(
    question: str,
    search_results: list[dict[str, Any]],
    client: Any | None = None,
    model: str | None = None,
    prompt_template: str | None = None,
    provider: str | None = None,
    http_client: Any | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """
    Generate a grounded answer.

    Args:
        question: User question.
        search_results: Retrieved document chunks.
        client: Optional OpenAI-compatible client.
        model: Optional answer model.
        prompt_template: Optional prompt template.
        provider: Optional provider override.
        http_client: Optional requests-compatible client.
        base_url: Optional Ollama base URL.

    Returns:
        Answer and formatted sources.
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not search_results:
        return {
            "answer": (
                "I do not know based on "
                "the provided document."
            ),
            "sources": [],
        }

    # Preserve compatibility with fake OpenAI tests.
    selected_provider = provider

    if selected_provider is None:
        selected_provider = (
            "openai"
            if client is not None
            else get_ai_provider()
        )

    answer_model = (
        model
        or get_answer_model(selected_provider)
    )

    template = (
        prompt_template
        or load_prompt_template()
    )

    final_prompt = build_rag_prompt(
        question=cleaned_question,
        search_results=search_results,
        prompt_template=template,
    )

    if selected_provider == "local":
        answer = generate_ollama_text(
            prompt=final_prompt,
            model=answer_model,
            base_url=base_url,
            http_client=http_client,
        )

    elif selected_provider == "openai":
        answer = generate_openai_text(
            prompt=final_prompt,
            client=client,
            model=answer_model,
        )

    else:
        raise ValueError(
            f"Unsupported AI provider: "
            f"{selected_provider}"
        )

    return {
        "answer": answer,
        "sources": format_source_passages(
            search_results
        ),
    }