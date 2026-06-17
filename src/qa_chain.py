"""
Grounded question-answer generation.

Supports:
- Local Ollama generation
- Optional OpenAI generation
- Retrieved-context formatting
- Source formatting
- Prompt templates
- Removal of reasoning-model thinking text
- Backward compatibility with earlier tests
"""

import os
import re
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "rag_answer.txt"
)

DEFAULT_AI_PROVIDER = "local"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"

ABSTENTION_TEXT = (
    "I do not know based on the provided document."
)


def get_ai_provider() -> str:
    """
    Return the configured answer provider.
    """
    return os.getenv(
        "AI_PROVIDER",
        DEFAULT_AI_PROVIDER,
    ).strip().lower()


def get_answer_model(
    provider: str | None = None,
) -> str:
    """
    Return the configured answer-generation model.
    """
    selected_provider = (
        provider or get_ai_provider()
    ).strip().lower()

    if selected_provider == "openai":
        return os.getenv(
            "OPENAI_ANSWER_MODEL",
            "gpt-4.1-mini",
        )

    return os.getenv(
        "OLLAMA_MODEL",
        DEFAULT_OLLAMA_MODEL,
    )


def get_ollama_base_url() -> str:
    """
    Return the configured Ollama server URL.
    """
    return os.getenv(
        "OLLAMA_BASE_URL",
        DEFAULT_OLLAMA_BASE_URL,
    ).rstrip("/")


def get_openai_client():
    """
    Create an OpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required when "
            "AI_PROVIDER=openai."
        )

    try:
        from openai import OpenAI

    except ImportError as error:
        raise ImportError(
            "Install the OpenAI package before "
            "using the OpenAI provider."
        ) from error

    return OpenAI(
        api_key=api_key
    )


def load_prompt_template(
    prompt_path: str | Path | None = None,
) -> str:
    """
    Load the RAG prompt template from a file.
    """
    path = Path(
        prompt_path or DEFAULT_PROMPT_PATH
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    ).strip()


def get_result_text(
    result: dict[str, Any],
) -> str:
    """
    Extract text from a retrieved result.
    """
    text = (
        result.get("text")
        or result.get("document")
        or result.get("content")
        or ""
    )

    return str(text).strip()


def get_result_metadata(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract metadata from a retrieved result.
    """
    metadata = result.get("metadata")

    if isinstance(metadata, dict):
        return metadata

    return {}


def get_page_number(
    result: dict[str, Any],
) -> int | None:
    """
    Extract a page number from a retrieved result.
    """
    metadata = get_result_metadata(
        result
    )

    page_number = (
        result.get("page_number")
        or metadata.get("page_number")
        or metadata.get("page")
    )

    if page_number is None:
        return None

    try:
        return int(page_number)

    except (TypeError, ValueError):
        return None


def get_chunk_id(
    result: dict[str, Any],
) -> Any:
    """
    Extract a chunk identifier from a result.
    """
    metadata = get_result_metadata(
        result
    )

    return (
        result.get("chunk_id")
        or metadata.get("chunk_id")
        or metadata.get("chunk_index")
        or result.get("id")
    )


def build_context(
    search_results: list[dict[str, Any]],
) -> str:
    """
    Format retrieved chunks as labeled context.
    """
    context_sections = []

    for index, result in enumerate(
        search_results,
        start=1,
    ):
        text = get_result_text(
            result
        )

        if not text:
            continue

        page_number = get_page_number(
            result
        )

        chunk_id = get_chunk_id(
            result
        )

        label_parts = []

        if page_number is not None:
            label_parts.append(
                f"Page {page_number}"
            )

        if chunk_id is not None:
            label_parts.append(
                f"Chunk {chunk_id}"
            )

        if label_parts:
            source_label = ", ".join(
                label_parts
            )

        else:
            source_label = (
                f"Source {index}"
            )

        context_sections.append(
            f"[{source_label}]\n{text}"
        )

    if not context_sections:
        return (
            "No relevant document context "
            "was retrieved."
        )

    return "\n\n".join(
        context_sections
    )


def build_rag_prompt(
    question: str,
    search_results: list[dict[str, Any]],
    prompt_template: str | None = None,
    prompt_path: str | Path | None = None,
) -> str:
    """
    Build the complete grounded RAG prompt.

    Supports:
    - prompt_template passed directly
    - prompt loaded from a file
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "Question cannot be empty."
        )

    selected_template = (
        prompt_template
        if prompt_template is not None
        else load_prompt_template(
            prompt_path
        )
    )

    context = build_context(
        search_results
    )

    return selected_template.format(
        context=context,
        question=cleaned_question,
    )


def build_prompt(
    question: str,
    search_results: list[dict[str, Any]],
    prompt_template: str | None = None,
    prompt_path: str | Path | None = None,
) -> str:
    """
    Alias for build_rag_prompt.
    """
    return build_rag_prompt(
        question=question,
        search_results=search_results,
        prompt_template=prompt_template,
        prompt_path=prompt_path,
    )


def remove_thinking_text(
    answer: str,
) -> str:
    """
    Remove hidden reasoning generated by reasoning models.
    """
    if not answer:
        return ""

    cleaned_answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    if re.search(
        r"</think>",
        cleaned_answer,
        flags=re.IGNORECASE,
    ):
        cleaned_answer = re.split(
            r"</think>",
            cleaned_answer,
            flags=re.IGNORECASE,
        )[-1]

    cleaned_answer = re.sub(
        r"<think>",
        "",
        cleaned_answer,
        flags=re.IGNORECASE,
    )

    return cleaned_answer.strip()


def format_sources(
    search_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert retrieved chunks into source dictionaries.
    """
    sources = []

    for index, result in enumerate(
        search_results,
        start=1,
    ):
        text = get_result_text(
            result
        )

        if not text:
            continue

        page_number = get_page_number(
            result
        )

        chunk_id = get_chunk_id(
            result
        )

        label_parts = []

        if page_number is not None:
            label_parts.append(
                f"Page {page_number}"
            )

        if chunk_id is not None:
            label_parts.append(
                f"Chunk {chunk_id}"
            )

        if label_parts:
            source_label = ", ".join(
                label_parts
            )

        else:
            source_label = (
                f"Source {index}"
            )

        source = {
            "source": source_label,
            "text": text,
            "page_number": page_number,
            "chunk_id": chunk_id,
        }

        distance = result.get(
            "distance"
        )

        if distance is not None:
            try:
                source["distance"] = float(
                    distance
                )

            except (TypeError, ValueError):
                source["distance"] = (
                    distance
                )

        sources.append(
            source
        )

    return sources


def generate_local_answer(
    prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    http_client=None,
) -> str:
    """
    Generate an answer using Ollama.

    http_client can be replaced with a fake client
    during unit testing.
    """
    selected_model = (
        model
        or get_answer_model("local")
    )

    selected_base_url = (
        base_url
        or get_ollama_base_url()
    ).rstrip("/")

    selected_http_client = (
        http_client
        or requests
    )

    response = selected_http_client.post(
        f"{selected_base_url}/api/generate",
        json={
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
            },
        },
        timeout=180,
    )

    if hasattr(
        response,
        "raise_for_status",
    ):
        response.raise_for_status()

    response_data = response.json()

    raw_answer = str(
        response_data.get(
            "response",
            "",
        )
    )

    cleaned_answer = remove_thinking_text(
        raw_answer
    )

    if not cleaned_answer:
        raise ValueError(
            "Ollama returned an empty answer."
        )

    return cleaned_answer


def generate_openai_answer(
    prompt: str,
    client=None,
    model: str | None = None,
) -> str:
    """
    Generate an answer using OpenAI.
    """
    openai_client = (
        client
        or get_openai_client()
    )

    selected_model = (
        model
        or get_answer_model("openai")
    )

    response = (
        openai_client.responses.create(
            model=selected_model,
            input=prompt,
        )
    )

    raw_answer = getattr(
        response,
        "output_text",
        "",
    )

    if not raw_answer:
        raw_answer = str(
            response
        )

    cleaned_answer = remove_thinking_text(
        raw_answer
    )

    if not cleaned_answer:
        raise ValueError(
            "OpenAI returned an empty answer."
        )

    return cleaned_answer


def generate_answer(
    question: str,
    search_results: list[dict[str, Any]],
    client=None,
    provider: str | None = None,
    model: str | None = None,
    prompt_template: str | None = None,
    prompt_path: str | Path | None = None,
    base_url: str | None = None,
    http_client=None,
) -> dict[str, Any]:
    """
    Generate a grounded answer and return its sources.

    Backward-compatible arguments:
    - prompt_template
    - base_url
    - http_client
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "Question cannot be empty."
        )

    if search_results is None:
        search_results = []

    if client is not None:
        selected_provider = "openai"

    else:
        selected_provider = (
            provider
            or get_ai_provider()
        ).strip().lower()

    if selected_provider not in {
        "local",
        "openai",
    }:
        raise ValueError(
            "AI_PROVIDER must be either "
            "'local' or 'openai'."
        )

    # When retrieval finds nothing, do not ask the
    # language model to guess. Abstain immediately.
    if not search_results:
        return {
            "answer": ABSTENTION_TEXT,
            "sources": [],
            "provider": selected_provider,
            "model": (
                model
                or get_answer_model(
                    selected_provider
                )
            ),
        }

    prompt = build_rag_prompt(
        question=cleaned_question,
        search_results=search_results,
        prompt_template=prompt_template,
        prompt_path=prompt_path,
    )

    if selected_provider == "local":
        answer = generate_local_answer(
            prompt=prompt,
            model=model,
            base_url=base_url,
            http_client=http_client,
        )

    else:
        answer = generate_openai_answer(
            prompt=prompt,
            client=client,
            model=model,
        )

    return {
        "answer": answer,
        "sources": format_sources(
            search_results
        ),
        "provider": selected_provider,
        "model": (
            model
            or get_answer_model(
                selected_provider
            )
        ),
    }