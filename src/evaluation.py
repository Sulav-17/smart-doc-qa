"""
RAG evaluation helpers.

This module handles:
- loading evaluation cases
- detecting expected abstentions
- checking source availability
- checking expected source pages
- producing structured evaluation results

These checks do not prove that every answer is correct.
They provide a repeatable baseline for manual RAG evaluation.
"""

import json
from pathlib import Path
from typing import Any


ABSTENTION_TEXT = (
    "I do not know based on the provided document."
)

VALID_EXPECTED_BEHAVIORS = {
    "answer",
    "abstain",
}


def normalize_text(text: str) -> str:
    """
    Normalize text for simple comparisons.

    Args:
        text: Input text.

    Returns:
        Lowercase text with repeated whitespace removed.
    """
    return " ".join(text.lower().split())


def is_abstention(answer: str) -> bool:
    """
    Check whether an answer contains the required abstention message.

    Args:
        answer: Generated answer.

    Returns:
        True when the required abstention text is present.
    """
    normalized_answer = normalize_text(answer)
    normalized_abstention = normalize_text(
        ABSTENTION_TEXT
    )

    return normalized_abstention in normalized_answer


def get_source_pages(
    sources: list[dict[str, Any]],
) -> list[int]:
    """
    Extract unique page numbers from formatted sources.

    Args:
        sources: Source dictionaries returned by the RAG pipeline.

    Returns:
        Sorted unique page numbers.
    """
    pages = {
        int(source["page_number"])
        for source in sources
        if source.get("page_number") is not None
    }

    return sorted(pages)


def load_evaluation_cases(
    file_path: str,
) -> list[dict[str, Any]]:
    """
    Load evaluation cases from JSON.

    Args:
        file_path: Path to the evaluation JSON file.

    Returns:
        List of evaluation cases.

    Raises:
        ValueError: If cases are missing required fields.
    """
    path = Path(file_path)

    cases = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(cases, list):
        raise ValueError(
            "Evaluation cases must be stored in a JSON list."
        )

    for case in cases:
        if not case.get("id"):
            raise ValueError(
                "Every evaluation case requires an id."
            )

        if not case.get("question"):
            raise ValueError(
                "Every evaluation case requires a question."
            )

        expected_behavior = case.get(
            "expected_behavior"
        )

        if (
            expected_behavior
            not in VALID_EXPECTED_BEHAVIORS
        ):
            raise ValueError(
                "expected_behavior must be "
                "'answer' or 'abstain'."
            )

    return cases


def evaluate_answer_result(
    answer_result: dict[str, Any],
    expected_behavior: str,
    expected_pages: list[int] | None = None,
) -> dict[str, Any]:
    """
    Evaluate one RAG answer.

    Args:
        answer_result: RAG answer containing answer and sources.
        expected_behavior: Either "answer" or "abstain".
        expected_pages: Optional acceptable source pages.

    Returns:
        Structured evaluation result.

    Raises:
        ValueError: If expected behavior is unsupported.
    """
    if (
        expected_behavior
        not in VALID_EXPECTED_BEHAVIORS
    ):
        raise ValueError(
            "expected_behavior must be "
            "'answer' or 'abstain'."
        )

    answer = str(
        answer_result.get("answer", "")
    ).strip()

    sources = answer_result.get("sources") or []
    source_pages = get_source_pages(sources)

    checks = {
        "answer_present": bool(answer),
    }

    if expected_behavior == "abstain":
        checks["correct_behavior"] = (
            is_abstention(answer)
        )

    else:
        checks["correct_behavior"] = (
            not is_abstention(answer)
        )

        checks["sources_present"] = (
            len(sources) > 0
        )

    if expected_pages:
        expected_page_set = set(expected_pages)
        actual_page_set = set(source_pages)

        checks["expected_page_retrieved"] = bool(
            expected_page_set
            & actual_page_set
        )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "answer": answer,
        "source_pages": source_pages,
        "source_count": len(sources),
        "expected_behavior": expected_behavior,
        "expected_pages": expected_pages or [],
    }