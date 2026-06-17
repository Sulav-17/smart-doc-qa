import json

import pytest

from src.evaluation import (
    evaluate_answer_result,
    get_source_pages,
    is_abstention,
    load_evaluation_cases,
)


def test_is_abstention_detects_required_message():
    answer = (
        "I do not know based on the "
        "provided document."
    )

    assert is_abstention(answer) is True


def test_is_abstention_rejects_normal_answer():
    answer = (
        "The project takes approximately "
        "three to four weeks."
    )

    assert is_abstention(answer) is False


def test_get_source_pages_returns_unique_sorted_pages():
    sources = [
        {"page_number": 3},
        {"page_number": 1},
        {"page_number": 3},
    ]

    assert get_source_pages(sources) == [1, 3]


def test_supported_answer_passes_with_sources():
    answer_result = {
        "answer": "The project takes 3–4 weeks.",
        "sources": [
            {
                "page_number": 9,
                "chunk_id": 4,
            }
        ],
    }

    result = evaluate_answer_result(
        answer_result=answer_result,
        expected_behavior="answer",
        expected_pages=[9, 10],
    )

    assert result["passed"] is True
    assert result["checks"][
        "sources_present"
    ] is True
    assert result["checks"][
        "expected_page_retrieved"
    ] is True


def test_supported_answer_without_sources_fails():
    answer_result = {
        "answer": "The project takes 3–4 weeks.",
        "sources": [],
    }

    result = evaluate_answer_result(
        answer_result=answer_result,
        expected_behavior="answer",
    )

    assert result["passed"] is False
    assert result["checks"][
        "sources_present"
    ] is False


def test_expected_abstention_passes():
    answer_result = {
        "answer": (
            "I do not know based on "
            "the provided document."
        ),
        "sources": [],
    }

    result = evaluate_answer_result(
        answer_result=answer_result,
        expected_behavior="abstain",
    )

    assert result["passed"] is True


def test_expected_abstention_fails_for_made_up_answer():
    answer_result = {
        "answer": "Sulav's favorite movie is Inception.",
        "sources": [],
    }

    result = evaluate_answer_result(
        answer_result=answer_result,
        expected_behavior="abstain",
    )

    assert result["passed"] is False


def test_load_evaluation_cases(tmp_path):
    cases = [
        {
            "id": "case-one",
            "question": "What is RAG?",
            "expected_behavior": "answer",
            "expected_pages": [1],
        }
    ]

    case_path = tmp_path / "cases.json"

    case_path.write_text(
        json.dumps(cases),
        encoding="utf-8",
    )

    loaded_cases = load_evaluation_cases(
        str(case_path)
    )

    assert len(loaded_cases) == 1
    assert loaded_cases[0]["id"] == "case-one"


def test_invalid_expected_behavior_raises_error():
    answer_result = {
        "answer": "An answer",
        "sources": [],
    }

    with pytest.raises(ValueError):
        evaluate_answer_result(
            answer_result=answer_result,
            expected_behavior="guess",
        )