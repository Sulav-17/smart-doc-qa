import pytest

from src.chunking import (
    chunk_document,
    normalize_text,
    split_text_into_chunks,
    validate_chunk_settings,
)


def test_normalize_text_removes_extra_whitespace():
    raw_text = "Hello     world\n\nThis is     a test."
    normalized = normalize_text(raw_text)

    assert normalized == "Hello world This is a test."


def test_split_text_into_chunks_creates_chunks():
    text = "a" * 250

    chunks = split_text_into_chunks(
        text=text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_document_preserves_page_numbers():
    pages = [
        {"page_number": 1, "text": "First page text. " * 50},
        {"page_number": 2, "text": "Second page text. " * 50},
    ]

    chunks = chunk_document(
        pages=pages,
        chunk_size=100,
        chunk_overlap=20,
    )

    page_numbers = [chunk["page_number"] for chunk in chunks]

    assert 1 in page_numbers
    assert 2 in page_numbers


def test_chunk_document_adds_chunk_ids():
    pages = [
        {"page_number": 1, "text": "Some text. " * 50},
    ]

    chunks = chunk_document(
        pages=pages,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert chunks[0]["chunk_id"] == 1
    assert chunks[1]["chunk_id"] == 2


def test_invalid_chunk_settings_raise_error():
    with pytest.raises(ValueError):
        validate_chunk_settings(chunk_size=100, chunk_overlap=100)