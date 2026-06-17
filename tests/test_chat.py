import pytest

from src.chat import (
    clear_chat_history,
    create_assistant_message,
    create_user_message,
    get_file_fingerprint,
    initialize_session_state,
    reset_document_state,
)


def test_initialize_session_state_adds_defaults():
    state = {}

    initialize_session_state(state)

    assert state["document_ready"] is False
    assert state["messages"] == []
    assert state["chunks"] == []


def test_initialize_session_state_preserves_existing_values():
    state = {
        "document_ready": True,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    initialize_session_state(state)

    assert state["document_ready"] is True
    assert len(state["messages"]) == 1


def test_file_fingerprint_is_stable():
    file_bytes = b"example PDF bytes"

    first_fingerprint = get_file_fingerprint(file_bytes)
    second_fingerprint = get_file_fingerprint(file_bytes)

    assert first_fingerprint == second_fingerprint


def test_different_files_have_different_fingerprints():
    first_fingerprint = get_file_fingerprint(b"first file")
    second_fingerprint = get_file_fingerprint(b"second file")

    assert first_fingerprint != second_fingerprint


def test_empty_file_fingerprint_raises_error():
    with pytest.raises(ValueError):
        get_file_fingerprint(b"")


def test_create_user_message_cleans_content():
    message = create_user_message("  What is RAG?  ")

    assert message["role"] == "user"
    assert message["content"] == "What is RAG?"
    assert message["sources"] == []


def test_create_user_message_rejects_empty_content():
    with pytest.raises(ValueError):
        create_user_message("   ")


def test_create_assistant_message_keeps_sources():
    answer_result = {
        "answer": "RAG retrieves relevant document chunks.",
        "sources": [
            {
                "source": "Page 1, Chunk 2",
                "text": "Source passage",
            }
        ],
    }

    message = create_assistant_message(answer_result)

    assert message["role"] == "assistant"
    assert "retrieves relevant" in message["content"]
    assert len(message["sources"]) == 1


def test_clear_chat_history_keeps_document_state():
    state = {
        "document_ready": True,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    clear_chat_history(state)

    assert state["messages"] == []
    assert state["document_ready"] is True


def test_reset_document_state_clears_document_and_messages():
    state = {
        "document_ready": True,
        "document_name": "report.pdf",
        "messages": [{"role": "user", "content": "Hello"}],
        "chunks": [{"chunk_id": 1}],
    }

    reset_document_state(state)

    assert state["document_ready"] is False
    assert state["document_name"] is None
    assert state["messages"] == []
    assert state["chunks"] == []