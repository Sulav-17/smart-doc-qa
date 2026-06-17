import pytest

from src.chat import (
    clear_chat_history,
    create_assistant_message,
    create_ready_message,
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
        "messages": [
            {
                "role": "user",
                "content": "Hello",
            }
        ],
    }

    initialize_session_state(state)

    assert state["document_ready"] is True
    assert len(state["messages"]) == 1


def test_file_fingerprint_is_stable():
    file_bytes = b"example PDF bytes"

    first_fingerprint = get_file_fingerprint(
        file_bytes
    )

    second_fingerprint = get_file_fingerprint(
        file_bytes
    )

    assert first_fingerprint == second_fingerprint


def test_different_files_have_different_fingerprints():
    first_fingerprint = get_file_fingerprint(
        b"first file"
    )

    second_fingerprint = get_file_fingerprint(
        b"second file"
    )

    assert first_fingerprint != second_fingerprint


def test_empty_file_fingerprint_raises_error():
    with pytest.raises(ValueError):
        get_file_fingerprint(b"")


def test_create_user_message_cleans_content():
    message = create_user_message(
        "  What is RAG?  "
    )

    assert message["role"] == "user"
    assert message["content"] == "What is RAG?"
    assert message["sources"] == []


def test_create_user_message_rejects_empty_content():
    with pytest.raises(ValueError):
        create_user_message("   ")


def test_create_assistant_message_keeps_sources():
    result = {
        "answer": "RAG retrieves relevant passages.",
        "sources": [
            {
                "source": "Page 1, Chunk 2",
                "text": "Source passage",
            }
        ],
    }

    message = create_assistant_message(result)

    assert message["role"] == "assistant"
    assert "retrieves relevant" in message["content"]
    assert len(message["sources"]) == 1


def test_create_ready_message_includes_document_name():
    message = create_ready_message(
        "report.pdf"
    )

    assert message["role"] == "assistant"
    assert "report.pdf" in message["content"]


def test_clear_chat_history_keeps_document():
    state = {
        "document_ready": True,
        "messages": [
            {
                "role": "user",
                "content": "Hello",
            }
        ],
    }

    clear_chat_history(state)

    assert state["messages"] == []
    assert state["document_ready"] is True


def test_reset_document_state_clears_everything():
    state = {
        "document_ready": True,
        "document_name": "report.pdf",
        "document_fingerprint": "abc123",
        "messages": [
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        "chunks": [{"chunk_id": 1}],
    }

    reset_document_state(state)

    assert state["document_ready"] is False
    assert state["document_name"] is None
    assert state["document_fingerprint"] is None
    assert state["messages"] == []
    assert state["chunks"] == []