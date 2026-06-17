"""
Chat helper module.

This module handles:
- initializing chat and document state
- creating consistent chat message dictionaries
- identifying uploaded documents
- resetting document and conversation state

Streamlit-specific rendering remains inside app.py.
"""

import copy
import hashlib
from collections.abc import MutableMapping
from typing import Any


DEFAULT_SESSION_STATE = {
    "document_ready": False,
    "document_fingerprint": None,
    "document_name": None,
    "document_stats": {},
    "extracted_pages": [],
    "full_text": "",
    "chunks": [],
    "messages": [],
}


def initialize_session_state(
    state: MutableMapping[str, Any],
) -> None:
    """
    Add missing application state values.

    Existing values are preserved.

    Args:
        state: Streamlit session state or a dictionary-like object.
    """
    for key, default_value in DEFAULT_SESSION_STATE.items():
        if key not in state:
            state[key] = copy.deepcopy(default_value)


def get_file_fingerprint(file_bytes: bytes) -> str:
    """
    Create a stable identifier for an uploaded file.

    Args:
        file_bytes: Raw uploaded file bytes.

    Returns:
        SHA-256 hexadecimal fingerprint.

    Raises:
        ValueError: If the file is empty.
    """
    if not file_bytes:
        raise ValueError("Cannot fingerprint an empty file.")

    return hashlib.sha256(file_bytes).hexdigest()


def create_user_message(content: str) -> dict[str, Any]:
    """
    Create a user chat message.

    Args:
        content: User's question.

    Returns:
        Chat message dictionary.

    Raises:
        ValueError: If the message is empty.
    """
    cleaned_content = content.strip()

    if not cleaned_content:
        raise ValueError("Chat message cannot be empty.")

    return {
        "role": "user",
        "content": cleaned_content,
        "sources": [],
    }


def create_assistant_message(
    answer_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Create an assistant message from a RAG answer result.

    Args:
        answer_result: Dictionary containing answer and sources.

    Returns:
        Chat message dictionary.
    """
    answer = str(answer_result.get("answer", "")).strip()
    sources = answer_result.get("sources") or []

    if not answer:
        answer = "I could not generate an answer."

    return {
        "role": "assistant",
        "content": answer,
        "sources": sources,
    }


def clear_chat_history(
    state: MutableMapping[str, Any],
) -> None:
    """
    Remove conversation messages without removing the processed document.

    Args:
        state: Streamlit session state or dictionary-like object.
    """
    state["messages"] = []


def reset_document_state(
    state: MutableMapping[str, Any],
) -> None:
    """
    Clear the processed document and conversation state.

    Args:
        state: Streamlit session state or dictionary-like object.
    """
    for key, default_value in DEFAULT_SESSION_STATE.items():
        state[key] = copy.deepcopy(default_value)