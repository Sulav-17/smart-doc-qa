"""
Chat and application-state helpers.

This module handles:
- initializing Streamlit session state
- identifying uploaded documents
- creating consistent chat messages
- clearing conversations
- resetting processed-document state

Streamlit UI rendering remains inside app.py.
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
    Add any missing application state values.

    Existing values are preserved.

    Args:
        state: Streamlit session state or another dictionary-like object.
    """
    for key, default_value in DEFAULT_SESSION_STATE.items():
        if key not in state:
            state[key] = copy.deepcopy(default_value)


def get_file_fingerprint(file_bytes: bytes) -> str:
    """
    Create a stable identifier for an uploaded file.

    Args:
        file_bytes: Raw bytes from an uploaded file.

    Returns:
        SHA-256 fingerprint for the file.

    Raises:
        ValueError: If the uploaded file is empty.
    """
    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    return hashlib.sha256(file_bytes).hexdigest()


def create_user_message(content: str) -> dict[str, Any]:
    """
    Create a user chat message.

    Args:
        content: User question.

    Returns:
        Standard chat message dictionary.

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
    Convert a RAG answer into a chat message.

    Args:
        answer_result: Dictionary containing answer and sources.

    Returns:
        Standard assistant chat message.
    """
    answer = str(
        answer_result.get("answer", "")
    ).strip()

    sources = answer_result.get("sources") or []

    if not answer:
        answer = "I could not generate an answer."

    return {
        "role": "assistant",
        "content": answer,
        "sources": sources,
    }


def create_ready_message(
    document_name: str,
) -> dict[str, Any]:
    """
    Create the first assistant message after processing a PDF.

    Args:
        document_name: Name of the processed PDF.

    Returns:
        Assistant chat message.
    """
    return {
        "role": "assistant",
        "content": (
            f"`{document_name}` is ready. "
            "Ask me a question about the document."
        ),
        "sources": [],
    }


def clear_chat_history(
    state: MutableMapping[str, Any],
) -> None:
    """
    Clear chat messages without deleting the processed document.

    Args:
        state: Streamlit session state or dictionary-like object.
    """
    state["messages"] = []


def reset_document_state(
    state: MutableMapping[str, Any],
) -> None:
    """
    Clear the processed document and conversation.

    Args:
        state: Streamlit session state or dictionary-like object.
    """
    for key, default_value in DEFAULT_SESSION_STATE.items():
        state[key] = copy.deepcopy(default_value)