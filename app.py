from io import BytesIO
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.chat import (
    clear_chat_history,
    create_assistant_message,
    create_ready_message,
    create_user_message,
    get_file_fingerprint,
    initialize_session_state,
    reset_document_state,
)
from src.chunking import chunk_document
from src.embeddings import (
    create_embedding,
    embed_chunks,
    get_embedding_model,
    get_embedding_provider,
)
from src.ingestion import (
    combine_pages,
    extract_text_from_pdf,
)
from src.qa_chain import (
    generate_answer,
    get_ai_provider,
    get_answer_model,
)
from src.retriever import (
    search_similar_chunks,
    store_embedded_chunks,
)


load_dotenv()


st.set_page_config(
    page_title="Smart Document Q&A",
    page_icon="📄",
    layout="wide",
)


initialize_session_state(
    st.session_state
)


def display_sources(
    sources: list[dict[str, Any]],
) -> None:
    """
    Display source passages under an assistant answer.

    Args:
        sources: Formatted citation dictionaries.
    """
    if not sources:
        return

    with st.expander(
        f"View sources ({len(sources)})"
    ):
        for index, source in enumerate(
            sources,
            start=1,
        ):
            st.markdown(
                f"**Source {index}: "
                f"{source['source']}**"
            )

            distance = source.get("distance")

            if distance is not None:
                st.caption(
                    f"Similarity distance: "
                    f"{distance:.4f}"
                )

            st.write(source["text"])

            if index < len(sources):
                st.divider()


def display_chat_message(
    message: dict[str, Any],
) -> None:
    """
    Render one stored chat message.

    Args:
        message: Standard chat message dictionary.
    """
    with st.chat_message(
        message["role"]
    ):
        st.write(
            message["content"]
        )

        if message["role"] == "assistant":
            display_sources(
                message.get("sources", [])
            )


# ---------------------------------------------------------
# Page heading
# ---------------------------------------------------------

st.title("📄 Smart Document Q&A")

st.caption(
    "Upload a PDF and ask grounded questions "
    "using local AI."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Document Settings")


chunk_size = st.sidebar.slider(
    "Chunk size",
    min_value=300,
    max_value=2000,
    value=1000,
    step=100,
    disabled=st.session_state.document_ready,
)


chunk_overlap = st.sidebar.slider(
    "Chunk overlap",
    min_value=0,
    max_value=500,
    value=200,
    step=50,
    disabled=st.session_state.document_ready,
)


top_k = st.sidebar.slider(
    "Sources per answer",
    min_value=1,
    max_value=5,
    value=3,
)


st.sidebar.header("AI Configuration")

st.sidebar.write(
    f"Embedding provider: "
    f"`{get_embedding_provider()}`"
)

st.sidebar.write(
    f"Embedding model: "
    f"`{get_embedding_model()}`"
)

st.sidebar.write(
    f"Answer provider: "
    f"`{get_ai_provider()}`"
)

st.sidebar.write(
    f"Answer model: "
    f"`{get_answer_model()}`"
)


st.sidebar.header("Pipeline")

st.sidebar.success("PDF extraction")
st.sidebar.success("Document chunking")
st.sidebar.success("Local embeddings")
st.sidebar.success("ChromaDB retrieval")
st.sidebar.success("Local Ollama answers")
st.sidebar.info("Chat interface")


if st.sidebar.button(
    "Clear conversation",
    disabled=not bool(
        st.session_state.messages
    ),
    use_container_width=True,
):
    clear_chat_history(
        st.session_state
    )

    st.rerun()


if st.sidebar.button(
    "Reset document",
    disabled=not bool(
        st.session_state.document_name
    ),
    use_container_width=True,
):
    reset_document_state(
        st.session_state
    )

    st.rerun()


# ---------------------------------------------------------
# PDF upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)


if uploaded_file is None:
    st.info(
        "Upload a text-based PDF to begin."
    )

    st.stop()


file_bytes = uploaded_file.getvalue()


try:
    current_fingerprint = get_file_fingerprint(
        file_bytes
    )

except ValueError as error:
    st.error(str(error))
    st.stop()


# A newly uploaded PDF should never reuse the previous PDF's
# chunks, vector records, or conversation.
if (
    st.session_state.document_fingerprint
    != current_fingerprint
):
    reset_document_state(
        st.session_state
    )

    st.session_state.document_fingerprint = (
        current_fingerprint
    )

    st.session_state.document_name = (
        uploaded_file.name
    )


st.success(
    f"Selected document: "
    f"{uploaded_file.name}"
)


# ---------------------------------------------------------
# Document processing
# ---------------------------------------------------------

if not st.session_state.document_ready:
    st.write(
        """
        Click **Process document** to extract the PDF text,
        create chunks, generate local embeddings, and store
        the vectors in ChromaDB.
        """
    )


    if st.button(
        "Process document",
        type="primary",
    ):
        try:
            with st.spinner(
                "Processing the document locally..."
            ):
                pdf_file = BytesIO(
                    file_bytes
                )

                extracted_pages = (
                    extract_text_from_pdf(
                        pdf_file
                    )
                )

                full_text = combine_pages(
                    extracted_pages
                )

                if not full_text.strip():
                    raise ValueError(
                        "No readable text was "
                        "extracted from this PDF."
                    )

                chunks = chunk_document(
                    pages=extracted_pages,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                if not chunks:
                    raise ValueError(
                        "No document chunks "
                        "were created."
                    )

                embedded_chunks = embed_chunks(
                    chunks
                )

                stored_count = (
                    store_embedded_chunks(
                        embedded_chunks
                    )
                )


            st.session_state.extracted_pages = (
                extracted_pages
            )

            st.session_state.full_text = (
                full_text
            )

            st.session_state.chunks = (
                chunks
            )

            st.session_state.document_stats = {
                "pages": len(extracted_pages),
                "characters": len(full_text),
                "chunks": len(chunks),
                "stored_records": stored_count,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }

            st.session_state.document_ready = True

            st.session_state.messages = [
                create_ready_message(
                    uploaded_file.name
                )
            ]

            st.rerun()


        except Exception as error:
            st.error(
                "Could not process the document."
            )

            st.warning(
                """
                Check that SentenceTransformers installed
                correctly and that the PDF contains
                selectable text.
                """
            )

            with st.expander(
                "Technical details"
            ):
                st.exception(error)


    st.stop()


# ---------------------------------------------------------
# Document statistics
# ---------------------------------------------------------

stats = st.session_state.document_stats


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Pages",
        stats.get("pages", 0),
    )


with col2:
    st.metric(
        "Characters",
        stats.get("characters", 0),
    )


with col3:
    st.metric(
        "Chunks",
        stats.get("chunks", 0),
    )


with col4:
    st.metric(
        "Stored Records",
        stats.get("stored_records", 0),
    )


with st.expander(
    "Document processing details"
):
    st.write(
        f"**Document:** "
        f"{st.session_state.document_name}"
    )

    st.write(
        f"**Chunk size:** "
        f"{stats.get('chunk_size')}"
    )

    st.write(
        f"**Chunk overlap:** "
        f"{stats.get('chunk_overlap')}"
    )

    st.write(
        f"**Embedding provider:** "
        f"{get_embedding_provider()}"
    )

    st.write(
        f"**Embedding model:** "
        f"{get_embedding_model()}"
    )

    st.write(
        f"**Answer provider:** "
        f"{get_ai_provider()}"
    )

    st.write(
        f"**Answer model:** "
        f"{get_answer_model()}"
    )

    st.caption(
        "Reset the document before changing "
        "chunk settings."
    )


st.divider()

st.subheader(
    "Chat with your document"
)


# ---------------------------------------------------------
# Existing conversation
# ---------------------------------------------------------

for message in st.session_state.messages:
    display_chat_message(message)


# ---------------------------------------------------------
# New chat message
# ---------------------------------------------------------

user_question = st.chat_input(
    "Ask a question about the uploaded PDF"
)


if user_question:
    try:
        user_message = create_user_message(
            user_question
        )

        st.session_state.messages.append(
            user_message
        )

        display_chat_message(
            user_message
        )


        with st.chat_message("assistant"):
            with st.spinner(
                "Searching the document "
                "and generating an answer..."
            ):
                question_embedding = (
                    create_embedding(
                        text=user_question,
                        task="query",
                    )
                )

                search_results = (
                    search_similar_chunks(
                        query_embedding=(
                            question_embedding
                        ),
                        top_k=top_k,
                    )
                )

                answer_result = generate_answer(
                    question=user_question,
                    search_results=search_results,
                )

                assistant_message = (
                    create_assistant_message(
                        answer_result
                    )
                )


            st.write(
                assistant_message["content"]
            )

            display_sources(
                assistant_message["sources"]
            )


        st.session_state.messages.append(
            assistant_message
        )


    except Exception as error:
        error_message = {
            "role": "assistant",
            "content": (
                "I could not answer that question. "
                "Check that Ollama is running and "
                "that the configured model is installed."
            ),
            "sources": [],
        }

        st.session_state.messages.append(
            error_message
        )


        with st.chat_message("assistant"):
            st.error(
                error_message["content"]
            )

            with st.expander(
                "Technical details"
            ):
                st.exception(error)