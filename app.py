import streamlit as st
from dotenv import load_dotenv

from src.chunking import chunk_document
from src.embeddings import (
    create_embedding,
    embed_chunks,
    get_embedding_model,
    get_embedding_provider,
)
from src.ingestion import combine_pages, extract_text_from_pdf
from src.qa_chain import (
    generate_answer,
    get_ai_provider,
    get_answer_model,
)
from src.retriever import (
    get_collection_count,
    preview_collection,
    search_similar_chunks,
    store_embedded_chunks,
)


load_dotenv()


st.set_page_config(
    page_title="Smart Document Q&A",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

DEFAULT_SESSION_STATE = {
    "current_file_name": None,
    "extracted_pages": [],
    "full_text": "",
    "chunks": [],
    "embedded_chunks": [],
    "search_results": [],
    "rag_answer": None,
}


for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def reset_document_state() -> None:
    """
    Reset document-specific state when a different PDF is uploaded.
    """
    st.session_state.extracted_pages = []
    st.session_state.full_text = ""
    st.session_state.chunks = []
    st.session_state.embedded_chunks = []
    st.session_state.search_results = []
    st.session_state.rag_answer = None


# ---------------------------------------------------------
# Page heading
# ---------------------------------------------------------

st.title("📄 Smart Document Q&A System")

st.write(
    """
    Upload a PDF, extract its text, split it into chunks, create local
    embeddings, store them in ChromaDB, retrieve relevant passages,
    and generate a grounded answer using Ollama.
    """
)


# ---------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------

st.sidebar.header("Project Status")

st.sidebar.success("Milestone 1: Project setup")
st.sidebar.success("Milestone 2: PDF extraction")
st.sidebar.success("Milestone 3: Document chunking")
st.sidebar.success("Milestone 4: Embeddings")
st.sidebar.success("Milestone 5: ChromaDB storage")
st.sidebar.success("Milestone 6: Similarity search")
st.sidebar.success("Milestone 7: Grounded answers")
st.sidebar.info("Milestone 8B: Local AI providers")


st.sidebar.header("Chunk Settings")

chunk_size = st.sidebar.slider(
    "Chunk size",
    min_value=300,
    max_value=2000,
    value=1000,
    step=100,
)

chunk_overlap = st.sidebar.slider(
    "Chunk overlap",
    min_value=0,
    max_value=500,
    value=200,
    step=50,
)


st.sidebar.header("AI Settings")

st.sidebar.write(
    f"Embedding provider: `{get_embedding_provider()}`"
)

st.sidebar.write(
    f"Embedding model: `{get_embedding_model()}`"
)

st.sidebar.write(
    f"Answer provider: `{get_ai_provider()}`"
)

st.sidebar.write(
    f"Answer model: `{get_answer_model()}`"
)


# ---------------------------------------------------------
# File upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)


if uploaded_file is None:
    st.info("Upload a text-based PDF to begin.")
    st.stop()


# Reset old results when a different PDF is uploaded.
if st.session_state.current_file_name != uploaded_file.name:
    reset_document_state()
    st.session_state.current_file_name = uploaded_file.name


st.success(f"Uploaded file: {uploaded_file.name}")


# ---------------------------------------------------------
# PDF extraction and chunking
# ---------------------------------------------------------

try:
    with st.spinner("Extracting text from PDF..."):
        extracted_pages = extract_text_from_pdf(
            uploaded_file
        )

        full_text = combine_pages(
            extracted_pages
        )

    with st.spinner("Splitting document into chunks..."):
        chunks = chunk_document(
            pages=extracted_pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    st.session_state.extracted_pages = extracted_pages
    st.session_state.full_text = full_text
    st.session_state.chunks = chunks

except Exception as error:
    st.error("Something went wrong while reading the PDF.")
    st.exception(error)
    st.stop()


total_pages = len(extracted_pages)
total_characters = len(full_text)
total_chunks = len(chunks)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Pages",
        total_pages,
    )

with col2:
    st.metric(
        "Extracted Characters",
        total_characters,
    )

with col3:
    st.metric(
        "Total Chunks",
        total_chunks,
    )


if not full_text.strip():
    st.warning(
        """
        No readable text was extracted from this PDF.

        The PDF may be scanned or image-based. OCR support has not
        been added to this MVP.
        """
    )
    st.stop()


if not chunks:
    st.warning("No chunks were created from the document.")
    st.stop()


# ---------------------------------------------------------
# Extracted text preview
# ---------------------------------------------------------

st.subheader("Extracted Text Preview")

st.text_area(
    "Document text",
    value=full_text[:5000],
    height=250,
    disabled=True,
)

if total_characters > 5000:
    st.caption("Showing the first 5,000 characters only.")


# ---------------------------------------------------------
# Chunk preview
# ---------------------------------------------------------

st.subheader("Document Chunks")

st.write(
    f"The document was split into **{total_chunks} chunks**."
)

with st.expander("View document chunks"):
    for chunk in chunks:
        st.markdown(
            f"### Chunk {chunk['chunk_id']} — "
            f"Page {chunk['page_number']}"
        )

        st.caption(
            f"Character count: "
            f"{chunk['character_count']}"
        )

        st.write(chunk["text"])


# ---------------------------------------------------------
# Local embeddings
# ---------------------------------------------------------

st.subheader("Step 1 — Generate Local Embeddings")

if st.button(
    "Generate embeddings",
    type="primary",
):
    try:
        with st.spinner(
            "Generating local document embeddings..."
        ):
            embedded_chunks = embed_chunks(
                chunks
            )

        st.session_state.embedded_chunks = (
            embedded_chunks
        )

        st.session_state.search_results = []
        st.session_state.rag_answer = None

        st.success(
            "Local embeddings generated successfully."
        )

    except Exception as error:
        st.error(
            "Could not generate local embeddings."
        )

        st.warning(
            """
            Check that `sentence-transformers` is installed
            and that the local embedding model downloaded correctly.
            """
        )

        st.exception(error)


if st.session_state.embedded_chunks:
    embedded_chunks = (
        st.session_state.embedded_chunks
    )

    embedding_count = len(embedded_chunks)

    first_embedding_length = len(
        embedded_chunks[0]["embedding"]
    )

    col4, col5 = st.columns(2)

    with col4:
        st.metric(
            "Embedded Chunks",
            embedding_count,
        )

    with col5:
        st.metric(
            "Embedding Dimensions",
            first_embedding_length,
        )

    with st.expander(
        "Preview first embedded chunk"
    ):
        first_chunk = embedded_chunks[0]

        st.write("Chunk text:")

        st.write(
            first_chunk["text"]
        )

        st.write("Embedding preview:")

        st.write(
            first_chunk["embedding"][:10]
        )

        st.caption(
            "Showing the first 10 numbers only."
        )


# ---------------------------------------------------------
# ChromaDB storage
# ---------------------------------------------------------

if st.session_state.embedded_chunks:
    st.subheader(
        "Step 2 — Store Embeddings in ChromaDB"
    )

    if st.button(
        "Store embeddings in ChromaDB"
    ):
        try:
            with st.spinner(
                "Storing embeddings in ChromaDB..."
            ):
                stored_count = store_embedded_chunks(
                    st.session_state.embedded_chunks
                )

            st.success(
                f"Stored {stored_count} chunks "
                f"in ChromaDB."
            )

            st.session_state.search_results = []
            st.session_state.rag_answer = None

        except Exception as error:
            st.error(
                "Could not store embeddings in ChromaDB."
            )

            st.exception(error)


    collection_count = get_collection_count()

    st.metric(
        "Records in ChromaDB",
        collection_count,
    )


    if collection_count > 0:
        stored_preview = preview_collection(
            limit=3
        )

        with st.expander(
            "Preview stored ChromaDB records"
        ):
            documents = (
                stored_preview.get("documents")
                or []
            )

            metadatas = (
                stored_preview.get("metadatas")
                or []
            )

            for index, document in enumerate(
                documents
            ):
                st.markdown(
                    f"### Stored Record {index + 1}"
                )

                st.write(document)

                if index < len(metadatas):
                    metadata = metadatas[index]

                    st.caption(
                        f"Page: "
                        f"{metadata.get('page_number')} | "
                        f"Chunk: "
                        f"{metadata.get('chunk_id')} | "
                        f"Provider: "
                        f"{metadata.get('embedding_provider', 'unknown')}"
                    )


# ---------------------------------------------------------
# Similarity search
# ---------------------------------------------------------

collection_count = get_collection_count()


if collection_count > 0:
    st.subheader(
        "Step 3 — Retrieve Relevant Chunks"
    )

    user_question = st.text_input(
        "Ask a question about the uploaded document",
        placeholder=(
            "Example: What is the main topic "
            "of this document?"
        ),
    )

    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=5,
        value=3,
    )


    if st.button(
        "Retrieve relevant chunks"
    ):
        if not user_question.strip():
            st.warning(
                "Please enter a question first."
            )

        else:
            try:
                with st.spinner(
                    "Embedding question and "
                    "searching ChromaDB..."
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

                st.session_state.search_results = (
                    search_results
                )

                st.session_state.rag_answer = None

                if search_results:
                    st.success(
                        f"Retrieved "
                        f"{len(search_results)} "
                        f"relevant chunks."
                    )

                else:
                    st.warning(
                        "No matching chunks were found."
                    )

            except Exception as error:
                st.error(
                    "Could not run similarity search."
                )

                st.warning(
                    """
                    Check that the stored ChromaDB embeddings
                    and the question embedding use the same
                    local embedding model.
                    """
                )

                st.exception(error)


# ---------------------------------------------------------
# Retrieved chunks
# ---------------------------------------------------------

if st.session_state.search_results:
    st.subheader("Retrieved Chunks")

    for result in (
        st.session_state.search_results
    ):
        st.markdown(
            f"### Page "
            f"{result['page_number']} — "
            f"Chunk {result['chunk_id']}"
        )

        distance = result.get("distance")

        if distance is not None:
            st.caption(
                f"Similarity distance: "
                f"{distance:.4f}"
            )

        st.write(result["text"])


# ---------------------------------------------------------
# Ollama grounded answer
# ---------------------------------------------------------

if st.session_state.search_results:
    st.subheader(
        "Step 4 — Generate Grounded Answer"
    )

    if st.button(
        "Generate grounded answer"
    ):
        if not user_question.strip():
            st.warning(
                "Please enter a question first."
            )

        else:
            try:
                with st.spinner(
                    "Generating a grounded answer "
                    "with local Ollama..."
                ):
                    rag_answer = generate_answer(
                        question=user_question,
                        search_results=(
                            st.session_state
                            .search_results
                        ),
                    )

                st.session_state.rag_answer = (
                    rag_answer
                )

                st.success(
                    "Grounded answer generated."
                )

            except Exception as error:
                st.error(
                    "Could not generate the "
                    "grounded answer."
                )

                st.warning(
                    """
                    Check that Ollama is running and that
                    the configured model has been downloaded.

                    Expected model: `qwen3:4b`
                    """
                )

                st.exception(error)


# ---------------------------------------------------------
# Final answer and sources
# ---------------------------------------------------------

if st.session_state.rag_answer:
    st.subheader("Grounded Answer")

    st.write(
        st.session_state
        .rag_answer["answer"]
    )

    st.subheader("Sources")

    sources = (
        st.session_state
        .rag_answer.get("sources", [])
    )

    if not sources:
        st.info(
            "No source passages were returned."
        )

    for source in sources:
        with st.expander(
            source["source"]
        ):
            distance = source.get(
                "distance"
            )

            if distance is not None:
                st.caption(
                    f"Similarity distance: "
                    f"{distance:.4f}"
                )

            st.write(
                source["text"]
            )


# ---------------------------------------------------------
# Page-by-page extraction preview
# ---------------------------------------------------------

with st.expander(
    "View extracted text by page"
):
    for page in extracted_pages:
        st.markdown(
            f"### Page {page['page_number']}"
        )

        if page["text"]:
            st.write(
                page["text"][:1500]
            )

            if len(page["text"]) > 1500:
                st.caption(
                    "Page preview truncated."
                )

        else:
            st.warning(
                "No text found on this page."
            )