import os

import streamlit as st
from dotenv import load_dotenv

from src.chunking import chunk_document
from src.embeddings import embed_chunks, get_embedding_model
from src.ingestion import combine_pages, extract_text_from_pdf
from src.retriever import (
    get_collection_count,
    preview_collection,
    store_embedded_chunks,
)


load_dotenv()


st.set_page_config(
    page_title="Smart Document Q&A",
    page_icon="📄",
    layout="wide",
)


st.title("📄 Smart Document Q&A System")

st.write(
    """
    Upload a PDF, extract its text, split it into chunks, generate embeddings,
    and store them in a local ChromaDB vector database.

    In this milestone, we are focusing on vector database storage.
    Vector similarity search and AI answers will come later.
    """
)


if "extracted_pages" not in st.session_state:
    st.session_state.extracted_pages = None

if "full_text" not in st.session_state:
    st.session_state.full_text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embedded_chunks" not in st.session_state:
    st.session_state.embedded_chunks = []


st.sidebar.header("Project Status")
st.sidebar.success("Milestone 1: Setup complete")
st.sidebar.success("Milestone 2: PDF text extraction complete")
st.sidebar.success("Milestone 3: Document chunking complete")
st.sidebar.success("Milestone 4: Embeddings setup complete")
st.sidebar.info("Milestone 5: ChromaDB vector storage")


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


st.sidebar.header("Embedding Settings")
embedding_model = get_embedding_model()
st.sidebar.write(f"Model: `{embedding_model}`")


uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)


if uploaded_file:
    st.success(f"Uploaded file: {uploaded_file.name}")

    try:
        with st.spinner("Extracting text from PDF..."):
            extracted_pages = extract_text_from_pdf(uploaded_file)
            full_text = combine_pages(extracted_pages)

        with st.spinner("Splitting document into chunks..."):
            chunks = chunk_document(
                pages=extracted_pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        st.session_state.extracted_pages = extracted_pages
        st.session_state.full_text = full_text
        st.session_state.chunks = chunks

        total_pages = len(extracted_pages)
        total_characters = len(full_text)
        total_chunks = len(chunks)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Pages", total_pages)

        with col2:
            st.metric("Extracted Characters", total_characters)

        with col3:
            st.metric("Total Chunks", total_chunks)

        if total_characters == 0:
            st.warning(
                """
                No readable text was extracted from this PDF.

                This can happen if the PDF is scanned or image-based.
                OCR support will not be added yet because we are keeping the MVP simple.
                """
            )

        else:
            st.subheader("Extracted Text Preview")

            st.text_area(
                "Preview",
                value=full_text[:5000],
                height=250,
            )

            if total_characters > 5000:
                st.caption("Showing the first 5,000 characters only.")

            st.subheader("Document Chunks")

            if not chunks:
                st.warning("No chunks were created from this document.")

            else:
                st.write(
                    f"""
                    The document was split into **{total_chunks} chunks**.

                    These chunks can now be converted into embeddings.
                    """
                )

                api_key_exists = bool(os.getenv("OPENAI_API_KEY"))

                if not api_key_exists:
                    st.warning(
                        """
                        OPENAI_API_KEY was not found.

                        Add your API key to a local `.env` file before generating embeddings.
                        """
                    )

                else:
                    if st.button("Generate embeddings"):
                        try:
                            with st.spinner(
                                "Generating embeddings for document chunks..."
                            ):
                                embedded_chunks = embed_chunks(chunks)

                            st.session_state.embedded_chunks = embedded_chunks

                            st.success("Embeddings generated successfully.")

                        except Exception as error:
                            st.error("Could not generate embeddings.")

                            st.warning(
                                """
                                This usually happens when the OpenAI API key is missing,
                                billing is not enabled, or the account has no remaining API quota.

                                Your PDF extraction and chunking still work.
                                """
                            )

                            st.exception(error)

                if st.session_state.embedded_chunks:
                    embedded_chunks = st.session_state.embedded_chunks

                    st.subheader("Embedding Results")

                    embedding_count = len(embedded_chunks)
                    first_embedding_length = len(
                        embedded_chunks[0]["embedding"]
                    )

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.metric("Embedded Chunks", embedding_count)

                    with col_b:
                        st.metric(
                            "Embedding Dimensions",
                            first_embedding_length,
                        )

                    with st.expander("Preview first embedded chunk"):
                        first_chunk = embedded_chunks[0]

                        st.write("Chunk text:")
                        st.write(first_chunk["text"])

                        st.write("Embedding preview:")
                        st.write(first_chunk["embedding"][:10])

                        st.caption(
                            "Showing only the first 10 numbers from the embedding."
                        )

                    st.subheader("Vector Database Storage")

                    if st.button("Store embeddings in ChromaDB"):
                        with st.spinner("Storing embeddings in ChromaDB..."):
                            stored_count = store_embedded_chunks(
                                embedded_chunks
                            )

                        st.success(
                            f"Stored {stored_count} chunks in ChromaDB."
                        )

                    collection_count = get_collection_count()

                    st.metric("Records in ChromaDB", collection_count)

                    if collection_count > 0:
                        stored_preview = preview_collection(limit=3)

                        with st.expander("Preview stored ChromaDB records"):
                            for index, document in enumerate(
                                stored_preview["documents"]
                            ):
                                st.markdown(f"### Stored Record {index + 1}")
                                st.write(document)

                                metadata = stored_preview["metadatas"][index]

                                st.caption(
                                    f"Page: {metadata['page_number']} | "
                                    f"Chunk: {metadata['chunk_id']}"
                                )

                with st.expander("View chunks"):
                    for chunk in chunks:
                        st.markdown(
                            f"### Chunk {chunk['chunk_id']} — Page {chunk['page_number']}"
                        )

                        st.caption(
                            f"Character count: {chunk['character_count']}"
                        )

                        st.write(chunk["text"])

            with st.expander("View extracted text by page"):
                for page in extracted_pages:
                    st.markdown(f"### Page {page['page_number']}")

                    if page["text"]:
                        st.write(page["text"][:1500])

                        if len(page["text"]) > 1500:
                            st.caption("Page preview truncated.")

                    else:
                        st.warning("No text found on this page.")

    except Exception as error:
        st.error("Something went wrong while processing the PDF.")
        st.exception(error)

else:
    st.warning("Upload a PDF to begin.")