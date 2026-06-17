import os

import streamlit as st
from dotenv import load_dotenv

from src.chunking import chunk_document
from src.embeddings import create_embedding, embed_chunks, get_embedding_model
from src.ingestion import combine_pages, extract_text_from_pdf
from src.qa_chain import generate_answer, get_answer_model
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


st.title("📄 Smart Document Q&A System")

st.write(
    """
    Upload a PDF, extract its text, split it into chunks, generate embeddings,
    store them in ChromaDB, retrieve relevant chunks, and generate a grounded answer.

    In this milestone, we are focusing on RAG answer generation with source passages.
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

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "rag_answer" not in st.session_state:
    st.session_state.rag_answer = None


st.sidebar.header("Project Status")
st.sidebar.success("Milestone 1: Setup complete")
st.sidebar.success("Milestone 2: PDF text extraction complete")
st.sidebar.success("Milestone 3: Document chunking complete")
st.sidebar.success("Milestone 4: Embeddings setup complete")
st.sidebar.success("Milestone 5: ChromaDB vector storage complete")
st.sidebar.success("Milestone 6: Vector similarity search complete")
st.sidebar.info("Milestone 7: RAG answer generation")


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
embedding_model = get_embedding_model()
answer_model = get_answer_model()

st.sidebar.write(f"Embedding model: `{embedding_model}`")
st.sidebar.write(f"Answer model: `{answer_model}`")


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
                height=220,
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

                    These chunks can be embedded, stored, searched, and used for grounded answers.
                    """
                )

                api_key_exists = bool(os.getenv("OPENAI_API_KEY"))

                if not api_key_exists:
                    st.warning(
                        """
                        OPENAI_API_KEY was not found.

                        Add your API key to a local `.env` file before generating embeddings or answers.
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
                            st.session_state.search_results = []
                            st.session_state.rag_answer = None

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

                        st.subheader("Ask a Question")

                        user_question = st.text_input(
                            "Ask a question about the uploaded document",
                            placeholder="Example: What is this document about?",
                        )

                        top_k = st.slider(
                            "Number of chunks to retrieve",
                            min_value=1,
                            max_value=5,
                            value=3,
                        )

                        if st.button("Retrieve relevant chunks"):
                            if not user_question.strip():
                                st.warning("Please enter a question first.")
                            else:
                                try:
                                    with st.spinner(
                                        "Embedding question and searching ChromaDB..."
                                    ):
                                        question_embedding = create_embedding(
                                            user_question
                                        )

                                        search_results = search_similar_chunks(
                                            query_embedding=question_embedding,
                                            top_k=top_k,
                                        )

                                    st.session_state.search_results = search_results
                                    st.session_state.rag_answer = None

                                    if not search_results:
                                        st.warning("No matching chunks found.")
                                    else:
                                        st.success(
                                            f"Retrieved {len(search_results)} relevant chunks."
                                        )

                                except Exception as error:
                                    st.error("Could not run similarity search.")

                                    st.warning(
                                        """
                                        This usually happens when the OpenAI API key is missing,
                                        billing is not enabled, or the account has no remaining API quota.

                                        The stored ChromaDB chunks are still valid.
                                        """
                                    )

                                    st.exception(error)

                        if st.session_state.search_results:
                            st.subheader("Retrieved Chunks")

                            for result in st.session_state.search_results:
                                st.markdown(
                                    f"### Page {result['page_number']} — Chunk {result['chunk_id']}"
                                )

                                st.caption(
                                    f"Distance: {result['distance']:.4f}"
                                )

                                st.write(result["text"])

                            if st.button("Generate grounded answer"):
                                try:
                                    with st.spinner(
                                        "Generating grounded answer from retrieved chunks..."
                                    ):
                                        rag_answer = generate_answer(
                                            question=user_question,
                                            search_results=st.session_state.search_results,
                                        )

                                    st.session_state.rag_answer = rag_answer

                                except Exception as error:
                                    st.error("Could not generate grounded answer.")

                                    st.warning(
                                        """
                                        This usually happens when OpenAI API billing or quota is unavailable.

                                        Your retrieval pipeline is still valid.
                                        """
                                    )

                                    st.exception(error)

                        if st.session_state.rag_answer:
                            st.subheader("Grounded Answer")

                            st.write(st.session_state.rag_answer["answer"])

                            st.subheader("Sources")

                            for source in st.session_state.rag_answer["sources"]:
                                with st.expander(source["source"]):
                                    if source["distance"] is not None:
                                        st.caption(
                                            f"Similarity distance: {source['distance']:.4f}"
                                        )

                                    st.write(source["text"])

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