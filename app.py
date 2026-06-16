import streamlit as st
from dotenv import load_dotenv

from src.chunking import chunk_document
from src.ingestion import combine_pages, extract_text_from_pdf


load_dotenv()


st.set_page_config(
    page_title="Smart Document Q&A",
    page_icon="📄",
    layout="wide",
)


st.title("📄 Smart Document Q&A System")

st.write(
    """
    Upload a PDF, extract its text, and split it into chunks for a future RAG pipeline.

    In this milestone, we are focusing on document chunking.
    Embeddings, retrieval, and AI answers will come later.
    """
)


st.sidebar.header("Project Status")
st.sidebar.success("Milestone 1: Setup complete")
st.sidebar.success("Milestone 2: PDF text extraction complete")
st.sidebar.info("Milestone 3: Document chunking")


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
                height=300,
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

                    These chunks will later be converted into embeddings and stored in a vector database.
                    """
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