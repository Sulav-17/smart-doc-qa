import streamlit as st
from dotenv import load_dotenv

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
    Upload a PDF, extract its text, and prepare it for a future RAG pipeline.

    In this milestone, we are focusing only on PDF text extraction.
    Chunking, embeddings, retrieval, and AI answers will come later.
    """
)


st.sidebar.header("Project Status")
st.sidebar.success("Milestone 1 setup complete")
st.sidebar.info("Milestone 2: PDF text extraction")


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

        total_pages = len(extracted_pages)
        total_characters = len(full_text)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Pages", total_pages)

        with col2:
            st.metric("Extracted Characters", total_characters)

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
                height=400,
            )

            if total_characters > 5000:
                st.caption("Showing the first 5,000 characters only.")

            with st.expander("View text by page"):
                for page in extracted_pages:
                    st.markdown(f"### Page {page['page_number']}")

                    if page["text"]:
                        st.write(page["text"][:1500])

                        if len(page["text"]) > 1500:
                            st.caption("Page preview truncated.")
                    else:
                        st.warning("No text found on this page.")

    except Exception as error:
        st.error("Something went wrong while reading the PDF.")
        st.exception(error)

else:
    st.warning("Upload a PDF to begin.")