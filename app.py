import streamlit as st
from dotenv import load_dotenv


load_dotenv()


st.set_page_config(
    page_title="Smart Document Q&A",
    page_icon="📄",
    layout="wide",
)


st.title("📄 Smart Document Q&A System")

st.write(
    """
    Upload a PDF, ask questions, and get answers grounded in the document.

    This is the starting version of the app.  
    PDF extraction, chunking, embeddings, retrieval, and AI answers will be added step by step.
    """
)


st.sidebar.header("Project Status")

st.sidebar.success("Milestone 1 setup complete")
st.sidebar.write("Next: PDF text extraction")


uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_file:
    st.info("PDF uploaded successfully. Text extraction will be added in the next milestone.")
else:
    st.warning("Upload a PDF to begin.")