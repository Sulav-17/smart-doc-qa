"""
PDF ingestion module.

This module handles:
- reading uploaded PDF files
- extracting text from each page
- keeping page numbers with the extracted text

This is the first step in the RAG pipeline.
"""

from typing import BinaryIO, Any

from pypdf import PdfReader


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text by removing extra blank lines
    and trimming unnecessary whitespace.

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text.
    """
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    return "\n".join(cleaned_lines)


def extract_text_from_pdf(pdf_file: BinaryIO) -> list[dict[str, Any]]:
    """
    Extract text from every page in a PDF file.

    Args:
        pdf_file: A file-like PDF object, such as a Streamlit uploaded file.

    Returns:
        A list of dictionaries. Each dictionary contains:
        - page_number
        - text
    """
    pdf_file.seek(0)

    reader = PdfReader(pdf_file)
    extracted_pages = []

    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_page_text = clean_text(raw_text)

        extracted_pages.append(
            {
                "page_number": page_index,
                "text": cleaned_page_text,
            }
        )

    return extracted_pages


def combine_pages(pages: list[dict[str, Any]]) -> str:
    """
    Combine extracted page text into one readable string.

    Args:
        pages: List of extracted page dictionaries.

    Returns:
        A single string containing all page text.
    """
    page_text_blocks = []

    for page in pages:
        if page["text"]:
            page_text_blocks.append(
                f"[Page {page['page_number']}]\n{page['text']}"
            )

    return "\n\n".join(page_text_blocks)