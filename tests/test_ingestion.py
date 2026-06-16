from src.ingestion import clean_text, combine_pages


def test_clean_text_removes_blank_lines_and_spaces():
    raw_text = "  Hello world  \n\n  This is a test.  \n\n"
    cleaned_text = clean_text(raw_text)

    assert cleaned_text == "Hello world\nThis is a test."


def test_combine_pages_includes_page_numbers():
    pages = [
        {"page_number": 1, "text": "First page text."},
        {"page_number": 2, "text": "Second page text."},
    ]

    combined_text = combine_pages(pages)

    assert "[Page 1]" in combined_text
    assert "First page text." in combined_text
    assert "[Page 2]" in combined_text
    assert "Second page text." in combined_text