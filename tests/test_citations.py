from src.citations import (
    build_context_block,
    build_source_label,
    format_source_passage,
    format_source_passages,
)


def sample_search_results():
    return [
        {
            "id": "page-1-chunk-1",
            "text": "This document explains retrieval augmented generation.",
            "metadata": {"page_number": 1, "chunk_id": 1},
            "distance": 0.12,
            "page_number": 1,
            "chunk_id": 1,
        },
        {
            "id": "page-2-chunk-3",
            "text": "Embeddings help compare the meaning of text.",
            "metadata": {"page_number": 2, "chunk_id": 3},
            "distance": 0.25,
            "page_number": 2,
            "chunk_id": 3,
        },
    ]


def test_build_source_label_returns_page_and_chunk():
    result = sample_search_results()[0]

    label = build_source_label(result)

    assert label == "Page 1, Chunk 1"


def test_format_source_passage_returns_expected_fields():
    result = sample_search_results()[0]

    source = format_source_passage(result)

    assert source["source"] == "Page 1, Chunk 1"
    assert source["page_number"] == 1
    assert source["chunk_id"] == 1
    assert "retrieval augmented generation" in source["text"]


def test_format_source_passages_formats_multiple_results():
    sources = format_source_passages(sample_search_results())

    assert len(sources) == 2
    assert sources[0]["source"] == "Page 1, Chunk 1"
    assert sources[1]["source"] == "Page 2, Chunk 3"


def test_build_context_block_includes_sources_and_text():
    context = build_context_block(sample_search_results())

    assert "[Page 1, Chunk 1]" in context
    assert "retrieval augmented generation" in context
    assert "[Page 2, Chunk 3]" in context
    assert "Embeddings help compare" in context