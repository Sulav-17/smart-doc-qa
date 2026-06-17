from src.retriever import search_similar_chunks, store_embedded_chunks


def sample_embedded_chunks():
    return [
        {
            "chunk_id": 1,
            "page_number": 1,
            "text": "Python is a programming language used for data science.",
            "character_count": 57,
            "embedding": [1.0, 0.0, 0.0],
            "embedding_model": "fake-embedding-model",
        },
        {
            "chunk_id": 2,
            "page_number": 2,
            "text": "Bananas are yellow fruits that grow in tropical regions.",
            "character_count": 58,
            "embedding": [0.0, 1.0, 0.0],
            "embedding_model": "fake-embedding-model",
        },
        {
            "chunk_id": 3,
            "page_number": 3,
            "text": "Streamlit is useful for building Python web apps.",
            "character_count": 52,
            "embedding": [0.9, 0.1, 0.0],
            "embedding_model": "fake-embedding-model",
        },
    ]


def test_search_similar_chunks_returns_top_results(tmp_path):
    persist_directory = str(tmp_path / "test_chroma_db")

    store_embedded_chunks(
        embedded_chunks=sample_embedded_chunks(),
        persist_directory=persist_directory,
        collection_name="test_search_chunks",
    )

    results = search_similar_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        persist_directory=persist_directory,
        collection_name="test_search_chunks",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["page_number"] == 1
    assert results[0]["chunk_id"] == 1
    assert "Python" in results[0]["text"]


def test_search_similar_chunks_returns_empty_list_for_empty_collection(tmp_path):
    persist_directory = str(tmp_path / "empty_chroma_db")

    results = search_similar_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        persist_directory=persist_directory,
        collection_name="empty_test_collection",
        top_k=2,
    )

    assert results == []