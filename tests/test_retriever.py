from src.retriever import (
    build_record_id,
    get_collection_count,
    prepare_chroma_records,
    store_embedded_chunks,
)


def sample_embedded_chunks():
    return [
        {
            "chunk_id": 1,
            "page_number": 1,
            "text": "This is the first chunk.",
            "character_count": 24,
            "embedding": [0.1, 0.2, 0.3],
            "embedding_model": "fake-embedding-model",
        },
        {
            "chunk_id": 2,
            "page_number": 1,
            "text": "This is the second chunk.",
            "character_count": 25,
            "embedding": [0.4, 0.5, 0.6],
            "embedding_model": "fake-embedding-model",
        },
    ]


def test_build_record_id_uses_page_and_chunk_id():
    chunk = {
        "chunk_id": 7,
        "page_number": 3,
    }

    record_id = build_record_id(chunk)

    assert record_id == "page-3-chunk-7"


def test_prepare_chroma_records_creates_expected_lists():
    embedded_chunks = sample_embedded_chunks()

    ids, documents, embeddings, metadatas = prepare_chroma_records(embedded_chunks)

    assert ids == ["page-1-chunk-1", "page-1-chunk-2"]
    assert documents[0] == "This is the first chunk."
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert metadatas[0]["page_number"] == 1
    assert metadatas[0]["chunk_id"] == 1


def test_store_embedded_chunks_saves_records(tmp_path):
    persist_directory = str(tmp_path / "test_chroma_db")
    embedded_chunks = sample_embedded_chunks()

    stored_count = store_embedded_chunks(
        embedded_chunks=embedded_chunks,
        persist_directory=persist_directory,
        collection_name="test_document_chunks",
    )

    collection_count = get_collection_count(
        persist_directory=persist_directory,
        collection_name="test_document_chunks",
    )

    assert stored_count == 2
    assert collection_count == 2