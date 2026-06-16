import pytest

from src.embeddings import create_embedding, embed_chunks


class FakeEmbeddingData:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, embedding):
        self.data = [FakeEmbeddingData(embedding)]


class FakeEmbeddingsClient:
    def create(self, model, input):
        return FakeEmbeddingResponse([0.1, 0.2, 0.3])


class FakeOpenAIClient:
    def __init__(self):
        self.embeddings = FakeEmbeddingsClient()


def test_create_embedding_returns_vector():
    fake_client = FakeOpenAIClient()

    embedding = create_embedding(
        text="This is a test chunk.",
        client=fake_client,
        model="fake-embedding-model",
    )

    assert embedding == [0.1, 0.2, 0.3]


def test_create_embedding_rejects_empty_text():
    fake_client = FakeOpenAIClient()

    with pytest.raises(ValueError):
        create_embedding(
            text="   ",
            client=fake_client,
            model="fake-embedding-model",
        )


def test_embed_chunks_adds_embedding_to_each_chunk():
    fake_client = FakeOpenAIClient()

    chunks = [
        {
            "chunk_id": 1,
            "page_number": 1,
            "text": "First chunk text.",
            "character_count": 17,
        },
        {
            "chunk_id": 2,
            "page_number": 1,
            "text": "Second chunk text.",
            "character_count": 18,
        },
    ]

    embedded_chunks = embed_chunks(
        chunks=chunks,
        client=fake_client,
        model="fake-embedding-model",
    )

    assert len(embedded_chunks) == 2
    assert embedded_chunks[0]["embedding"] == [0.1, 0.2, 0.3]
    assert embedded_chunks[1]["embedding"] == [0.1, 0.2, 0.3]
    assert embedded_chunks[0]["embedding_model"] == "fake-embedding-model"