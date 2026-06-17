import pytest

from src.qa_chain import build_rag_prompt, generate_answer


class FakeResponse:
    def __init__(self, output_text):
        self.output_text = output_text


class FakeResponsesClient:
    def create(self, model, input):
        return FakeResponse(
            "RAG uses retrieved document chunks to answer questions."
        )


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponsesClient()


def sample_search_results():
    return [
        {
            "id": "page-1-chunk-1",
            "text": "RAG retrieves relevant chunks before generating an answer.",
            "metadata": {"page_number": 1, "chunk_id": 1},
            "distance": 0.1,
            "page_number": 1,
            "chunk_id": 1,
        }
    ]


def test_build_rag_prompt_adds_context_and_question():
    prompt_template = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    prompt = build_rag_prompt(
        question="What does RAG do?",
        search_results=sample_search_results(),
        prompt_template=prompt_template,
    )

    assert "RAG retrieves relevant chunks" in prompt
    assert "What does RAG do?" in prompt
    assert "[Page 1, Chunk 1]" in prompt


def test_generate_answer_returns_answer_and_sources():
    fake_client = FakeOpenAIClient()

    result = generate_answer(
        question="What does RAG do?",
        search_results=sample_search_results(),
        client=fake_client,
        model="fake-answer-model",
        prompt_template="Context:\n{context}\nQuestion:\n{question}\nAnswer:",
    )

    assert result["answer"] == "RAG uses retrieved document chunks to answer questions."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "Page 1, Chunk 1"


def test_generate_answer_rejects_empty_question():
    fake_client = FakeOpenAIClient()

    with pytest.raises(ValueError):
        generate_answer(
            question="   ",
            search_results=sample_search_results(),
            client=fake_client,
            model="fake-answer-model",
            prompt_template="Context:\n{context}\nQuestion:\n{question}\nAnswer:",
        )


def test_generate_answer_handles_empty_search_results():
    fake_client = FakeOpenAIClient()

    result = generate_answer(
        question="What does the document say?",
        search_results=[],
        client=fake_client,
        model="fake-answer-model",
        prompt_template="Context:\n{context}\nQuestion:\n{question}\nAnswer:",
    )

    assert result["answer"] == "I do not know based on the provided document."
    assert result["sources"] == []