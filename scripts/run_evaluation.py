"""
Run an end-to-end RAG evaluation from the command line.

Example:

python scripts/run_evaluation.py --pdf "sample_docs/Build With AI Playbook.pdf"
"""

import argparse
import shutil
import sys
from io import BytesIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from dotenv import load_dotenv

from src.chunking import chunk_document
from src.embeddings import create_embedding, embed_chunks
from src.evaluation import (
    evaluate_answer_result,
    load_evaluation_cases,
)
from src.ingestion import extract_text_from_pdf
from src.qa_chain import generate_answer
from src.retriever import (
    search_similar_chunks,
    store_embedded_chunks,
)


load_dotenv()


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the Smart Document Q&A RAG pipeline."
    )

    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the evaluation PDF.",
    )

    parser.add_argument(
        "--cases",
        default="evals/playbook_cases.json",
        help="Path to evaluation cases JSON.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks retrieved per question.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the complete evaluation pipeline.
    """
    args = parse_arguments()

    pdf_path = Path(args.pdf)
    cases_path = Path(args.cases)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if not cases_path.exists():
        raise FileNotFoundError(
            f"Evaluation cases not found: {cases_path}"
        )

    cases = load_evaluation_cases(
        str(cases_path)
    )

    if not cases:
        raise ValueError(
            "No evaluation cases were loaded. "
            "Check evals/playbook_cases.json."
        )

    print(
        f"Loaded {len(cases)} evaluation cases."
    )

    pdf_bytes = pdf_path.read_bytes()

    extracted_pages = extract_text_from_pdf(
        BytesIO(pdf_bytes)
    )

    chunks = chunk_document(
        pages=extracted_pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(
        f"\nExtracted {len(extracted_pages)} pages "
        f"and created {len(chunks)} chunks."
    )

    print(
        "Generating local document embeddings..."
    )

    embedded_chunks = embed_chunks(
        chunks
    )

    evaluation_directory = (
        PROJECT_ROOT / "evaluation_chroma_db"
    )

    collection_name = "rag_evaluation"

    if evaluation_directory.exists():
        try:
            shutil.rmtree(
                evaluation_directory
            )
        except PermissionError:
            print(
                "\nCould not remove the previous evaluation "
                "database because Windows is still using it."
            )

            print(
                "Using the existing folder and resetting "
                "the ChromaDB collection instead."
            )

    evaluation_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    stored_count = store_embedded_chunks(
        embedded_chunks=embedded_chunks,
        persist_directory=str(
            evaluation_directory
        ),
        collection_name=collection_name,
        reset=True,
    )

    print(
        f"Stored {stored_count} evaluation chunks."
    )

    passed_count = 0
    results = []

    for case_number, case in enumerate(
        cases,
        start=1,
    ):
        question = case["question"]

        print("\n" + "=" * 70)

        print(
            f"Running case {case_number}/{len(cases)}: "
            f"{case['id']}"
        )

        question_embedding = create_embedding(
            text=question,
            task="query",
        )

        search_results = search_similar_chunks(
            query_embedding=question_embedding,
            persist_directory=str(
                evaluation_directory
            ),
            collection_name=collection_name,
            top_k=args.top_k,
        )

        answer_result = generate_answer(
            question=question,
            search_results=search_results,
        )

        evaluation = evaluate_answer_result(
            answer_result=answer_result,
            expected_behavior=case[
                "expected_behavior"
            ],
            expected_pages=case.get(
                "expected_pages"
            ),
        )

        evaluation["id"] = case["id"]
        evaluation["question"] = question

        results.append(
            evaluation
        )

        if evaluation["passed"]:
            passed_count += 1

        status = (
            "PASS"
            if evaluation["passed"]
            else "REVIEW"
        )

        print(
            f"{status}: {case['id']}"
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Expected: "
            f"{case['expected_behavior']}"
        )

        print(
            f"Answer: "
            f"{evaluation['answer']}"
        )

        print(
            f"Source pages: "
            f"{evaluation['source_pages']}"
        )

        print(
            f"Checks: "
            f"{evaluation['checks']}"
        )

    total_cases = len(results)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Passed: {passed_count}/{total_cases}"
    )

    print(
        f"Needs review: "
        f"{total_cases - passed_count}/{total_cases}"
    )

    if passed_count != total_cases:
        print(
            "\nSome cases need manual review. "
            "This is normal during RAG evaluation."
        )


if __name__ == "__main__":
    main()