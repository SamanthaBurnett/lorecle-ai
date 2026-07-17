from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer

from answer_question import generate_answer, print_answer_with_sources
from retrieve_chunks import load_chunks, retrieve_top_chunks


def main() -> None:
    """
    Run LorecleAI as a simple command-line interface.

    Flow:
    1. Load chunked source data
    2. Load the retrieval model once
    3. Repeatedly prompt the user for a question
    4. Retrieve the most relevant chunks
    5. Generate a grounded answer
    6. Print the answer and supporting sources

    The loop continues until the user types 'exit' or 'quit'.
    """

    chunks_path = Path("data/processed/chunks.json")

    chunks = load_chunks(chunks_path)

    print(f"Loaded {len(chunks)} chunks.")

    print("Loading retrieval model...")

    # Load the embedding model once so we do not reinitialize it on every question.
    retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("LorecleAI is ready.")

    print("Ask a question about the source material.")

    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Question: ").strip()

        # Let the user leave the CLI cleanly.
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        # Prevent empty submissions from running the pipeline.
        if not question:
            print("Please enter a question.\n")
            continue

        try:
            results = retrieve_top_chunks(
                query=question,
                chunks=chunks,
                model=retrieval_model,
                top_k=3,
            )

            answer = generate_answer(question, results)

            print_answer_with_sources(question, answer, results)

            print()

        except Exception as error:
            # Keep the CLI alive even if one question fails.
            print(f"\nSomething went wrong: {error}\n")


if __name__ == "__main__":
    main()
