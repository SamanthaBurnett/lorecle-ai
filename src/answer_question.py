from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from retrieve_chunks import RetrievedChunk, load_chunks, retrieve_top_chunks
from sentence_transformers import SentenceTransformer


def build_context_block(results: List[RetrievedChunk]) -> str:
    """
    Build a readable context block from retrieved chunks.

    Why this exists:
    The generation model should answer from retrieved source material,
    not from vague general knowledge. This function formats that source
    material into a single prompt section.
    """
    context_parts: List[str] = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            (
                f"[Source {index}]\n"
                f"Chunk ID: {result['chunk_id']}\n"
                f"File: {result['source']}\n"
                f"Character Range: {result['start_char']} - {result['end_char']}\n"
                f"Content:\n{result['text']}\n"
            )
        )

    return "\n\n".join(context_parts)


def build_prompt(question: str, context_block: str) -> str:
    """
    Build the grounded prompt for answer generation.

    Prompt rules:
    - Answer only from the provided context
    - Do not invent missing facts
    - Be honest when context is insufficient
    - Reference source numbers when useful
    """
    return f"""
You are helping answer questions about a person's experience using only the provided source context.

Instructions:
- Answer the question using only the provided context.
- Do not invent or assume facts that are not present.
- If the context is insufficient, say so plainly.
- Prefer a concise, grounded answer.
- When helpful, mention which source(s) support the answer.

Question:
{question}

Context:
{context_block}

""".strip()


def generate_answer(question: str, results: List[RetrievedChunk]) -> str:
    """
    Generate a grounded answer from retrieved chunks using the OpenAI API.
    """

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    model_name = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env")

    if not model_name:
        raise ValueError("OPENAI_MODEL is missing from .env")

    client = OpenAI(api_key=api_key)

    context_block = build_context_block(results)

    prompt = build_prompt(question, context_block)

    response = client.responses.create(
        model=model_name,
        input=prompt,
    )

    return response.output_text.strip()


def print_answer_with_sources(
    question: str,
    answer: str,
    results: List[RetrievedChunk],
) -> None:
    """
    Print the final answer followed by the supporting retrieved chunks.
    """

    print("\n" + "=" * 80)

    print(f"Question: {question}")

    print("=" * 80)

    print("\nAnswer:\n")

    print(answer)

    print("\nSupporting sources:\n")

    for index, result in enumerate(results, start=1):
        preview = result["text"][:220].replace("\n", " ")

        print(f"[Source {index}] {result['source']} | Score: {result['score']:.4f}")

        print(f"Chunk ID: {result['chunk_id']}")

        print(f"Preview: {preview}")

        print("-" * 80)


def main() -> None:
    """
    End-to-end local runner:
    1. load chunks
    2. retrieve top matches
    3. generate grounded answer
    4. print answer + sources
    """

    chunks_path = Path("data/processed/chunks.json")

    chunks = load_chunks(chunks_path)

    # Local retrieval model from the previous ticket.
    retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Some test questions
    # question = "What examples show observability work?"
    # question = "What projects suggest distributed systems work?"
    question = "Which stories show debugging and ownership?"

    results = retrieve_top_chunks(
        query=question,
        chunks=chunks,
        model=retrieval_model,
        top_k=3,
    )

    answer = generate_answer(question, results)

    print_answer_with_sources(question, answer, results)


if __name__ == "__main__":

    main()
