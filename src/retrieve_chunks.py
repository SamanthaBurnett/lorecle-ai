from __future__ import annotations


import json
from pathlib import Path
from typing import List, TypedDict

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class Chunk(TypedDict):
    """
    Represents one chunk loaded from the processed chunks JSON file.
    """

    chunk_id: str
    source: str
    start_char: int
    end_char: int
    text: str


class RetrievedChunk(TypedDict):
    """
    Represents one retrieved chunk along with its similarity score.
    """

    chunk_id: str
    source: str
    start_char: int
    end_char: int
    text: str
    score: float


def load_chunks(chunks_path: Path) -> List[Chunk]:
    """
    Load chunk data from a JSON file.

    Args:
        chunks_path: Path to the chunks.json file.

    Returns:
        A list of chunk dictionaries.

    Raises:
        FileNotFoundError: If the chunks file does not exist.
        ValueError: If the file is empty.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks: List[Chunk] = json.load(file)

    if not chunks:
        raise ValueError(f"No chunks found in: {chunks_path}")

    return chunks


def embed_texts(model: SentenceTransformer, texts: List[str]):
    """
    Convert a list of texts into embeddings.

    Why this exists:
    Embeddings let us compare text semantically, which means by meaning
    rather than exact keyword overlap.

    Args:
        model: The local embedding model.
        texts: The texts to embed.

    Returns:
        A tensor-like structure of embeddings.
    """
    return model.encode(texts, convert_to_tensor=True)


def retrieve_top_chunks(
    query: str,
    chunks: List[Chunk],
    model: SentenceTransformer,
    top_k: int = 3,
) -> List[RetrievedChunk]:
    """
    Retrieve the most semantically relevant chunks for a query.

    Args:
        query: The user question.
        chunks: Available source chunks.
        model: Local embedding model.
        top_k: Number of top matches to return.

    Returns:
        A list of retrieved chunks sorted by descending similarity.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    chunk_texts = [chunk["text"] for chunk in chunks]

    # Create embeddings for all stored chunks.
    chunk_embeddings = embed_texts(model, chunk_texts)

    # Create an embedding for the user's question.
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compare the query embedding against every chunk embedding.
    similarity_scores = cos_sim(query_embedding, chunk_embeddings)[0]

    # Get the indices of the top-k highest scoring chunks.
    top_results = similarity_scores.topk(k=min(top_k, len(chunks)))

    retrieved_chunks: List[RetrievedChunk] = []

    for score, index in zip(top_results.values, top_results.indices):
        chunk = chunks[int(index)]

        retrieved_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "text": chunk["text"],
                "score": float(score),
            }
        )

    return retrieved_chunks


def print_retrieved_chunks(query: str, results: List[RetrievedChunk]) -> None:
    """
    Print retrieval results in a readable format for manual inspection.
    """

    print(f"\nQuery: {query}")

    print("=" * 80)

    for result in results:
        preview = result["text"][:250].replace("\n", " ")

        print(f"Chunk ID: {result['chunk_id']}")

        print(f"Source: {result['source']}")

        print(f"Range: {result['start_char']} - {result['end_char']}")

        print(f"Score: {result['score']:.4f}")

        print(f"Preview: {preview}")

        print("-" * 80)


def main() -> None:
    """
    Local test runner for semantic retrieval.
    """

    chunks_path = Path("data/processed/chunks.json")

    chunks = load_chunks(chunks_path)

    print(f"Loaded {len(chunks)} chunks.")

    # A strong local starter model for semantic text similarity.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    test_queries = [
        "What examples show observability work?",
        "What experience supports backend platform roles?",
        "Which stories show debugging and ownership?",
    ]

    for query in test_queries:
        results = retrieve_top_chunks(query=query, chunks=chunks, model=model, top_k=3)
        print_retrieved_chunks(query, results)


if __name__ == "__main__":
    main()
