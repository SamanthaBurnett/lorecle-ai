from __future__ import annotations

import json
from pathlib import Path
from typing import List, TypedDict

from load_documents import Document, load_markdown_documents


class Chunk(TypedDict):
    """
    Represents one chunk derived from a source document.

    Fields:
        chunk_id: Unique identifier for the chunk.
        source: Original filename the chunk came from.
        start_char: Inclusive starting character index in the source text.
        end_char: Exclusive ending character index in the source text.
        text: The chunk text itself.
    """

    chunk_id: str
    source: str
    start_char: int
    end_char: int
    text: str


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[Chunk]:
    """
    Split one document into overlapping character-based chunks.

    Why this exists:
    RAG systems usually retrieve better from smaller focused pieces of text
    rather than one large full-document blob.

    Args:
        text: Full source text to chunk.
        source: Source filename for metadata.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to repeat between neighboring chunks.

    Returns:
        A list of chunk dictionaries.

    Raises:
        ValueError: If chunk_size is invalid relative to overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []

    start = 0

    chunk_index = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk_text_value = text[start:end].strip()

        if chunk_text_value:
            chunks.append(
                {
                    "chunk_id": f"{source}_chunk_{chunk_index}",
                    "source": source,
                    "start_char": start,
                    "end_char": end,
                    "text": chunk_text_value,
                }
            )

            chunk_index += 1

        # Move forward, but keep overlap so adjacent chunks share some context.
        start += chunk_size - overlap

    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[Chunk]:
    """
    Chunk a list of loaded documents into retrievable pieces.

    Args:
        documents: Loaded source documents.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        A flattened list of all chunks across all documents.
    """
    all_chunks: List[Chunk] = []

    for document in documents:

        document_chunks = chunk_text(
            text=document["text"],
            source=document["source"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        all_chunks.extend(document_chunks)

    return all_chunks


def save_chunks(chunks: List[Chunk], output_path: Path) -> None:
    """
    Save chunks to a JSON file.

    Args:
        chunks: Chunk list to save.
        output_path: Destination JSON path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)


def print_chunk_samples(chunks: List[Chunk], sample_count: int = 3) -> None:
    """
    Print a few sample chunks for manual inspection.

    Args:
        chunks: All generated chunks.
        sample_count: Number of chunks to preview.
    """
    print(f"\nCreated {len(chunks)} chunk(s).\n")

    for chunk in chunks[:sample_count]:
        preview = chunk["text"][:200].replace("\n", " ")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Range: {chunk['start_char']} - {chunk['end_char']}")
        print(f"Preview: {preview}")
        print("-" * 60)


def main() -> None:
    """
    Load markdown documents, chunk them, and save the output locally.
    """
    raw_data_dir = Path("data/raw")

    output_path = Path("data/processed/chunks.json")

    documents = load_markdown_documents(raw_data_dir)

    print(f"Loaded {len(documents)} document(s).")

    chunks = chunk_documents(documents, chunk_size=800, overlap=120)

    save_chunks(chunks, output_path)

    print(f"Saved chunks to: {output_path}")

    print_chunk_samples(chunks)


if __name__ == "__main__":
    main()
