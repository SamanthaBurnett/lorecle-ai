from __future__ import annotations

from pathlib import Path
from typing import List, TypedDict


class Document(TypedDict):
    """
    Represents one source document loaded from data/raw.

    Fields:
        source: The filename the text came from.
        text: The full cleaned text of the document.
    """

    source: str
    text: str


def normalize_text(text: str) -> str:
    """
    Perform light text cleanup while preserving useful structure.

    Why this exists:
    We want to remove obvious formatting noise without destroying headings,
    bullets, or paragraph boundaries that may help later during chunking.

    What this does:
    - Strips leading/trailing whitespace
    - Normalizes Windows line endings to Unix-style newlines
    - Replaces runs of 3+ blank lines with 2 blank lines

    We keep this intentionally conservative for now.
    """
    # Normalize line endings so text behaves consistently across environments.
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Reduce excessive blank lines without flattening the document too much.
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")

    return cleaned


def load_markdown_documents(raw_data_dir: Path) -> List[Document]:
    """
    Load all markdown documents from the given directory.

    Args:
        raw_data_dir: Path to the folder containing raw markdown files.

    Returns:
        A list of dictionaries containing source filename and cleaned text.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no markdown files are found.
    """
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    # Find all markdown files in a stable sorted order so runs are reproducible.
    markdown_files = sorted(raw_data_dir.glob("*.md"))

    if not markdown_files:
        raise ValueError(f"No markdown files found in: {raw_data_dir}")

    documents: List[Document] = []

    for file_path in markdown_files:
        # Read the file as UTF-8 text.
        raw_text = file_path.read_text(encoding="utf-8")

        # Apply light cleanup before storing the document.
        cleaned_text = normalize_text(raw_text)

        # Skip empty files so they do not create noise later.
        if not cleaned_text:
            continue

        documents.append(
            {
                "source": file_path.name,
                "text": cleaned_text,
            }
        )

    return documents


def main() -> None:
    """
    Small local test runner for document loading.

    This lets us verify:
    - files are being found
    - text is loading correctly
    - documents are not empty
    """
    raw_data_dir = Path("data/raw")

    documents = load_markdown_documents(raw_data_dir)

    print(f"Loaded {len(documents)} document(s).\n")

    for document in documents:
        print(f"Source: {document['source']}")
        print(f"Character count: {len(document['text'])}")
        print("-" * 50)


if __name__ == "__main__":
    main()
