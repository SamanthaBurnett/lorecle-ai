## Key learnings

- RAG depends on having usable source material, so document quality matters before any model or retrieval logic is added.

- For a first prototype, manually cleaned text or markdown files are a better choice than overcomplicating the setup with document parsing.

- Keeping the first document set small makes it easier to debug retrieval quality later.

- Preparing source documents for RAG may require curation and redaction, not just format conversion, because sensitive or irrelevant content can hurt both safety and retrieval quality.

- Chunking is a core part of RAG, not just preprocessing, because retrieval depends on whether the source material is broken into useful, focused pieces.
  Smaller chunks improve retrieval precision, but if they are too small they can lose important context.

- Overlap helps preserve continuity between chunks so ideas are less likely to be split too abruptly across chunk boundaries.

- For a small local prototype, it can make sense to delay persisting embeddings until retrieval works end to end, because that keeps the system easier to understand and debug.

- The “generation” part of RAG should still be tightly grounded in retrieved context; retrieval alone is not enough if the prompt allows the model to drift beyond the sources.

- Showing supporting chunks alongside the final answer makes it easier to debug whether a weak result came from retrieval or from generation.

- RAG makes sense when the goal is to answer from document-based knowledge, while a skill or tool makes more sense when the model needs to take an action or fetch live structured data.

## RAG vs other approaches

- RAG makes sense when the model needs grounded answers from source documents rather than relying on general knowledge.

- RAG is best for unstructured text like resumes, project writeups, brag docs, and review summaries.

- A tool or skill makes more sense when the model needs to take an action or fetch structured/live data, not just retrieve context.

- MCP is not a replacement for RAG; it is more about exposing tools and resources in a standard reusable way.

- This project is a good fit for RAG because the main problem is retrieving and using document-based career context.

## Python learnings

- Python is a practical choice for this prototype because it reduces friction around AI experimentation and local iteration.

- The setup process helped reinforce that Python virtual environments are useful for isolating project dependencies from the rest of the machine.
