# LorecleAI

## Project overview

LorecleAI is a small personal RAG prototype that answers questions using my own career materials rather than relying only on a model’s general knowledge.

The current version is designed around personal career materials such as a redacted resume, project stories, and selected review notes. It loads local source documents, chunks them into smaller retrievable units, retrieves the most relevant chunks for a question, and then uses a generation model to produce a grounded answer.

## Why I built this

I built this project because I was curious about what a RAG was. I also wanted to understand what use cases RAG supports.

## How it works

LorecleAI currently supports a simple end-to-end RAG flow:

1. Load markdown source documents from `data/raw/`
2. Normalize and chunk those documents into smaller overlapping pieces
3. Create local embeddings for semantic retrieval
4. Retrieve the most relevant chunks for a question
5. Use a generation model to answer from the retrieved context
6. Display the final answer along with supporting source chunks

## Tech stack

- Python
- Local markdown source files
- `sentence-transformers` for local embeddings and semantic retrieval
- `openai` for the current generation implementation
- `python-dotenv` for environment variable loading

## Run locally

1. Clone the repo
2. Create and activate a virtual environment
3. Install dependencies from `requirements.txt`
4. Create a .env file with your generation model configuration
5. Add your own source documents to data/raw/
6. Generate chunked source data
7. Start LorecleAI from the CLI

Example setup on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\chunk_documents.py
python .\src\main.py
```

## Usage notes

When the CLI starts, you can type a question about the source material and LorecleAI will:

- retrieve the most relevant chunks
- generate an answer from those chunks
- show the supporting source snippets

Type exit or quit to leave the CLI.

## Current scope and limitations

- This is a local learning prototype, not a production system
- Source documents are expected to be provided locally by the user
- Raw and processed source data are intentionally gitignored
- Retrieval is local, but the current answer-generation implementation uses a user-provided OpenAI API key
- The generation layer can be adapted later to another API provider or a local model
- The first version is intentionally small so retrieval quality is easier to inspect and debug

##Future improvements

- Persist embeddings locally so chunks do not need to be re-embedded on each run
- Add a small web UI
- Support more configurable chunking strategies
- Add provider abstraction for easier swapping between OpenAI, local Llama-based models, or other APIs
- Improve source citation formatting in answers
