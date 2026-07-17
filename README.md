# LorecleAI

## Project overview

LorecleAI is a small personal RAG prototype that answers questions using my own career materials rather than relying only on a model’s general knowledge.

The first version is focused on a small set of local markdown source files, such as a redacted resume, selected project stories, and a few performance review summaries.

## Why I built this

I built this project because I kept seeing RAG mentioned and wanted to understand what it actually is by building a small grounded prototype myself.

## How it works

At a high level, the project will:

1. Load source documents from `data/raw/`
2. Split them into smaller chunks
3. Create embeddings for those chunks
4. Retrieve the most relevant chunks for a question
5. Generate an answer using the retrieved context

The goal is to make answers more grounded in the source material and easier to inspect than a plain prompt-only approach.

## Tech stack

- Python
- Local markdown source files
- Additional libraries will be added as the retrieval pipeline is implemented

## Run locally

1. Clone the repo
2. Create and activate a virtual environment
3. Install dependencies from `requirements.txt`
4. Add your own source documents to `data/raw/`
5. Run the project locally

Example setup on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\main.py
```
