# Agentic AI Research Platform

A local-first PDF research assistant. It extracts and indexes PDF text, retrieves relevant passages, and answers questions with page-level citations using a locally running Ollama model.

## What it does

- Extracts text from PDFs and splits it into overlapping chunks while retaining source and page metadata.
- Embeds chunks with Sentence Transformers and stores them in a persistent local ChromaDB collection.
- Retrieves relevant passages, optionally prioritizing distinct pages.
- Generates concise answers from retrieved evidence with validated inline citations; abstains when no evidence is available.
- Provides a Streamlit chat interface for uploading, indexing, and asking questions about selected PDFs.
- Evaluates retrieval ranking using labeled PDF pages and Hit@k, Recall@k, and MRR.

The PDF, embedding model, vector database, and Ollama model run locally. Sentence Transformers may download its model from Hugging Face on first use unless a local model directory is provided. Scanned PDFs without an OCR text layer are not supported.

## Requirements

- Python 3.11 or newer
- Ollama for generated answers and chatbot use

## Setup

Create and activate a virtual environment, then install dependencies.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install Ollama from [ollama.com](https://ollama.com/), start it, and download the default chat model:

```bash
ollama pull llama3.2:3b
```

## Run the chatbot

From the project directory:

```bash
streamlit run streamlit_app.py
```

Upload one or more PDFs in the sidebar and select **Index selected PDFs**. Ask questions in the chat; retrieval is restricted to the selected PDFs, and answers include inline citations and expandable source-page details. Re-indexing a PDF from the interface replaces its existing chunks for that filename. Ollama must be running with the selected model available.

## Command-line usage

Inspect extraction and chunking without loading an embedding model:

```bash
python main.py inspect data/raw/sample_banking_genai_research.pdf
```

Index a PDF:

```bash
python main.py index path/to/document.pdf
```

Search indexed passages:

```bash
python main.py search "What should employees verify before relying on an answer?" --diversify-pages
```

Generate a cited answer with Ollama:

```bash
python main.py ask "What should employees verify before relying on an answer?"
```

The first indexing, search, or answer command downloads the default embedding model (`sentence-transformers/all-MiniLM-L6-v2`) if it is not already available. ChromaDB persists its index in `chroma_db/`. Use the same `--embedding-model` value for indexing and subsequent searches or answers. A local model directory can be passed with `--embedding-model path/to/model`.

The CLI index command upserts chunks; if a PDF changes, old chunks that are no longer present may remain in the index. The chatbot uploader replaces stale chunks when re-indexing the same filename.

## Evaluate retrieval

Index the sample PDF, then run the labeled retrieval evaluation:

```bash
python main.py index data/raw/sample_banking_genai_research.pdf
python main.py evaluate --k 3 --diversify-pages
```

The report includes Hit@k (whether any relevant page was retrieved), Recall@k (coverage of labeled source-page pairs), and MRR (rank of the first relevant page). The bundled evaluation set is a small synthetic wiring check, not a representative benchmark; it measures retrieval ranking, not answer faithfulness.

## Tests

```bash
python -m pytest -q
```

## Project layout

```text
app/
  frontend/   Streamlit chatbot and PDF upload ingestion
  rag/        PDF extraction, chunking, embeddings, retrieval, generation, evaluation
data/
  raw/        Source PDFs, including the synthetic sample
chroma_db/    Persistent local vector index (created at runtime)
main.py       CLI entry point
streamlit_app.py
tests/        Unit and integration tests
```

`data/raw/sample_banking_genai_research.pdf` is synthetic test data. Its organizations and examples are fictional and should not be treated as evidence about real banks or the banking industry. Regenerate it with:

```bash
python scripts/create_synthetic_sample_pdf.py
```
