# Agentic AI Research & Intelligence Platform

A local-first research assistant for turning source documents into evidence-traceable reports.

`data/raw/sample_banking_genai_research.pdf` is synthetic test data. Its organizations and examples are fictional; it is not evidence about real banks or the banking industry. Regenerate it with `python scripts/create_synthetic_sample_pdf.py`.

## Setup

Use Python 3.11 or newer.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Verify

```bash
python -m pytest -q
```

## Inspect a PDF

Inspect extraction and chunks without loading an embedding model:

```powershell
python main.py inspect data/raw/sample_banking_genai_research.pdf --chunk-size 250 --chunk-overlap 50
```

## Index and search

Index the synthetic fixture locally:

```powershell
python main.py index data/raw/sample_banking_genai_research.pdf
```

Search the indexed chunks and display the source page for each result:

```powershell
python main.py search "What should employees verify before relying on an answer?"
```

The first `index` or `search` command downloads the configured embedding model (`sentence-transformers/all-MiniLM-L6-v2`) from Hugging Face unless it is already cached. It runs locally on CPU after download. The vector database is stored in `chroma_db/`, which is ignored by Git.

### Transfer the model from a connected computer

On a computer that can access Hugging Face, from the project directory, download the model files:

```powershell
python -m pip install huggingface_hub
python scripts/download_embedding_model.py
```

Copy the resulting `models/all-MiniLM-L6-v2/` directory to the same relative path in this project on the offline computer. The `models/` directory is intentionally ignored by Git because model weights are large. Then index and search using that local path:

```powershell
$env:HF_HUB_OFFLINE = "1"
python main.py index data/raw/sample_banking_genai_research.pdf --embedding-model models/all-MiniLM-L6-v2
python main.py search "What should employees verify before relying on an answer?" --embedding-model models/all-MiniLM-L6-v2
```

The model files are read locally; the offline environment does not need Hugging Face access.

`--chunk-size`, `--chunk-overlap`, `--persist-directory`, `--collection-name`, and `--embedding-model` can be used to adjust the pipeline. Other raw and processed data remain excluded from Git by default.

## Dependencies

- PyPDF extracts text from PDFs.
- LangChain text splitters provide recursive, character-based splitting.
- Sentence Transformers creates local text embeddings.
- ChromaDB stores embeddings and source/page metadata persistently.
- pytest runs the tests.
- ReportLab generates the synthetic sample and deterministic test PDFs.