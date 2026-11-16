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

Add `--diversify-pages` to prefer one result per source page before returning additional chunks from the same page:

```powershell
python main.py search "What should employees verify before relying on an answer?" --diversify-pages
```

The first `index` or `search` command downloads the configured embedding model (`sentence-transformers/all-MiniLM-L6-v2`) from Hugging Face unless it is already cached. It runs locally on CPU after download. The vector database is stored in `chroma_db/`, which is ignored by Git. Re-indexing uses stable chunk IDs and updates matching records; it does not yet remove stale chunks if a source PDF changes. Scanned PDFs are not supported yet because they require OCR.

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

## Ask questions with cited answers

Install and start [Ollama](https://ollama.com/), then download the default local chat model:

```bash
ollama pull llama3.2:3b
```

Ask a question against the indexed evidence:

```bash
python main.py ask "What should employees verify before relying on an answer?" \
  --embedding-model models/all-MiniLM-L6-v2
```

The command retrieves page-diversified evidence, asks Ollama to answer only from those passages, and prints source-page citations. It rejects unknown or missing citations and returns an explicit insufficient-evidence response when no chunks are indexed. To use another Ollama model or API host, pass `--ollama-model` or `--ollama-url`. The embedding model and the Ollama chat model are separate.

## Evaluate retrieval

After indexing the sample PDF, evaluate the top three results against the labeled pages:

```powershell
python main.py evaluate --embedding-model models/all-MiniLM-L6-v2 --k 3
```

Add `--diversify-pages` to evaluate page-diversified retrieval:

```powershell
python main.py evaluate --embedding-model models/all-MiniLM-L6-v2 --k 3 --diversify-pages
```

`Hit@k` is the share of questions with at least one labeled relevant page in the top `k`. `Recall@k` is the average share of each question's unique labeled source-page pairs found in the top `k`, so repeated chunks from one page do not inflate coverage. The per-case output includes page recall. `MRR` (mean reciprocal rank) rewards relevant pages appearing earlier. Page diversification is opt-in and fetches additional candidates as needed. This small, synthetic dataset is a wiring check, not a representative benchmark. It evaluates page ranking only; it does not measure answer faithfulness or whether the system abstains when evidence is absent.

`--chunk-size`, `--chunk-overlap`, `--persist-directory`, `--collection-name`, and `--embedding-model` can be used to adjust the pipeline. Other raw and processed data remain excluded from Git by default.

## Dependencies

- PyPDF extracts text from PDFs.
- LangChain text splitters provide recursive, character-based splitting.
- Sentence Transformers creates local text embeddings.
- ChromaDB stores embeddings and source/page metadata persistently.
- pytest runs the tests.
- ReportLab generates the synthetic sample and deterministic test PDFs.