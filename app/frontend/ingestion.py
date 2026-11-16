from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, TypedDict

from pypdf.errors import PdfReadError

from app.rag.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    TextChunk,
    chunk_documents,
)
from app.rag.document_loader import load_pdf


class UploadedPdfResult(TypedDict):
    source: str
    page_count: int
    chunk_count: int


class UploadedPdf(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


class SourceIndexer(Protocol):
    def replace_source(self, chunks: Sequence[TextChunk]) -> int: ...


def index_uploaded_pdf(
    uploaded: UploadedPdf,
    retriever: SourceIndexer,
    *,
    storage_directory: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> UploadedPdfResult:
    source = _validate_filename(uploaded.name)
    content = uploaded.getvalue()
    if not content:
        raise ValueError(f"{source} is empty.")

    with TemporaryDirectory() as temporary_directory:
        temporary_pdf = Path(temporary_directory) / source
        temporary_pdf.write_bytes(content)
        try:
            pages = load_pdf(temporary_pdf)
        except PdfReadError as error:
            raise ValueError(f"Could not read uploaded PDF {source}: {error}") from error

    chunks = chunk_documents(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"No extractable text found in {source}.")

    storage_path = Path(storage_directory)
    storage_path.mkdir(parents=True, exist_ok=True)
    (storage_path / source).write_bytes(content)

    indexed_count = retriever.replace_source(chunks)
    return {
        "source": source,
        "page_count": len(pages),
        "chunk_count": indexed_count,
    }


def _validate_filename(filename: str) -> str:
    if (
        not filename
        or Path(filename.replace("\\", "/")).name != filename
        or filename in {".", ".."}
    ):
        raise ValueError("Uploaded PDF must have a simple filename.")
    if Path(filename).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")
    return filename
