from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from app.frontend.ingestion import index_uploaded_pdf
from app.rag.chunking import TextChunk


class FakeUploadedPdf:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


class FakeRetriever:
    def __init__(self) -> None:
        self.replaced_chunks: list[TextChunk] = []

    def replace_source(self, chunks: Sequence[TextChunk]) -> int:
        self.replaced_chunks = list(chunks)
        return len(chunks)


def create_pdf() -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "Employees verify the cited source version and date.")
    pdf.save()
    return buffer.getvalue()


def test_index_uploaded_pdf_persists_and_indexes_extracted_pages(
    tmp_path: Path,
) -> None:
    uploaded = FakeUploadedPdf("research.pdf", create_pdf())
    retriever = FakeRetriever()

    result = index_uploaded_pdf(
        uploaded,
        retriever,
        storage_directory=tmp_path,
    )

    assert result["source"] == "research.pdf"
    assert result["page_count"] == 1
    assert result["chunk_count"] == 1
    assert (tmp_path / "research.pdf").read_bytes() == uploaded.getvalue()
    assert retriever.replaced_chunks[0]["metadata"]["source"] == "research.pdf"


@pytest.mark.parametrize("filename", ["../outside.pdf", "folder\\outside.pdf", "notes.txt"])
def test_index_uploaded_pdf_rejects_unsafe_or_unsupported_filenames(
    filename: str,
    tmp_path: Path,
) -> None:
    retriever = FakeRetriever()

    with pytest.raises(ValueError):
        index_uploaded_pdf(
            FakeUploadedPdf(filename, create_pdf()),
            retriever,
            storage_directory=tmp_path,
        )

    assert not list(tmp_path.iterdir())


def test_index_uploaded_pdf_rejects_invalid_pdf_without_saving_it(
    tmp_path: Path,
) -> None:
    retriever = FakeRetriever()

    with pytest.raises(ValueError, match="Could not read uploaded PDF"):
        index_uploaded_pdf(
            FakeUploadedPdf("invalid.pdf", b"not a pdf"),
            retriever,
            storage_directory=tmp_path,
        )

    assert not list(tmp_path.iterdir())
