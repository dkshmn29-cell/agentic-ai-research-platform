from pathlib import Path

import pytest

from app.rag.chunking import chunk_documents
from app.rag.document_loader import load_pdf


def test_chunk_documents_preserves_page_metadata_and_stable_ids() -> None:
    documents = [
        {
            "content": "Customer service drafts responses for staff. " * 12,
            "metadata": {"source": "research.pdf", "page": 1},
        },
        {
            "content": "Operations staff verify current procedures. " * 12,
            "metadata": {"source": "research.pdf", "page": 3},
        },
    ]

    chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 2
    assert {chunk["metadata"]["page"] for chunk in chunks} == {1, 3}
    assert all(chunk["metadata"]["source"] == "research.pdf" for chunk in chunks)
    assert all(chunk["metadata"]["chunk_id"] for chunk in chunks)
    assert chunks == chunk_documents(documents, chunk_size=100, chunk_overlap=20)

    for page_number in (1, 3):
        page_chunks = [
            chunk for chunk in chunks if chunk["metadata"]["page"] == page_number
        ]
        assert [chunk["metadata"]["chunk_index"] for chunk in page_chunks] == list(
            range(1, len(page_chunks) + 1)
        )


def test_chunk_documents_keeps_overlap_between_neighboring_chunks() -> None:
    content = " ".join(f"term{index:02d}" for index in range(40))
    document = {
        "content": content,
        "metadata": {"source": "overlap.pdf", "page": 1},
    }

    chunks = chunk_documents([document], chunk_size=60, chunk_overlap=15)

    assert len(chunks) > 1
    assert any(
        set(left["content"].split()) & set(right["content"].split())
        for left, right in zip(chunks, chunks[1:])
    )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10)],
)
def test_chunk_documents_rejects_invalid_settings(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_synthetic_pdf_runs_through_ingestion_and_chunking() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pdf_path = project_root / "data" / "raw" / "sample_banking_genai_research.pdf"

    pages = load_pdf(pdf_path)
    chunks = chunk_documents(pages, chunk_size=250, chunk_overlap=50)

    assert len(pages) == 3
    assert {page["metadata"]["page"] for page in pages} == {1, 2, 3}
    assert len(chunks) > len(pages)
    assert {chunk["metadata"]["page"] for chunk in chunks} == {1, 2, 3}
    assert all(
        chunk["metadata"]["source"] == pdf_path.name
        for chunk in chunks
    )