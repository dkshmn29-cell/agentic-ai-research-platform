from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from app.rag.document_loader import clean_text, load_pdf


def create_test_pdf(pdf_path: Path, page_texts: list[str]) -> None:
    pdf = canvas.Canvas(str(pdf_path))
    for page_text in page_texts:
        if page_text:
            pdf.drawString(72, 720, page_text)
        pdf.showPage()
    pdf.save()


def test_clean_text_normalizes_whitespace() -> None:
    assert clean_text("  Alpha   beta\t gamma\n\n\nDelta  ") == "Alpha beta gamma\n\nDelta"


def test_load_pdf_extracts_text_and_preserves_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "research.pdf"
    create_test_pdf(pdf_path, ["First page evidence.", "", "Evidence on page three."])

    documents = load_pdf(pdf_path)

    assert [document["metadata"]["page"] for document in documents] == [1, 3]
    assert [document["metadata"]["source"] for document in documents] == [
        "research.pdf",
        "research.pdf",
    ]
    assert documents[0]["content"] == "First page evidence."
    assert documents[1]["content"] == "Evidence on page three."


def test_load_pdf_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        load_pdf(tmp_path / "missing.pdf")


def test_load_pdf_rejects_non_pdf_file(tmp_path: Path) -> None:
    text_path = tmp_path / "notes.txt"
    text_path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Only PDF files are supported"):
        load_pdf(text_path)


def test_load_pdf_rejects_encrypted_pdf(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    pdf_path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt(user_password="test-password")
    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    with pytest.raises(ValueError, match="Encrypted PDFs are not supported"):
        load_pdf(pdf_path)