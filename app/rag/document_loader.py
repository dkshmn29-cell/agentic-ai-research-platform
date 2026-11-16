import re
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader


class DocumentMetadata(TypedDict):
    source: str
    page: int


class PageDocument(TypedDict):
    content: str
    metadata: DocumentMetadata


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(pdf_path: str | Path) -> list[PageDocument]:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a PDF file, got a non-file path: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    documents: list[PageDocument] = []
    for page_number, page in enumerate(reader.pages, start=1):
        content = clean_text(page.extract_text() or "")
        if not content:
            continue

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": path.name,
                    "page": page_number,
                },
            }
        )

    return documents