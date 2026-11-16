import hashlib
from typing import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.document_loader import PageDocument

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


class ChunkMetadata(TypedDict):
    source: str
    page: int
    chunk_id: str
    chunk_index: int


class TextChunk(TypedDict):
    content: str
    metadata: ChunkMetadata


def chunk_documents(
    documents: list[PageDocument],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[TextChunk] = []

    for document in documents:
        source = document["metadata"]["source"]
        page = document["metadata"]["page"]
        page_chunks = splitter.split_text(document["content"])

        for chunk_index, content in enumerate(page_chunks, start=1):
            identity = "\0".join((source, str(page), str(chunk_index), content))
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        "source": source,
                        "page": page,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks