from pathlib import Path

import pytest

from app.rag.chunking import TextChunk
from app.rag.vector_store import ChromaVectorStore


def create_chunk(
    chunk_id: str,
    content: str,
    page: int,
) -> TextChunk:
    return {
        "content": content,
        "metadata": {
            "source": "research.pdf",
            "page": page,
            "chunk_id": chunk_id,
            "chunk_index": 1,
        },
    }


def test_store_upserts_searches_and_persists_metadata(tmp_path: Path) -> None:
    chunks = [
        create_chunk("transfer-policy", "Transfer policy for customer accounts.", 1),
        create_chunk("review-risks", "Review risks before approving a draft.", 3),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    store = ChromaVectorStore(
        tmp_path,
        collection_name="test_chunks",
        embedding_model="test-model",
    )

    assert store.upsert_chunks(chunks, embeddings) == 2
    assert store.upsert_chunks(chunks, embeddings) == 2
    assert store.count() == 2

    results = store.search_by_embedding([1.0, 0.0], n_results=1)

    assert len(results) == 1
    assert results[0]["content"] == chunks[0]["content"]
    assert results[0]["metadata"] == chunks[0]["metadata"]
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-6)

    reopened_store = ChromaVectorStore(
        tmp_path,
        collection_name="test_chunks",
        embedding_model="test-model",
    )
    assert reopened_store.count() == 2
    assert reopened_store.search_by_embedding([0.0, 1.0], n_results=1)[0][
        "metadata"
    ]["page"] == 3


def test_store_handles_empty_index(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path, embedding_model="test-model")

    assert store.upsert_chunks([], []) == 0
    assert store.search_by_embedding([1.0, 0.0]) == []


def test_store_rejects_chunk_embedding_count_mismatch(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path, embedding_model="test-model")
    chunks = [create_chunk("chunk-1", "Text", 1)]

    with pytest.raises(ValueError, match="exactly one embedding"):
        store.upsert_chunks(chunks, [])


def test_store_rejects_duplicate_chunk_ids_in_one_batch(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path, embedding_model="test-model")
    chunk = create_chunk("same-id", "Text", 1)

    with pytest.raises(ValueError, match="unique"):
        store.upsert_chunks([chunk, chunk], [[1.0, 0.0], [0.0, 1.0]])


def test_store_rejects_inconsistent_embedding_dimensions(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path, embedding_model="test-model")
    chunks = [create_chunk("one", "First", 1), create_chunk("two", "Second", 2)]

    with pytest.raises(ValueError, match="same dimension"):
        store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0, 0.0]])


def test_store_rejects_non_finite_embeddings(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path, embedding_model="test-model")
    chunk = create_chunk("one", "First", 1)

    with pytest.raises(ValueError, match="finite"):
        store.upsert_chunks([chunk], [[float("nan"), 0.0]])


def test_store_rejects_non_positive_result_count(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path, embedding_model="test-model")

    with pytest.raises(ValueError, match="greater than zero"):
        store.search_by_embedding([1.0, 0.0], n_results=0)