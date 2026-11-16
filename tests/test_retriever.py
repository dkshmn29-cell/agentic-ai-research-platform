from collections.abc import Sequence
from pathlib import Path

import pytest

from app.rag.chunking import TextChunk
from app.rag.retriever import ResearchRetriever
from app.rag.vector_store import ChromaVectorStore


class KeywordTestEmbedder:
    model_name = "keyword-test-model"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            normalized_text = text.lower()
            if "risk" in normalized_text or "review" in normalized_text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors


class RankedTestEmbedder:
    model_name = "ranked-test-model"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [1.0, 0.0]
            if text == "query"
            else {
                "first chunk": [1.0, 0.0],
                "second chunk": [0.9, 0.1],
                "other page": [0.8, 0.2],
            }[text]
            for text in texts
        ]


def create_chunk(chunk_id: str, content: str, page: int) -> TextChunk:
    return {
        "content": content,
        "metadata": {
            "source": "research.pdf",
            "page": page,
            "chunk_id": chunk_id,
            "chunk_index": 1,
        },
    }


def create_retriever(tmp_path: Path) -> ResearchRetriever:
    embedder = KeywordTestEmbedder()
    store = ChromaVectorStore(
        tmp_path,
        collection_name="retriever_test",
        embedding_model=embedder.model_name,
    )
    return ResearchRetriever(embedder, store)


def test_retriever_indexes_and_returns_relevant_cited_chunk(tmp_path: Path) -> None:
    retriever = create_retriever(tmp_path)
    chunks = [
        create_chunk("service", "Customer service procedure.", 1),
        create_chunk("risk-review", "Review the risks before use.", 3),
    ]

    assert retriever.index(chunks) == 2
    results = retriever.search("review risks", n_results=1)

    assert len(results) == 1
    assert results[0]["metadata"]["chunk_id"] == "risk-review"
    assert results[0]["metadata"]["source"] == "research.pdf"
    assert results[0]["metadata"]["page"] == 3


def test_retriever_rejects_empty_query(tmp_path: Path) -> None:
    retriever = create_retriever(tmp_path)

    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.search("   ")


def test_retriever_rejects_embedder_store_model_mismatch(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path, embedding_model="another-model")

    with pytest.raises(ValueError, match="same model"):
        ResearchRetriever(KeywordTestEmbedder(), store)


def test_search_can_prioritize_distinct_source_pages(tmp_path: Path) -> None:
    embedder = RankedTestEmbedder()
    store = ChromaVectorStore(
        tmp_path,
        collection_name="page_diversity_test",
        embedding_model=embedder.model_name,
    )
    retriever = ResearchRetriever(embedder, store)
    chunks = [
        create_chunk("first", "first chunk", 1),
        create_chunk("second", "second chunk", 1),
        create_chunk("other", "other page", 2),
    ]
    retriever.index(chunks)

    ordinary_results = retriever.search("query", n_results=2)
    diverse_results = retriever.search(
        "query",
        n_results=2,
        diversify_by_page=True,
    )

    assert [result["metadata"]["page"] for result in ordinary_results] == [1, 1]
    assert [result["metadata"]["page"] for result in diverse_results] == [1, 2]