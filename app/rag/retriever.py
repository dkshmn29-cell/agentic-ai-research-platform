from collections.abc import Sequence

from app.rag.chunking import TextChunk
from app.rag.embeddings import TextEmbedder
from app.rag.vector_store import ChromaVectorStore, SearchResult


class ResearchRetriever:
    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: ChromaVectorStore,
    ) -> None:
        if embedder.model_name != vector_store.embedding_model:
            raise ValueError(
                "The query embedder and vector store must use the same model."
            )

        self._embedder = embedder
        self._vector_store = vector_store

    def count(self) -> int:
        return self._vector_store.count()

    def index(self, chunks: Sequence[TextChunk]) -> int:
        if not chunks:
            return 0

        embeddings = self._embedder.embed([chunk["content"] for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("The embedder must return one vector per chunk.")

        return self._vector_store.upsert_chunks(chunks, embeddings)

    def replace_source(self, chunks: Sequence[TextChunk]) -> int:
        if not chunks:
            return 0

        embeddings = self._embedder.embed([chunk["content"] for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("The embedder must return one vector per chunk.")

        return self._vector_store.replace_source_chunks(chunks, embeddings)

    def search(
        self,
        query: str,
        *,
        n_results: int = 5,
        diversify_by_page: bool = False,
    ) -> list[SearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty.")

        embeddings = self._embedder.embed([normalized_query])
        if len(embeddings) != 1:
            raise ValueError("The embedder must return one vector for the query.")

        if diversify_by_page:
            return self._search_with_page_diversity(embeddings[0], n_results)

        return self._vector_store.search_by_embedding(
            embeddings[0],
            n_results=n_results,
        )

    def _search_with_page_diversity(
        self,
        embedding: list[float],
        n_results: int,
    ) -> list[SearchResult]:
        collection_count = self._vector_store.count()
        if collection_count == 0:
            return []

        candidate_count = min(n_results, collection_count)
        while True:
            candidates = self._vector_store.search_by_embedding(
                embedding,
                n_results=candidate_count,
            )
            unique_pages = {
                (result["metadata"]["source"], result["metadata"]["page"])
                for result in candidates
            }
            if len(unique_pages) >= n_results or candidate_count >= collection_count:
                break
            candidate_count = min(collection_count, candidate_count * 2)

        distinct_results: list[SearchResult] = []
        duplicate_results: list[SearchResult] = []
        seen_pages: set[tuple[str, int]] = set()
        for result in candidates:
            metadata = result["metadata"]
            page_reference = (metadata["source"], metadata["page"])
            if page_reference in seen_pages:
                duplicate_results.append(result)
            else:
                seen_pages.add(page_reference)
                distinct_results.append(result)

        return (distinct_results + duplicate_results)[:n_results]