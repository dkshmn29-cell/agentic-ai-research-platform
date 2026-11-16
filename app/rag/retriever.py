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

    def index(self, chunks: Sequence[TextChunk]) -> int:
        if not chunks:
            return 0

        embeddings = self._embedder.embed([chunk["content"] for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("The embedder must return one vector per chunk.")

        return self._vector_store.upsert_chunks(chunks, embeddings)

    def search(self, query: str, *, n_results: int = 5) -> list[SearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty.")

        embeddings = self._embedder.embed([normalized_query])
        if len(embeddings) != 1:
            raise ValueError("The embedder must return one vector for the query.")

        return self._vector_store.search_by_embedding(
            embeddings[0],
            n_results=n_results,
        )