import math
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import chromadb

from app.rag.chunking import ChunkMetadata, TextChunk
from app.rag.embeddings import DEFAULT_EMBEDDING_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION_NAME = "research_chunks"


class SearchResult(TypedDict):
    content: str
    metadata: ChunkMetadata
    distance: float


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model

        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_model,
            },
            embedding_function=None,
        )

        metadata = self._collection.metadata or {}
        if metadata.get("embedding_model") != embedding_model:
            raise ValueError(
                "The existing collection uses a different embedding model. "
                "Choose a new collection name or use its original model."
            )
        if metadata.get("hnsw:space") != "cosine":
            raise ValueError("The existing collection does not use cosine distance.")

    def count(self) -> int:
        return self._collection.count()

    def upsert_chunks(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        if not chunks:
            return 0

        vectors = _validate_embeddings(embeddings)
        chunk_ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Chunk IDs must be unique within an upsert batch.")

        self._collection.upsert(
            ids=chunk_ids,
            embeddings=vectors,
            documents=[chunk["content"] for chunk in chunks],
            metadatas=[
                {
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                }
                for chunk in chunks
            ],
        )
        return len(chunks)

    def search_by_embedding(
        self,
        embedding: Sequence[float],
        *,
        n_results: int = 5,
    ) -> list[SearchResult]:
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")

        query_embedding = _validate_embeddings([embedding])[0]
        collection_count = self.count()
        if collection_count == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection_count),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"]
        metadatas = result["metadatas"]
        distances = result["distances"]

        if documents is None or metadatas is None or distances is None:
            raise RuntimeError("The vector store returned incomplete search results.")

        results: list[SearchResult] = []
        for chunk_id, content, metadata, distance in zip(
            ids,
            documents[0],
            metadatas[0],
            distances[0],
            strict=True,
        ):
            if content is None or metadata is None or distance is None:
                continue

            source = metadata.get("source")
            page = metadata.get("page")
            chunk_index = metadata.get("chunk_index")
            if (
                not isinstance(source, str)
                or not isinstance(page, int)
                or not isinstance(chunk_index, int)
            ):
                raise RuntimeError("A stored chunk is missing source/page metadata.")

            results.append(
                {
                    "content": content,
                    "metadata": {
                        "source": source,
                        "page": page,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                    },
                    "distance": float(distance),
                }
            )

        return results


def _validate_embeddings(
    embeddings: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not embeddings:
        return []

    vectors: list[list[float]] = []
    expected_dimension: int | None = None
    for embedding in embeddings:
        vector = [float(value) for value in embedding]
        if not vector:
            raise ValueError("Embeddings cannot be empty.")
        if expected_dimension is None:
            expected_dimension = len(vector)
        elif len(vector) != expected_dimension:
            raise ValueError("All embeddings in a batch must have the same dimension.")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding values must be finite numbers.")
        vectors.append(vector)

    return vectors