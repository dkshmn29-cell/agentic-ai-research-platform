from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 32


class TextEmbedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = "cpu",
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        model_path: str | Path | None = None,
        model: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model_name = model_name
        self.batch_size = batch_size

        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(str(model_path or model_name), device=device)

        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(value) for value in vector] for vector in vectors]