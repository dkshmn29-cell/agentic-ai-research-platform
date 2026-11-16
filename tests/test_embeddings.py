from collections.abc import Sequence

from app.rag.embeddings import SentenceTransformerEmbedder


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.call_options: dict[str, object] = {}

    def encode(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> Sequence[Sequence[float]]:
        self.call_options = kwargs
        return [[0.6, 0.8] for _ in texts]


def test_sentence_transformer_adapter_returns_float_vectors() -> None:
    model = FakeSentenceTransformer()
    embedder = SentenceTransformerEmbedder(
        "test/model",
        batch_size=8,
        model=model,
    )

    vectors = embedder.embed(["First passage", "Second passage"])

    assert embedder.model_name == "test/model"
    assert vectors == [[0.6, 0.8], [0.6, 0.8]]
    assert model.call_options == {
        "batch_size": 8,
        "show_progress_bar": False,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
    }


def test_empty_text_batch_returns_without_calling_model() -> None:
    model = FakeSentenceTransformer()
    embedder = SentenceTransformerEmbedder(model=model)

    assert embedder.embed([]) == []
    assert model.call_options == {}