from app.rag.cli import build_parser


def test_index_command_accepts_local_embedding_model_path() -> None:
    args = build_parser().parse_args(
        [
            "index",
            "data/raw/research.pdf",
            "--embedding-model",
            "models/all-MiniLM-L6-v2",
        ]
    )

    assert args.embedding_model == "models/all-MiniLM-L6-v2"


def test_search_command_accepts_local_embedding_model_path() -> None:
    args = build_parser().parse_args(
        [
            "search",
            "find the review steps",
            "--embedding-model",
            "models/all-MiniLM-L6-v2",
        ]
    )

    assert args.embedding_model == "models/all-MiniLM-L6-v2"