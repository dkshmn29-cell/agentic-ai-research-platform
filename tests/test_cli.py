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


def test_ask_command_defaults_to_page_diversity_and_local_ollama() -> None:
    args = build_parser().parse_args(
        [
            "ask",
            "What should be verified?",
            "--embedding-model",
            "models/all-MiniLM-L6-v2",
        ]
    )

    assert args.diversify_pages
    assert args.ollama_model == "llama3.2:3b"
    assert args.ollama_url == "http://localhost:11434"
    assert args.embedding_model == "models/all-MiniLM-L6-v2"


def test_evaluate_command_accepts_case_file_and_local_model_path() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--cases",
            "evaluation/retrieval_cases.json",
            "--k",
            "3",
            "--diversify-pages",
            "--embedding-model",
            "models/all-MiniLM-L6-v2",
        ]
    )

    assert args.cases.name == "retrieval_cases.json"
    assert args.k == 3
    assert args.diversify_pages
    assert args.embedding_model == "models/all-MiniLM-L6-v2"