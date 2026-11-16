import argparse
from pathlib import Path

from app.rag.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_documents,
)
from app.rag.document_loader import load_pdf
from app.rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from app.rag.evaluation import (
    DEFAULT_RETRIEVAL_CASES,
    evaluate_retrieval,
    load_retrieval_cases,
)
from app.rag.generation import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaAnswerGenerator,
)
from app.rag.retriever import ResearchRetriever
from app.rag.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    ChromaVectorStore,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest, index, and search research PDFs locally."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect",
        help="Extract and chunk a PDF without loading an embedding model.",
    )
    inspect_parser.add_argument("pdf_path", type=Path, help="Path to a PDF file")
    _add_chunk_options(inspect_parser)

    index_parser = commands.add_parser(
        "index",
        help="Extract, chunk, embed, and persist a PDF's contents.",
    )
    index_parser.add_argument("pdf_path", type=Path, help="Path to a PDF file")
    _add_chunk_options(index_parser)
    _add_vector_store_options(index_parser)

    search_parser = commands.add_parser(
        "search",
        help="Retrieve the most similar stored chunks for a question.",
    )
    search_parser.add_argument("query", help="Research question or search phrase")
    search_parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Maximum results to return (default: %(default)s)",
    )
    search_parser.add_argument(
        "--diversify-pages",
        action="store_true",
        help="Prefer distinct source pages before returning duplicate-page chunks",
    )
    _add_vector_store_options(search_parser)

    ask_parser = commands.add_parser(
        "ask",
        help="Generate an answer grounded in retrieved passages with citations.",
    )
    ask_parser.add_argument("query", help="Question to answer from indexed evidence")
    ask_parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Maximum evidence chunks to provide (default: %(default)s)",
    )
    ask_parser.add_argument(
        "--no-diversify-pages",
        dest="diversify_pages",
        action="store_false",
        help="Allow multiple evidence chunks from the same source page",
    )
    ask_parser.set_defaults(diversify_pages=True)
    ask_parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})",
    )
    ask_parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama API base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    _add_vector_store_options(ask_parser)

    evaluation_parser = commands.add_parser(
        "evaluate",
        help="Measure retrieval ranking against labeled source pages.",
    )
    evaluation_parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_RETRIEVAL_CASES,
        help="JSON file with retrieval questions and relevant pages",
    )
    evaluation_parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top results to evaluate (default: %(default)s)",
    )
    evaluation_parser.add_argument(
        "--diversify-pages",
        action="store_true",
        help="Prefer distinct source pages before returning duplicate-page chunks",
    )
    _add_vector_store_options(evaluation_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "inspect":
            _inspect_pdf(args.pdf_path, args.chunk_size, args.chunk_overlap)
        elif args.command == "index":
            _index_pdf(args)
        elif args.command == "search":
            _search(args)
        elif args.command == "ask":
            _ask(args)
        else:
            _evaluate(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


def _inspect_pdf(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    pages = load_pdf(pdf_path)
    chunks = chunk_documents(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        print(f"No extractable text found in {pdf_path.name}.")
        return

    print(
        f"Extracted text from {len(pages)} page(s) and created "
        f"{len(chunks)} chunk(s) in {pdf_path.name}."
    )
    for chunk in chunks:
        metadata = chunk["metadata"]
        print(
            f"Page {metadata['page']}, chunk {metadata['chunk_index']}: "
            f"{len(chunk['content'])} characters "
            f"(id {metadata['chunk_id'][:12]})"
        )


def _index_pdf(args: argparse.Namespace) -> None:
    pages = load_pdf(args.pdf_path)
    chunks = chunk_documents(
        pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if not chunks:
        print(f"No extractable text found in {args.pdf_path.name}; nothing indexed.")
        return

    retriever = _create_retriever(args)
    indexed_count = retriever.index(chunks)
    print(
        f"Indexed {indexed_count} chunk(s) from {len(pages)} page(s) "
        f"in {args.pdf_path.name}."
    )
    print(f"Vector database: {args.persist_directory}")


def _search(args: argparse.Namespace) -> None:
    retriever = _create_retriever(args)
    results = retriever.search(
        args.query,
        n_results=args.n_results,
        diversify_by_page=args.diversify_pages,
    )
    if not results:
        print("No indexed evidence found.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"[{rank}] {metadata['source']}, page {metadata['page']} "
            f"(cosine distance: {result['distance']:.4f})"
        )
        print(result["content"])
        print()


def _ask(args: argparse.Namespace) -> None:
    retriever = _create_retriever(args)
    evidence = retriever.search(
        args.query,
        n_results=args.n_results,
        diversify_by_page=args.diversify_pages,
    )
    answer = OllamaAnswerGenerator(
        args.ollama_model,
        base_url=args.ollama_url,
    ).generate(args.query, evidence)

    print(answer["answer"])
    if answer["citations"]:
        print("\nSources:")
        for citation in answer["citations"]:
            print(
                f"[{citation['reference']}] {citation['source']}, "
                f"page {citation['page']} "
                f"(cosine distance: {citation['distance']:.4f})"
            )


def _evaluate(args: argparse.Namespace) -> None:
    cases = load_retrieval_cases(args.cases)
    report = evaluate_retrieval(
        _create_retriever(args),
        cases,
        k=args.k,
        diversify_by_page=args.diversify_pages,
    )

    print(f"Hit@{report['k']}: {report['hit_rate_at_k']:.1%}")
    print(f"Recall@{report['k']}: {report['recall_at_k']:.1%}")
    print(f"MRR: {report['mean_reciprocal_rank']:.3f}")
    for case in report["cases"]:
        rank = case["first_relevant_rank"]
        rank_text = f"rank {rank}" if rank is not None else "not found"
        pages = ", ".join(
            f"{page['source']} p.{page['page']}"
            for page in case["retrieved_pages"]
        )
        print(
            f"{case['id']}: first relevant page {rank_text}; "
            f"page recall {case['page_recall']:.1%}; retrieved: {pages}"
        )


def _create_retriever(args: argparse.Namespace) -> ResearchRetriever:
    try:
        embedder = SentenceTransformerEmbedder(args.embedding_model)
    except OSError as error:
        raise RuntimeError(
            f"Could not load embedding model '{args.embedding_model}'. "
            "The model is downloaded from Hugging Face on first use; "
            "check network access or provide an available local model path."
        ) from error

    store = ChromaVectorStore(
        args.persist_directory,
        collection_name=args.collection_name,
        embedding_model=embedder.model_name,
    )
    return ResearchRetriever(embedder, store)


def _add_chunk_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum chunk size in characters (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Target overlap in characters (default: %(default)s)",
    )


def _add_vector_store_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=DEFAULT_PERSIST_DIRECTORY,
        help="Local Chroma database directory",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection (default: {DEFAULT_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence Transformers model (default: {DEFAULT_EMBEDDING_MODEL})",
    )