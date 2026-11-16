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
    _add_vector_store_options(search_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "inspect":
            _inspect_pdf(args.pdf_path, args.chunk_size, args.chunk_overlap)
        elif args.command == "index":
            _index_pdf(args)
        else:
            _search(args)
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
    results = retriever.search(args.query, n_results=args.n_results)
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