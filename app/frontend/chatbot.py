from collections import Counter
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Literal, TypedDict

import streamlit as st

from app.frontend.ingestion import index_uploaded_pdf
from app.rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
LOCAL_EMBEDDING_MODEL = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
DEFAULT_UI_EMBEDDING_MODEL = (
    LOCAL_EMBEDDING_MODEL.relative_to(PROJECT_ROOT).as_posix()
    if LOCAL_EMBEDDING_MODEL.is_dir()
    else DEFAULT_EMBEDDING_MODEL
)


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    citations: list[dict[str, object]]


@st.cache_resource(show_spinner="Loading the local embedding model and index...")
def _get_components(
    embedding_model: str,
    ollama_model: str,
    ollama_url: str,
) -> tuple[ResearchRetriever, OllamaAnswerGenerator]:
    model_identifier, model_path = _resolve_embedding_model(embedding_model)
    embedder = SentenceTransformerEmbedder(
        model_identifier,
        model_path=model_path,
    )
    store = ChromaVectorStore(
        DEFAULT_PERSIST_DIRECTORY,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_model=embedder.model_name,
    )
    return (
        ResearchRetriever(embedder, store),
        OllamaAnswerGenerator(ollama_model, base_url=ollama_url),
    )


def _resolve_embedding_model(embedding_model: str) -> tuple[str, str | None]:
    candidate = Path(embedding_model).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if not candidate.is_dir():
        return embedding_model, None

    model_path = candidate.resolve()
    try:
        model_identifier = model_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        model_identifier = str(model_path)
    return model_identifier, str(model_path)


def _render_sources(citations: list[dict[str, object]]) -> None:
    if not citations:
        return
    with st.expander("Sources"):
        for citation in citations:
            st.markdown(
                f"[{citation['reference']}] **{citation['source']}**, "
                f"page {citation['page']} "
                f"(cosine distance: {float(citation['distance']):.4f})"
            )


def _render_chat_history(messages: list[ChatMessage]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message["citations"])


def _synchronize_chat_scope(
    session_state: MutableMapping[str, object],
    sources: Sequence[str],
) -> bool:
    source_scope = tuple(dict.fromkeys(sources))
    if session_state.get("chat_source_scope") == source_scope:
        return False

    had_history = bool(session_state.get("chat_messages"))
    session_state["chat_messages"] = []
    session_state["chat_source_scope"] = source_scope
    return had_history


def main() -> None:
    st.set_page_config(page_title="Research Assistant")
    st.title("Research Assistant")
    st.caption("Ask questions about your PDFs and get answers with page citations.")

    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Add PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )
        embedding_model = st.text_input(
            "Embedding model or local path",
            value=DEFAULT_UI_EMBEDDING_MODEL,
            help="Must match the embedding model used by the current local index.",
        )
        ollama_model = st.text_input("Ollama chat model", value=DEFAULT_OLLAMA_MODEL)
        ollama_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL)
        n_results = st.slider("Evidence pages", min_value=1, max_value=10, value=5)

        if st.button("Clear conversation"):
            st.session_state["chat_messages"] = []
            st.rerun()

        if st.button("Index selected PDFs", disabled=not uploaded_files):
            filenames = [uploaded.name for uploaded in uploaded_files or []]
            duplicates = [
                name
                for name, count in Counter(
                    filename.casefold() for filename in filenames
                ).items()
                if count > 1
            ]
            if duplicates:
                st.error(
                    "Uploaded PDFs must have unique filenames: "
                    + ", ".join(sorted(duplicates))
                )
            else:
                try:
                    retriever, _ = _get_components(
                        embedding_model,
                        ollama_model,
                        ollama_url,
                    )
                    indexed_sources: list[str] = []
                    for uploaded in uploaded_files or []:
                        try:
                            with st.spinner(f"Indexing {uploaded.name}..."):
                                result = index_uploaded_pdf(
                                    uploaded,
                                    retriever,
                                    storage_directory=RAW_DATA_DIRECTORY,
                                )
                            st.success(
                                f"Indexed {result['source']}: "
                                f"{result['page_count']} page(s), "
                                f"{result['chunk_count']} chunk(s)."
                            )
                            indexed_sources.append(result["source"])
                        except (OSError, RuntimeError, ValueError) as error:
                            st.error(f"{uploaded.name}: {error}")
                    if indexed_sources:
                        st.session_state["current_pdf_sources"] = indexed_sources
                        st.session_state["chat_messages"] = []
                except (OSError, RuntimeError, ValueError) as error:
                    st.error(f"Could not load the index for uploaded files: {error}")

    try:
        retriever, generator = _get_components(
            embedding_model,
            ollama_model,
            ollama_url,
        )
        indexed_chunk_count = retriever.count()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not load the research index: {error}")
        st.info("Check the embedding model path and Chroma collection settings.")
        return

    current_sources = [
        uploaded.name for uploaded in uploaded_files or []
    ] or st.session_state.get("current_pdf_sources", [])
    if _synchronize_chat_scope(st.session_state, current_sources):
        st.info(
            "The PDF selection changed, so the previous conversation was cleared. "
            "Ask again to get an answer from the current PDF(s)."
        )
    if current_sources:
        st.sidebar.caption("Indexed chunks for current PDF(s):")
        for source in dict.fromkeys(current_sources):
            st.sidebar.caption(f"{source}: {retriever.count_for_source(source)}")
    else:
        st.sidebar.caption("Select and index a PDF before asking questions.")

    if indexed_chunk_count == 0:
        st.info("Upload and index a PDF from the sidebar to start chatting.")

    messages = st.session_state.setdefault("chat_messages", [])
    _render_chat_history(messages)
    question = st.chat_input("Ask a question about your indexed PDFs")
    if not question:
        return

    messages.append({"role": "user", "content": question, "citations": []})
    with st.chat_message("user"):
        st.markdown(question)

    if not current_sources:
        answer = "Select and index a PDF before asking questions."
        with st.chat_message("assistant"):
            st.markdown(answer)
        messages.append({"role": "assistant", "content": answer, "citations": []})
        return

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching sources and preparing an answer..."):
                evidence = retriever.search(
                    question,
                    n_results=n_results,
                    diversify_by_page=True,
                    sources=list(dict.fromkeys(current_sources)),
                )
                answer = generator.generate(question, evidence)
            st.markdown(answer["answer"])
            serialized_citations: list[dict[str, object]] = [
                dict(citation) for citation in answer["citations"]
            ]
            _render_sources(serialized_citations)
        messages.append(
            {
                "role": "assistant",
                "content": answer["answer"],
                "citations": serialized_citations,
            }
        )
    except (OSError, RuntimeError, ValueError) as error:
        st.error(str(error))


if __name__ == "__main__":
    main()
