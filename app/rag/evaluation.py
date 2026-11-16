import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypedDict

from app.rag.vector_store import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVAL_CASES = PROJECT_ROOT / "evaluation" / "retrieval_cases.json"


class RelevantPage(TypedDict):
    source: str
    page: int


class RetrievalCase(TypedDict):
    id: str
    query: str
    relevant_pages: list[RelevantPage]


class CaseEvaluation(TypedDict):
    id: str
    query: str
    first_relevant_rank: int | None
    retrieved_pages: list[RelevantPage]
    page_recall: float
    reciprocal_rank: float


class EvaluationReport(TypedDict):
    k: int
    hit_rate_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    cases: list[CaseEvaluation]


class Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        n_results: int = 5,
        diversify_by_page: bool = False,
    ) -> list[SearchResult]: ...


def load_retrieval_cases(path: str | Path) -> list[RetrievalCase]:
    case_path = Path(path)
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Evaluation file must contain a 'cases' list.")
    if not payload["cases"]:
        raise ValueError("Evaluation file must contain at least one case.")

    cases: list[RetrievalCase] = []
    for case_number, raw_case in enumerate(payload["cases"], start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Evaluation case {case_number} must be an object.")

        case_id = raw_case.get("id")
        query = raw_case.get("query")
        raw_pages = raw_case.get("relevant_pages")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Evaluation case {case_number} needs a non-empty id.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Evaluation case {case_id!r} needs a non-empty query.")
        if not isinstance(raw_pages, list) or not raw_pages:
            raise ValueError(
                f"Evaluation case {case_id!r} needs at least one relevant page."
            )

        relevant_pages: list[RelevantPage] = []
        for page_number, raw_page in enumerate(raw_pages, start=1):
            if not isinstance(raw_page, dict):
                raise ValueError(
                    f"Relevant page {page_number} in case {case_id!r} must be an object."
                )
            source = raw_page.get("source")
            page = raw_page.get("page")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    f"Relevant page {page_number} in case {case_id!r} "
                    "needs a non-empty source."
                )
            if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
                raise ValueError(
                    f"Relevant page {page_number} in case {case_id!r} "
                    "must be a positive integer."
                )
            relevant_pages.append({"source": source, "page": page})

        cases.append(
            {
                "id": case_id,
                "query": query,
                "relevant_pages": relevant_pages,
            }
        )

    return cases


def evaluate_retrieval(
    searcher: Searcher,
    cases: Sequence[RetrievalCase],
    *,
    k: int = 5,
    diversify_by_page: bool = False,
) -> EvaluationReport:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if not cases:
        raise ValueError("At least one evaluation case is required.")

    evaluations: list[CaseEvaluation] = []
    hit_count = 0
    page_recall_total = 0.0
    reciprocal_rank_total = 0.0

    for case in cases:
        results = searcher.search(
            case["query"],
            n_results=k,
            diversify_by_page=diversify_by_page,
        )[:k]
        relevant_pages = {
            (page["source"], page["page"]) for page in case["relevant_pages"]
        }
        retrieved_relevant_pages: set[tuple[str, int]] = set()
        retrieved_pages: list[RelevantPage] = []
        first_relevant_rank: int | None = None

        for rank, result in enumerate(results, start=1):
            source = result["metadata"]["source"]
            page = result["metadata"]["page"]
            retrieved_pages.append({"source": source, "page": page})
            page_reference = (source, page)
            if page_reference in relevant_pages:
                retrieved_relevant_pages.add(page_reference)
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        page_recall = len(retrieved_relevant_pages) / len(relevant_pages)
        reciprocal_rank = (
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        )
        if first_relevant_rank is not None:
            hit_count += 1
        page_recall_total += page_recall
        reciprocal_rank_total += reciprocal_rank
        evaluations.append(
            {
                "id": case["id"],
                "query": case["query"],
                "first_relevant_rank": first_relevant_rank,
                "retrieved_pages": retrieved_pages,
                "page_recall": page_recall,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    case_count = len(cases)
    return {
        "k": k,
        "hit_rate_at_k": hit_count / case_count,
        "recall_at_k": page_recall_total / case_count,
        "mean_reciprocal_rank": reciprocal_rank_total / case_count,
        "cases": evaluations,
    }