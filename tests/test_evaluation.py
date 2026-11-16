import json
from pathlib import Path

import pytest

from app.rag.evaluation import evaluate_retrieval, load_retrieval_cases
from app.rag.vector_store import SearchResult


class FixedResultSearcher:
    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self.results_by_query = results_by_query

    def search(
        self,
        query: str,
        *,
        n_results: int = 5,
        diversify_by_page: bool = False,
    ) -> list[SearchResult]:
        return self.results_by_query.get(query, [])[:n_results]


def create_result(source: str, page: int) -> SearchResult:
    return {
        "content": f"Text from page {page}",
        "metadata": {
            "source": source,
            "page": page,
            "chunk_id": f"chunk-{page}",
            "chunk_index": 1,
        },
        "distance": 0.1,
    }


def test_evaluation_reports_hit_rate_and_reciprocal_rank() -> None:
    cases = [
        {
            "id": "verify-source",
            "query": "What should be verified?",
            "relevant_pages": [{"source": "research.pdf", "page": 2}],
        },
        {
            "id": "missing-evidence",
            "query": "What is not in this file?",
            "relevant_pages": [{"source": "research.pdf", "page": 3}],
        },
    ]
    searcher = FixedResultSearcher(
        {
            "What should be verified?": [
                create_result("research.pdf", 1),
                create_result("research.pdf", 2),
            ],
            "What is not in this file?": [create_result("research.pdf", 1)],
        }
    )

    report = evaluate_retrieval(searcher, cases, k=3)

    assert report["hit_rate_at_k"] == 0.5
    assert report["recall_at_k"] == pytest.approx(0.5)
    assert report["mean_reciprocal_rank"] == pytest.approx(0.25)
    assert report["cases"][0]["first_relevant_rank"] == 2
    assert report["cases"][0]["page_recall"] == 1.0
    assert report["cases"][0]["retrieved_pages"] == [
        {"source": "research.pdf", "page": 1},
        {"source": "research.pdf", "page": 2},
    ]
    assert report["cases"][1]["first_relevant_rank"] is None
    assert report["cases"][1]["page_recall"] == 0.0


def test_recall_counts_unique_relevant_pages_and_averages_per_case() -> None:
    cases = [
        {
            "id": "multi-page",
            "query": "Find both pages",
            "relevant_pages": [
                {"source": "research.pdf", "page": 1},
                {"source": "research.pdf", "page": 2},
            ],
        },
        {
            "id": "single-page",
            "query": "Find one page",
            "relevant_pages": [{"source": "research.pdf", "page": 3}],
        },
    ]
    searcher = FixedResultSearcher(
        {
            "Find both pages": [
                create_result("research.pdf", 1),
                create_result("research.pdf", 1),
                create_result("research.pdf", 3),
            ],
            "Find one page": [create_result("research.pdf", 3)],
        }
    )

    report = evaluate_retrieval(searcher, cases, k=3)

    assert report["cases"][0]["page_recall"] == 0.5
    assert report["cases"][1]["page_recall"] == 1.0
    assert report["recall_at_k"] == pytest.approx(0.75)


def test_evaluation_rejects_non_positive_k() -> None:
    searcher = FixedResultSearcher({})
    cases = [
        {
            "id": "verify-source",
            "query": "What should be verified?",
            "relevant_pages": [{"source": "research.pdf", "page": 2}],
        }
    ]

    with pytest.raises(ValueError, match="k must be greater than zero"):
        evaluate_retrieval(searcher, cases, k=0)


def test_load_retrieval_cases_validates_page_references(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "verify-source",
                        "query": "What should be verified?",
                        "relevant_pages": [
                            {"source": "research.pdf", "page": 2}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_retrieval_cases(cases_path)

    assert cases[0]["id"] == "verify-source"
    assert cases[0]["relevant_pages"] == [
        {"source": "research.pdf", "page": 2}
    ]


def test_load_retrieval_cases_rejects_invalid_page_number(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "bad-page",
                        "query": "Question",
                        "relevant_pages": [{"source": "research.pdf", "page": 0}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive integer"):
        load_retrieval_cases(cases_path)