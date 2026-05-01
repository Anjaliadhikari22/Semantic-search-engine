"""Lightweight smoke tests for core project behavior.

This script intentionally avoids loading large ML models so it can run quickly
in CI and local pre-flight checks.
"""

from __future__ import annotations

from data_loader import _generate_synthetic_data
from evaluation import compare_retrievers, ndcg_at_k, precision_at_k, reciprocal_rank
from search_engine import SearchResult


def test_synthetic_data_generation() -> None:
    docs, samples = _generate_synthetic_data(count=500)
    assert len(docs) == 500, "Synthetic document count should be 500"
    assert len(samples) == 500, "Synthetic sample count should be 500"

    sample_doc = docs[0]
    assert sample_doc.id and sample_doc.title and sample_doc.text, "Document fields should be populated"
    assert sample_doc.relevant_query, "relevant_query should be populated"


def test_metric_functions() -> None:
    ranked_ids = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d3"}

    p5 = precision_at_k(ranked_ids, relevant, k=5)
    rr = reciprocal_rank(ranked_ids, relevant)
    ndcg10 = ndcg_at_k(ranked_ids, relevant, k=10)

    assert abs(p5 - 0.2) < 1e-9, f"Unexpected Precision@5: {p5}"
    assert abs(rr - (1 / 3)) < 1e-9, f"Unexpected RR: {rr}"
    assert 0.0 <= ndcg10 <= 1.0, f"NDCG should be normalized, got {ndcg10}"


def test_comparison_pipeline() -> None:
    docs, samples = _generate_synthetic_data(count=20)
    doc_ids = [doc.id for doc in docs]
    relevant_by_query = {sample.query_text: next(iter(sample.relevant_doc_ids)) for sample in samples}

    def strong_retriever(query: str, top_k: int) -> list[SearchResult]:
        # Always place the known relevant document first.
        lead_doc = relevant_by_query.get(query, doc_ids[0])
        ordered = [lead_doc] + [d for d in doc_ids if d != lead_doc]
        return [SearchResult(doc_id=d, title=d, text=d, score=1.0 / (i + 1)) for i, d in enumerate(ordered[:top_k])]

    def weak_retriever(query: str, top_k: int) -> list[SearchResult]:
        ordered = list(reversed(doc_ids))
        return [SearchResult(doc_id=d, title=d, text=d, score=1.0 / (i + 1)) for i, d in enumerate(ordered[:top_k])]

    table = compare_retrievers(strong_retriever, weak_retriever, samples)
    assert table.shape == (2, 4), f"Unexpected comparison table shape: {table.shape}"
    assert set(table["Retriever"]) == {
        "Two-Stage (Bi + Cross Encoder)",
        "BM25 Baseline",
    }, "Retriever labels mismatch"


def main() -> None:
    test_synthetic_data_generation()
    test_metric_functions()
    test_comparison_pipeline()
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
