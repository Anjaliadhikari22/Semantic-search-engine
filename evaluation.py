"""Evaluation metrics and comparison helpers for retrieval systems."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Set

import numpy as np
import pandas as pd

from data_loader import EvalSample
from search_engine import SearchResult


SearchFunction = Callable[[str, int], List[SearchResult]]


def precision_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: Set[str], k: int = 5) -> float:
    """Compute Precision@k for a single query."""
    if k <= 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / k


def reciprocal_rank(ranked_doc_ids: Sequence[str], relevant_doc_ids: Set[str]) -> float:
    """Compute reciprocal rank for a single query."""
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: Set[str], k: int = 10) -> float:
    """Compute binary NDCG@k for a single query."""
    if k <= 0:
        return 0.0
    gains = np.array([1.0 if doc_id in relevant_doc_ids else 0.0 for doc_id in ranked_doc_ids[:k]], dtype=np.float32)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))

    ideal_len = min(k, len(relevant_doc_ids))
    if ideal_len == 0:
        return 0.0
    ideal_gains = np.ones(ideal_len, dtype=np.float32)
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2))
    idcg = float(np.sum(ideal_gains * ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retriever(
    search_fn: SearchFunction,
    eval_samples: Sequence[EvalSample],
    ndcg_k: int = 10,
    p_at_k: int = 5,
) -> Dict[str, float]:
    """Evaluate a retriever on NDCG@k, MRR, and Precision@k."""
    if not eval_samples:
        return {"NDCG@10": 0.0, "MRR": 0.0, "Precision@5": 0.0}

    ndcg_scores: List[float] = []
    rr_scores: List[float] = []
    p5_scores: List[float] = []

    for sample in eval_samples:
        # Pull enough candidates so metrics have sufficient depth.
        depth = max(ndcg_k, p_at_k, 20)
        results = search_fn(sample.query_text, depth)
        ranked_ids = [result.doc_id for result in results]

        ndcg_scores.append(ndcg_at_k(ranked_ids, sample.relevant_doc_ids, k=ndcg_k))
        rr_scores.append(reciprocal_rank(ranked_ids, sample.relevant_doc_ids))
        p5_scores.append(precision_at_k(ranked_ids, sample.relevant_doc_ids, k=p_at_k))

    return {
        f"NDCG@{ndcg_k}": float(np.mean(ndcg_scores)),
        "MRR": float(np.mean(rr_scores)),
        f"Precision@{p_at_k}": float(np.mean(p5_scores)),
    }


def compare_retrievers(
    two_stage_search_fn: SearchFunction,
    bm25_search_fn: SearchFunction,
    eval_samples: Sequence[EvalSample],
) -> pd.DataFrame:
    """Build a metrics comparison table for two-stage and BM25 retrievers."""
    two_stage_metrics = evaluate_retriever(two_stage_search_fn, eval_samples, ndcg_k=10, p_at_k=5)
    bm25_metrics = evaluate_retriever(bm25_search_fn, eval_samples, ndcg_k=10, p_at_k=5)

    return pd.DataFrame(
        {
            "Retriever": ["Two-Stage (Bi + Cross Encoder)", "BM25 Baseline"],
            "NDCG@10": [two_stage_metrics["NDCG@10"], bm25_metrics["NDCG@10"]],
            "MRR": [two_stage_metrics["MRR"], bm25_metrics["MRR"]],
            "Precision@5": [two_stage_metrics["Precision@5"], bm25_metrics["Precision@5"]],
        }
    )
