"""Two-stage semantic search engine and BM25 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from data_loader import Document


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class SearchResult:
    """Single retrieval result with multiple score views."""

    doc_id: str
    title: str
    text: str
    score: float
    bi_score: float | None = None
    cross_score: float | None = None
    bm25_score: float | None = None


class TwoStageSearchEngine:
    """Bi-encoder retrieval + cross-encoder reranking search engine."""

    def __init__(
        self,
        documents: Sequence[Document],
        bi_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.documents: List[Document] = list(documents)
        self.bi_model_name = bi_model_name
        self.cross_model_name = cross_model_name

        self.bi_model: SentenceTransformer | None = None
        self.cross_encoder: CrossEncoder | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.doc_embeddings: np.ndarray | None = None
        self.doc_by_id: Dict[str, Document] = {doc.id: doc for doc in self.documents}
        self.doc_ids_by_pos: List[str] = [doc.id for doc in self.documents]

        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: List[List[str]] = []

    def build_indices(self, progress_cb: ProgressCallback | None = None) -> None:
        """Build all models and indexes once at application startup."""
        if not self.documents:
            raise ValueError("Cannot build indices without documents.")

        self._report(progress_cb, 0.05, "Loading bi-encoder model...")
        self.bi_model = SentenceTransformer(self.bi_model_name)

        self._report(progress_cb, 0.20, "Encoding document corpus...")
        embeddings = self.bi_model.encode(
            [f"{doc.title}. {doc.text}" for doc in self.documents],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        self.doc_embeddings = embeddings

        self._report(progress_cb, 0.45, "Building FAISS flat index...")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.faiss_index = index

        self._report(progress_cb, 0.62, "Loading cross-encoder reranker...")
        self.cross_encoder = CrossEncoder(self.cross_model_name)

        self._report(progress_cb, 0.78, "Preparing BM25 baseline index...")
        self.tokenized_corpus = [self._tokenize(f"{doc.title} {doc.text}") for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self._report(progress_cb, 1.0, "Search indices ready.")

    def search_two_stage(self, query: str, top_k: int = 5, candidate_k: int = 20) -> List[SearchResult]:
        """Retrieve candidates with bi-encoder then rerank with cross-encoder."""
        self._ensure_ready(two_stage=True)
        assert self.bi_model is not None and self.cross_encoder is not None and self.faiss_index is not None

        query_embedding = self.bi_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        candidate_k = min(candidate_k, len(self.documents))
        bi_scores, indices = self.faiss_index.search(query_embedding, candidate_k)
        idxs = indices[0].tolist()
        scores = bi_scores[0].tolist()

        candidate_docs: List[Document] = [self.documents[i] for i in idxs if i >= 0]
        candidate_bi_scores: List[float] = [s for i, s in zip(idxs, scores) if i >= 0]
        if not candidate_docs:
            return []

        pairs = [(query, f"{doc.title}. {doc.text}") for doc in candidate_docs]
        cross_scores = self.cross_encoder.predict(pairs).tolist()

        reranked = sorted(
            zip(candidate_docs, candidate_bi_scores, cross_scores),
            key=lambda x: x[2],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                doc_id=doc.id,
                title=doc.title,
                text=doc.text,
                score=float(cross_score),
                bi_score=float(bi_score),
                cross_score=float(cross_score),
            )
            for doc, bi_score, cross_score in reranked
        ]

    def search_bm25(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve results using BM25 baseline ranking."""
        self._ensure_ready(two_stage=False)
        assert self.bm25 is not None

        query_tokens = self._tokenize(query)
        raw_scores = self.bm25.get_scores(query_tokens)
        if raw_scores.size == 0:
            return []

        top_k = min(top_k, len(self.documents))
        top_idxs = np.argsort(raw_scores)[::-1][:top_k]
        results: List[SearchResult] = []
        for idx in top_idxs:
            doc = self.documents[int(idx)]
            bm25_score = float(raw_scores[int(idx)])
            results.append(
                SearchResult(
                    doc_id=doc.id,
                    title=doc.title,
                    text=doc.text,
                    score=bm25_score,
                    bm25_score=bm25_score,
                )
            )
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token for token in text.lower().split() if token]

    @staticmethod
    def _report(progress_cb: ProgressCallback | None, value: float, message: str) -> None:
        if progress_cb is not None:
            progress_cb(value, message)

    def _ensure_ready(self, two_stage: bool) -> None:
        if two_stage:
            if self.bi_model is None or self.cross_encoder is None or self.faiss_index is None:
                raise RuntimeError("Two-stage indices are not initialized. Call build_indices() first.")
        else:
            if self.bm25 is None:
                raise RuntimeError("BM25 index is not initialized. Call build_indices() first.")
