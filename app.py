"""Streamlit app for a two-stage semantic search engine."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from data_loader import EvalSample, load_documents_and_eval_data
from evaluation import compare_retrievers
from search_engine import SearchResult, TwoStageSearchEngine


st.set_page_config(page_title="Two-Stage Semantic Search", page_icon=":mag:", layout="wide")


def initialize_system() -> tuple[TwoStageSearchEngine, list[EvalSample], str]:
    """Load data and build retrieval indexes once per app session."""
    progress = st.progress(0)
    status = st.empty()

    status.info("Loading dataset...")
    documents, eval_samples, dataset_source = load_documents_and_eval_data(max_docs=500)
    progress.progress(0.25)

    engine = TwoStageSearchEngine(documents=documents)

    def on_progress(value: float, message: str) -> None:
        # 25% already used by dataset loading; reserve remaining for indexing.
        progress.progress(min(1.0, 0.25 + value * 0.75))
        status.info(message)

    engine.build_indices(progress_cb=on_progress)
    progress.progress(1.0)
    status.success("Initialization complete.")
    return engine, eval_samples, dataset_source


def render_results(results: list[SearchResult]) -> None:
    """Render retrieved results with score details."""
    if not results:
        st.warning("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {result.title}**")
            st.write(result.text)

            score_parts = [f"Overall: `{result.score:.4f}`"]
            if result.bi_score is not None:
                score_parts.append(f"Bi-encoder: `{result.bi_score:.4f}`")
            if result.cross_score is not None:
                score_parts.append(f"Cross-encoder: `{result.cross_score:.4f}`")
            if result.bm25_score is not None:
                score_parts.append(f"BM25: `{result.bm25_score:.4f}`")
            st.caption(" | ".join(score_parts))


def main() -> None:
    st.title("Two-Stage Semantic Search Engine")
    st.caption("Bi-encoder retrieval + cross-encoder reranking with BM25 baseline comparison")

    if "engine" not in st.session_state:
        st.session_state.engine, st.session_state.eval_samples, st.session_state.dataset_source = initialize_system()

    engine: TwoStageSearchEngine = st.session_state.engine
    eval_samples: list[EvalSample] = st.session_state.eval_samples
    dataset_source: str = st.session_state.dataset_source

    st.success(f"Dataset source: **{dataset_source}** | Documents indexed: **{len(engine.documents)}**")

    search_tab, evaluation_tab = st.tabs(["Search", "Evaluation Dashboard"])

    with search_tab:
        st.subheader("Live Querying")
        retrieval_mode = st.toggle("Use Two-Stage Pipeline (off = BM25 Baseline)", value=True)
        query = st.text_input(
            "Search Query",
            placeholder="Ask something like: What are best practices for cloud migration strategy?",
        ).strip()

        if query:
            if retrieval_mode:
                results = engine.search_two_stage(query=query, top_k=5, candidate_k=20)
                st.info("Showing top-5 from two-stage pipeline (top-20 retrieved then reranked).")
            else:
                results = engine.search_bm25(query=query, top_k=5)
                st.info("Showing top-5 from BM25 baseline.")
            render_results(results)

    with evaluation_tab:
        st.subheader("Retrieval Metrics Comparison")
        st.write("Metrics: NDCG@10, MRR, Precision@5")

        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Running evaluation across query set..."):
                comparison_df = compare_retrievers(
                    two_stage_search_fn=lambda q, k: engine.search_two_stage(q, top_k=k, candidate_k=20),
                    bm25_search_fn=lambda q, k: engine.search_bm25(q, top_k=k),
                    eval_samples=eval_samples,
                )

            st.dataframe(comparison_df, width="stretch")

            chart_data = comparison_df.melt(
                id_vars="Retriever",
                value_vars=["NDCG@10", "MRR", "Precision@5"],
                var_name="Metric",
                value_name="Score",
            )

            chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("Metric:N", title="Metric"),
                    y=alt.Y("Score:Q", title="Score"),
                    color=alt.Color("Retriever:N", title="Retriever"),
                    column=alt.Column("Retriever:N", title=None),
                    tooltip=["Retriever", "Metric", alt.Tooltip("Score:Q", format=".4f")],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width="stretch")
            st.caption(f"Evaluated on {len(eval_samples)} queries.")


if __name__ == "__main__":
    main()
