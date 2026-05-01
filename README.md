# Two-Stage Semantic Search Engine

A clean, production-style semantic search project with:

- **Two-stage retrieval**: bi-encoder recall + cross-encoder reranking
- **BM25 baseline** for lexical comparison
- **Evaluation pipeline** with NDCG@10, MRR, and Precision@5
- **Streamlit UI** for live search and metric dashboarding

## Project Structure

- `app.py` - Streamlit user interface
- `search_engine.py` - Two-stage retrieval and BM25 implementation
- `evaluation.py` - Retrieval metrics and comparison table builder
- `data_loader.py` - Dataset loading/preprocessing + synthetic fallback
- `requirements.txt` - Python dependencies

## Retrieval Architecture

### Stage 1: Bi-Encoder Candidate Retrieval

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Document embeddings are precomputed once at startup
- Embeddings are stored in a **FAISS flat inner-product index**
- For each query, top-20 candidates are retrieved

### Stage 2: Cross-Encoder Reranking

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Query-document pairs from top-20 are scored
- Final top-5 results are returned

### Baseline: BM25

- Implemented using `rank_bm25`
- Used as a lexical retrieval baseline in evaluation

## Dataset Logic

The application tries datasets in this order:

1. `BeIR/arguana`
2. `sentence-transformers/embedding-training-data`

If both are unavailable (e.g., offline or access issue), it generates **500 realistic synthetic QA pairs** across:

- Technology
- Science
- Business
- Health

Each record has:

`{id, title, text, relevant_query}`

## Evaluation Metrics

The evaluation module compares **Two-Stage** vs **BM25** using:

- **NDCG@10**
- **MRR** (Mean Reciprocal Rank)
- **Precision@5**

Results are displayed as:

- A comparison table
- A bar chart dashboard

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Or use:

```bash
make run
```

On startup, the app:

1. Loads dataset (or synthetic fallback)
2. Builds FAISS, model loaders, and BM25 index
3. Shows progress in the UI

Indexes are built once per Streamlit session.

## Notes

- No Groq API key is required.
- First startup can take time due to model downloads.
- For reproducibility of fallback data, synthetic generation uses a fixed random seed.

## Handy Commands

```bash
make setup   # install dependencies
make smoke   # run lightweight verification checks
make run     # start streamlit app
```
