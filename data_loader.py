"""Dataset loading and preprocessing utilities for semantic search."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Sequence, Set, Tuple

from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class Document:
    """A searchable document unit."""

    id: str
    title: str
    text: str
    relevant_query: str


@dataclass(frozen=True)
class EvalSample:
    """A single query and its relevant documents for evaluation."""

    query_id: str
    query_text: str
    relevant_doc_ids: Set[str]


def load_documents_and_eval_data(max_docs: int = 500) -> Tuple[List[Document], List[EvalSample], str]:
    """
    Load documents and evaluation samples from Hugging Face datasets.

    Falls back to synthetic data if remote datasets are unavailable.
    """
    loaders = (
        _try_load_beir_arguana,
        _try_load_embedding_training_data,
    )
    for loader in loaders:
        docs, samples, source = loader(max_docs=max_docs)
        if docs and samples:
            return docs, samples, source

    docs, samples = _generate_synthetic_data(count=max_docs)
    return docs, samples, "Synthetic QA (fallback)"


def _try_load_beir_arguana(max_docs: int) -> Tuple[List[Document], List[EvalSample], str]:
    """Attempt to load and normalize BeIR ArguAna format."""
    try:
        ds = load_dataset("BeIR/arguana")
    except Exception:
        return [], [], ""

    if not isinstance(ds, DatasetDict):
        return [], [], ""

    corpus = _get_split(ds, ("corpus", "train"))
    queries = _get_split(ds, ("queries", "validation", "test"))
    qrels = _get_split(ds, ("qrels", "train"))
    if corpus is None or queries is None:
        return [], [], ""

    doc_map: Dict[str, Document] = {}
    for row in corpus.select(range(min(len(corpus), max_docs))):
        doc_id = str(row.get("_id") or row.get("doc_id") or row.get("id") or f"doc-{len(doc_map)}")
        title = (row.get("title") or "Untitled").strip()
        text = (row.get("text") or row.get("body") or "").strip()
        if not text:
            continue
        doc_map[doc_id] = Document(
            id=doc_id,
            title=title,
            text=text,
            relevant_query="",
        )

    if not doc_map:
        return [], [], ""

    query_text_by_id: Dict[str, str] = {}
    for row in queries:
        qid = str(row.get("_id") or row.get("query_id") or row.get("id") or "")
        qtext = (row.get("text") or row.get("query") or "").strip()
        if qid and qtext:
            query_text_by_id[qid] = qtext

    relevant_by_qid: Dict[str, Set[str]] = {}
    if qrels is not None:
        for row in qrels:
            qid = str(
                row.get("query-id")
                or row.get("query_id")
                or row.get("qid")
                or row.get("query")
                or ""
            )
            did = str(
                row.get("corpus-id")
                or row.get("doc_id")
                or row.get("cid")
                or row.get("document")
                or ""
            )
            score = float(row.get("score", 1))
            if qid and did and score > 0 and did in doc_map:
                relevant_by_qid.setdefault(qid, set()).add(did)

    samples: List[EvalSample] = []
    for qid, qtext in query_text_by_id.items():
        rel_docs = relevant_by_qid.get(qid)
        if rel_docs:
            samples.append(EvalSample(query_id=qid, query_text=qtext, relevant_doc_ids=rel_docs))
        if len(samples) >= 200:
            break

    if not samples:
        # Build weak supervision by assigning each doc's title as query.
        docs = list(doc_map.values())
        for i, doc in enumerate(docs[: min(200, len(docs))]):
            qid = f"arguana-weak-{i}"
            query = doc.title if doc.title and doc.title != "Untitled" else doc.text[:80]
            samples.append(EvalSample(query_id=qid, query_text=query, relevant_doc_ids={doc.id}))
        docs = _attach_relevant_queries(docs, samples)
        return docs, samples, "BeIR/arguana (weak labels)"

    docs = list(doc_map.values())
    docs = _attach_relevant_queries(docs, samples)
    return docs, samples, "BeIR/arguana"


def _try_load_embedding_training_data(max_docs: int) -> Tuple[List[Document], List[EvalSample], str]:
    """Attempt to load sentence-transformers embedding training pairs."""
    try:
        ds = load_dataset("sentence-transformers/embedding-training-data")
    except Exception:
        return [], [], ""

    if isinstance(ds, DatasetDict):
        split = _get_split(ds, ("train", "validation", "test"))
    else:
        split = ds if isinstance(ds, Dataset) else None
    if split is None:
        return [], [], ""

    docs: List[Document] = []
    samples: List[EvalSample] = []
    seen_positive: Dict[str, str] = {}
    limit = min(len(split), max_docs)

    for i, row in enumerate(split.select(range(limit))):
        query = (row.get("anchor") or row.get("query") or row.get("sentence1") or "").strip()
        positive = (row.get("positive") or row.get("text") or row.get("sentence2") or "").strip()
        if not query or not positive:
            continue
        if positive not in seen_positive:
            doc_id = f"emb-doc-{len(seen_positive)}"
            seen_positive[positive] = doc_id
            docs.append(
                Document(
                    id=doc_id,
                    title=f"Training Doc {len(seen_positive)}",
                    text=positive,
                    relevant_query=query,
                )
            )
        qid = f"emb-q-{i}"
        samples.append(
            EvalSample(
                query_id=qid,
                query_text=query,
                relevant_doc_ids={seen_positive[positive]},
            )
        )

    if not docs or not samples:
        return [], [], ""
    return docs, samples, "sentence-transformers/embedding-training-data"


def _generate_synthetic_data(count: int = 500) -> Tuple[List[Document], List[EvalSample]]:
    """Generate realistic synthetic QA-style records as robust fallback."""
    topics = {
        "technology": [
            ("Cloud migration strategy", "A phased cloud migration reduces downtime by moving low-risk services first."),
            ("Cybersecurity zero trust", "Zero trust requires continuous verification for users and devices."),
            ("Microservices observability", "Tracing, metrics, and logs are essential for diagnosing distributed systems."),
            ("AI model deployment", "Model serving pipelines should include drift monitoring and rollback safeguards."),
        ],
        "science": [
            ("Climate change impacts", "Rising temperatures increase frequency of extreme weather events."),
            ("Gene editing ethics", "CRISPR offers medical breakthroughs but raises consent and equity concerns."),
            ("Quantum computing basics", "Qubits can represent superpositions, enabling parallel state exploration."),
            ("Vaccine development", "Clinical trials evaluate safety, dosage, and efficacy across multiple phases."),
        ],
        "business": [
            ("Revenue growth tactics", "Product-led growth can lower customer acquisition cost over time."),
            ("Supply chain resilience", "Diversified suppliers reduce risk during geopolitical disruptions."),
            ("Pricing strategy", "Value-based pricing aligns product price with measurable customer outcomes."),
            ("Team productivity", "Clear KPIs and feedback loops improve execution and accountability."),
        ],
        "health": [
            ("Sleep hygiene", "Consistent sleep schedules improve cognitive performance and metabolic health."),
            ("Heart health prevention", "Regular exercise and balanced nutrition reduce cardiovascular risk factors."),
            ("Mental wellness habits", "Mindfulness and social connection can reduce chronic stress."),
            ("Diabetes management", "Continuous glucose monitoring helps patients make real-time diet adjustments."),
        ],
    }

    query_templates = (
        "What are best practices for {title}?",
        "How does {title} work in practice?",
        "Why is {title} important for organizations?",
        "Explain key considerations in {title}.",
    )

    documents: List[Document] = []
    samples: List[EvalSample] = []
    rng = random.Random(42)
    all_topic_names = list(topics.keys())

    for i in range(count):
        topic = all_topic_names[i % len(all_topic_names)]
        title, base_text = rng.choice(topics[topic])
        title_with_topic = f"{title} in {topic.capitalize()}"
        detail = (
            f"{base_text} This document discusses implementation examples, common pitfalls, "
            f"and measurable outcomes for {topic} teams."
        )
        query = rng.choice(query_templates).format(title=title.lower())
        doc_id = f"syn-{i:04d}"
        documents.append(
            Document(
                id=doc_id,
                title=title_with_topic,
                text=detail,
                relevant_query=query,
            )
        )
        samples.append(EvalSample(query_id=f"syn-q-{i:04d}", query_text=query, relevant_doc_ids={doc_id}))

    return documents, samples


def _attach_relevant_queries(documents: Sequence[Document], samples: Sequence[EvalSample]) -> List[Document]:
    """Populate the relevant_query field from available evaluation samples."""
    query_by_doc: Dict[str, str] = {}
    for sample in samples:
        for did in sample.relevant_doc_ids:
            query_by_doc.setdefault(did, sample.query_text)

    updated_docs: List[Document] = []
    for doc in documents:
        updated_docs.append(
            Document(
                id=doc.id,
                title=doc.title,
                text=doc.text,
                relevant_query=query_by_doc.get(doc.id, doc.relevant_query),
            )
        )
    return updated_docs


def _get_split(ds: DatasetDict, candidates: Sequence[str]) -> Dataset | None:
    """Return the first existing split from candidate names."""
    for name in candidates:
        if name in ds:
            return ds[name]
    return None
