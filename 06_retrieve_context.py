# -*- coding: utf-8 -*-
"""
06_retrieve_context.py
------------------------
Step 6 of the RAG pipeline: CONTEXT RETRIEVAL.

Given a user question, embeds it with the same model used in
04_vector_representation.py, queries the Chroma collection built in
05_create_chroma_store.py, and returns the top-k most relevant chunks.

Also includes standard retrieval evaluation metrics (Precision@k, Recall@k,
Hit Rate, Reciprocal Rank) so retrieval quality can be measured against a
small labeled test set of (question -> relevant chunk_ids) pairs. This is
useful both for the project report and for tuning CHUNK_SIZE / top_k.

This module is imported by both 07_prompting.py and streamlit_app.py -
it is not meant to be run standalone in production, but running it directly
(`python 06_retrieve_context.py`) executes a small self-test with a sample
question so you can sanity-check retrieval quality on its own.
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "credit_policy_chunks"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve_context(question: str, top_k: int = 4) -> list:
    """Return the top_k most relevant chunks for `question`.

    Each returned item is a dict with: chunk_id, text, title, source_file,
    doc_id, text_quality, distance (lower = more similar, since cosine
    distance is used).
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([question], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "title": metadata["title"],
            "source_file": metadata["source_file"],
            "doc_id": metadata["doc_id"],
            "text_quality": metadata["text_quality"],
            "distance": results["distances"][0][i],
        })

    return retrieved


# ---------------------------------------------------------------------------
# Retrieval evaluation metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of the top-k retrieved chunks that are actually relevant."""
    top_k_ids = retrieved_ids[:k]
    if not top_k_ids:
        return 0.0
    num_relevant_in_top_k = sum(1 for cid in top_k_ids if cid in relevant_ids)
    return num_relevant_in_top_k / len(top_k_ids)


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of all relevant chunks that were found in the top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k_ids = retrieved_ids[:k]
    num_relevant_found = sum(1 for cid in top_k_ids if cid in relevant_ids)
    return num_relevant_found / len(relevant_ids)


def hit_rate(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """1.0 if at least one relevant chunk appears in the top-k, else 0.0."""
    top_k_ids = retrieved_ids[:k]
    return 1.0 if any(cid in relevant_ids for cid in top_k_ids) else 0.0


def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """1 / (rank of the first relevant chunk), or 0.0 if none is found."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(test_cases: list, top_k: int = 4) -> dict:
    """Run retrieval over a labeled test set and average the metrics.

    test_cases: list of {"question": str, "relevant_chunk_ids": [str, ...]}
    """
    precisions, recalls, hits, rrs = [], [], [], []

    for case in test_cases:
        retrieved = retrieve_context(case["question"], top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        relevant_ids = set(case["relevant_chunk_ids"])

        precisions.append(precision_at_k(retrieved_ids, relevant_ids, top_k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, top_k))
        hits.append(hit_rate(retrieved_ids, relevant_ids, top_k))
        rrs.append(reciprocal_rank(retrieved_ids, relevant_ids))

    n = len(test_cases)
    return {
        "precision_at_k": sum(precisions) / n,
        "recall_at_k": sum(recalls) / n,
        "hit_rate": sum(hits) / n,
        "mean_reciprocal_rank": sum(rrs) / n,
        "top_k": top_k,
        "num_test_cases": n,
    }


if __name__ == "__main__":
    sample_question = "ما هو الحد الأقصى لنسبة أقساط القروض الاستهلاكية إلى الدخل الشهري؟"
    print(f"Question: {sample_question}\n")

    results = retrieve_context(sample_question, top_k=4)
    for i, r in enumerate(results, start=1):
        print(f"[{i}] {r['title']}  (distance={r['distance']:.4f}, quality={r['text_quality']})")
        print(r["text"][:200].replace("\n", " ") + "...\n")
