"""Information retrieval metrics for precedent transaction ranking quality: NDCG@5, MAP, MRR, Precision@5, Recall@10."""

import numpy as np

from src.evaluation.cross_validation import build_relevance_labels
from src.retrieval.faiss_index import search


def dcg_at_k(relevances: list[float], k: int) -> float:
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(relevances: list[float], k: int = 5) -> float:
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal_dcg


def mean_reciprocal_rank(relevant_ranks: list[int]) -> float:
    reciprocal_ranks = [1.0 / r for r in relevant_ranks if r > 0]
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def precision_at_k(relevances: list[float], k: int = 5) -> float:
    relevances = np.asarray(relevances)[:k]
    return float(np.sum(relevances > 0) / k) if k > 0 else 0.0


def recall_at_k(relevances: list[float], total_relevant: int, k: int = 10) -> float:
    if total_relevant == 0:
        return 0.0
    relevances = np.asarray(relevances)[:k]
    return float(np.sum(relevances > 0) / total_relevant)


def average_precision(relevances: list[float]) -> float:
    relevances = np.asarray(relevances)
    hits = np.where(relevances > 0)[0]
    if hits.size == 0:
        return 0.0
    precisions = [(i + 1) / (rank + 1) for i, rank in enumerate(hits)]
    return float(np.mean(precisions))


def evaluate_retrieval(deals: list[dict], embeddings: np.ndarray, index, relevance: np.ndarray = None) -> dict:
    """Leave-one-out retrieval evaluation over the whole deal corpus: each deal is a query in turn,
    ranked against every other deal, scored against a graded relevance matrix (defaults to the
    industry/size/structure/recency heuristic; pass `relevance` explicitly to use a different one,
    e.g. the outcome-grounded relevance based on realized valuation multiples)."""
    if relevance is None:
        relevance = build_relevance_labels(deals)
    n = len(deals)
    per_query = {"ndcg@5": [], "map": [], "mrr": [], "precision@5": [], "recall@10": []}
    evaluable = 0

    for query_idx in range(n):
        total_relevant = int((relevance[query_idx] > 0).sum())
        if total_relevant == 0:
            continue
        evaluable += 1
        _, indices = search(index, embeddings[query_idx].copy(), top_k=n)
        ranked_relevances = [relevance[query_idx, idx] for idx in indices if idx != query_idx and idx != -1]

        per_query["ndcg@5"].append(ndcg_at_k(ranked_relevances, k=5))
        per_query["map"].append(average_precision(ranked_relevances))
        per_query["precision@5"].append(precision_at_k(ranked_relevances, k=5))
        per_query["recall@10"].append(recall_at_k(ranked_relevances, total_relevant, k=10))
        hit_ranks = [i + 1 for i, r in enumerate(ranked_relevances) if r > 0]
        per_query["mrr"].append(1.0 / hit_ranks[0] if hit_ranks else 0.0)

    summary = {name: round(float(np.mean(vals)), 4) if vals else None for name, vals in per_query.items()}
    summary["n_evaluable_queries"] = evaluable
    summary["n_total_deals"] = n
    return summary


def evaluate_similarity_matrix(deals: list[dict], sim: np.ndarray, relevance: np.ndarray = None) -> dict:
    """Leave-one-out retrieval evaluation driven directly by an (n x n) similarity matrix, for retrievers
    that don't produce a FAISS-indexable vector (BM25, precomputed text-similarity baselines). Semantics
    match evaluate_retrieval(): each deal is a query, ranked against all others by descending similarity,
    scored against the graded relevance matrix. Also returns within-industry NDCG@5/@10."""
    if relevance is None:
        relevance = build_relevance_labels(deals)
    n = len(deals)
    per_query = {"ndcg@5": [], "map": [], "mrr": [], "precision@5": [], "recall@10": []}
    within5, within10 = [], []
    evaluable = 0

    for query_idx in range(n):
        total_relevant = int((relevance[query_idx] > 0).sum())
        if total_relevant == 0:
            continue
        evaluable += 1
        order = np.argsort(-sim[query_idx])
        ranked = [relevance[query_idx, j] for j in order if j != query_idx]

        per_query["ndcg@5"].append(ndcg_at_k(ranked, k=5))
        per_query["map"].append(average_precision(ranked))
        per_query["precision@5"].append(precision_at_k(ranked, k=5))
        per_query["recall@10"].append(recall_at_k(ranked, total_relevant, k=10))
        hit_ranks = [i + 1 for i, r in enumerate(ranked) if r > 0]
        per_query["mrr"].append(1.0 / hit_ranks[0] if hit_ranks else 0.0)

        if total_relevant >= 3:
            within = [r for r in ranked if r > 0]
            within5.append(ndcg_at_k(within, k=5))
            within10.append(ndcg_at_k(within, k=10))

    summary = {name: round(float(np.mean(v)), 4) if v else None for name, v in per_query.items()}
    summary["within_industry_ndcg@5"] = round(float(np.mean(within5)), 4) if within5 else None
    summary["within_industry_ndcg@10"] = round(float(np.mean(within10)), 4) if within10 else None
    summary["n_evaluable_queries"] = evaluable
    return summary


def evaluate_within_industry_differentiation(
    deals: list[dict], embeddings: np.ndarray, index, min_peers: int = 3, relevance: np.ndarray = None
) -> dict:
    """NDCG@5/@10 computed ONLY over same-2-digit-SIC-group candidates, using the full graded relevance
    score (exact-SIC / size / consideration-structure / recency bonuses on top of the industry floor).

    Why this exists: build_relevance_labels() gives every same-industry pair a relevance floor >= 1.0,
    so standard MAP/MRR/Precision/Recall (binary relevant-vs-not) ceiling out for ANY retriever that
    separates industries cleanly -- e.g. a bare industry one-hot scores MAP=MRR=1.0 by construction,
    which says nothing about whether it ranks WITHIN an industry any better than chance. Restricting to
    same-industry candidates and scoring with graded NDCG isolates that: does the embedding correctly
    prioritize the closer-matching precedents (same exact SIC, similar size, same deal structure, recent)
    over the merely same-broad-industry ones? This is the real test of whether structure/legal/macro
    signal adds value once industry has already been correctly identified.
    """
    if relevance is None:
        relevance = build_relevance_labels(deals)
    n = len(deals)
    ndcg5_scores, ndcg10_scores = [], []
    evaluable = 0

    for query_idx in range(n):
        same_industry_count = int((relevance[query_idx] > 0).sum())
        if same_industry_count < min_peers:
            continue
        evaluable += 1
        _, indices = search(index, embeddings[query_idx].copy(), top_k=n)
        within_industry_relevances = [
            relevance[query_idx, idx] for idx in indices
            if idx != query_idx and idx != -1 and relevance[query_idx, idx] > 0
        ]
        ndcg5_scores.append(ndcg_at_k(within_industry_relevances, k=5))
        ndcg10_scores.append(ndcg_at_k(within_industry_relevances, k=10))

    return {
        "within_industry_ndcg@5": round(float(np.mean(ndcg5_scores)), 4) if ndcg5_scores else None,
        "within_industry_ndcg@10": round(float(np.mean(ndcg10_scores)), 4) if ndcg10_scores else None,
        "within_industry_n_evaluable_queries": evaluable,
    }
