"""Learn per-block similarity weights for DealDNA via a cross-validated pairwise ranking objective,
replacing the earlier hand-picked constants (industry=1.5, numeric=1.0, legal=0.35).

Why this matters: those constants were arbitrary. A fair test of whether DealDNA's fuller feature set
can compete with a naive industry+size retriever requires letting the DATA decide how much each block
should count -- but fitting those weights on the exact same relevance labels used to report the final
metric would be circular (of course a retriever tuned to match the labels scores well against them).

fit_block_weights() is fit ONLY on a GroupKFold TRAIN split (grouped by 2-digit SIC, same grouping used
elsewhere), so weights for a held-out industry are learned entirely from OTHER industries' pairwise
relevance patterns. evaluate_learned_metric() runs proper nested cross-validation: for each fold, fit
weights on train, then score only that fold's test queries with those weights, and aggregate out-of-fold
results across all folds. The reported metric is therefore not circular with the labels used to fit it.
"""

import random

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.cross_validation import build_relevance_labels, deal_level_splits
from src.evaluation.metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k


def compute_block_similarities(records: list[dict], blocks: tuple) -> dict:
    """Per-block, corpus-normalized cosine similarity matrices (n x n), one per block, unweighted."""
    sims = {}
    n = len(records)

    if "industry" in blocks:
        sic_groups = sorted({r["sic_group"] for r in records})
        idx = {g: i for i, g in enumerate(sic_groups)}
        onehot = np.zeros((n, len(sic_groups)))
        for i, r in enumerate(records):
            onehot[i, idx[r["sic_group"]]] = 1.0
        sims["industry"] = onehot @ onehot.T

    for block in ("financial", "structure", "macro"):
        if block not in blocks:
            continue
        mat = np.vstack([r[f"{block}_vec"] for r in records])
        mat = SimpleImputer(strategy="median").fit_transform(mat)
        mat = StandardScaler().fit_transform(mat)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat_n = mat / np.where(norms == 0, 1.0, norms)
        sims[block] = mat_n @ mat_n.T

    if "legal" in blocks:
        mat = np.vstack([r["legal_vec"] for r in records])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat_n = mat / np.where(norms == 0, 1.0, norms)
        sims["legal"] = mat_n @ mat_n.T

    return sims


def _sample_training_pairs(indices, relevance: np.ndarray, max_pairs_per_query: int, rng: random.Random):
    pairs = []
    for i in indices:
        candidates = [j for j in indices if j != i]
        rng.shuffle(candidates)
        rel_i = relevance[i]
        collected = 0
        for a in range(len(candidates)):
            if collected >= max_pairs_per_query:
                break
            j = candidates[a]
            for b in range(a + 1, len(candidates)):
                k = candidates[b]
                if rel_i[j] == rel_i[k]:
                    continue
                pairs.append((i, j, k, 1 if rel_i[j] > rel_i[k] else 0))
                collected += 1
                if collected >= max_pairs_per_query:
                    break
    return pairs


def fit_block_weights(
    train_indices: list[int], relevance: np.ndarray, block_sims: dict, block_names: list[str],
    max_pairs_per_query: int = 25, seed: int = 0,
) -> dict:
    """Fit non-negative block weights from a pairwise ranking logistic regression on TRAIN indices only."""
    rng = random.Random(seed)
    pairs = _sample_training_pairs(train_indices, relevance, max_pairs_per_query, rng)
    if len(pairs) < 20:
        return {b: 1.0 for b in block_names}

    X = np.array([[block_sims[b][i, j] - block_sims[b][i, k] for b in block_names] for i, j, k, _ in pairs])
    y = np.array([label for _, _, _, label in pairs])

    clf = LogisticRegression(fit_intercept=False, max_iter=1000)
    clf.fit(X, y)
    weights = np.clip(clf.coef_.flatten(), a_min=0.0, a_max=None)
    if weights.sum() == 0:
        weights = np.ones(len(block_names))
    weights = weights / weights.sum() * len(block_names)  # same total "mass" as uniform weighting
    return dict(zip(block_names, weights.tolist()))


def combine_weighted_similarity(block_sims: dict, weights: dict) -> np.ndarray:
    n = next(iter(block_sims.values())).shape[0]
    combined = np.zeros((n, n))
    for block, sim in block_sims.items():
        combined += weights.get(block, 1.0) * sim
    return combined


def evaluate_learned_metric(
    deals: list[dict], records: list[dict], block_names: list[str], n_splits: int = 5, relevance: np.ndarray = None
) -> dict:
    """Nested-CV evaluation: fit block weights per fold on TRAIN deals, score only TEST deals' queries
    with those weights, aggregate out-of-fold. Also returns the fold-averaged learned weights themselves.
    Pass `relevance` explicitly to evaluate against a different relevance definition (e.g. the
    outcome-grounded one in cross_validation.build_outcome_based_relevance); defaults to the standard
    industry/size/structure/recency heuristic."""
    if relevance is None:
        relevance = build_relevance_labels(deals)
    block_sims = compute_block_similarities(records, tuple(block_names))
    groups = [d["sic"][:2] for d in deals]
    n = len(deals)
    n_splits_actual, splits = deal_level_splits(list(range(n)), groups, n_splits=n_splits)

    per_query = {"ndcg@5": [], "map": [], "mrr": [], "precision@5": [], "recall@10": []}
    within_ndcg5, within_ndcg10 = [], []
    fold_weights = []

    for train_idx, test_idx in splits:
        weights = fit_block_weights(list(train_idx), relevance, block_sims, block_names)
        fold_weights.append(weights)
        combined = combine_weighted_similarity(block_sims, weights)

        for query_idx in test_idx:
            total_relevant = int((relevance[query_idx] > 0).sum())
            if total_relevant == 0:
                continue
            order = np.argsort(-combined[query_idx])
            ranked_relevances = [relevance[query_idx, j] for j in order if j != query_idx]

            per_query["ndcg@5"].append(ndcg_at_k(ranked_relevances, k=5))
            per_query["map"].append(average_precision(ranked_relevances))
            per_query["precision@5"].append(precision_at_k(ranked_relevances, k=5))
            per_query["recall@10"].append(recall_at_k(ranked_relevances, total_relevant, k=10))
            hit_ranks = [i + 1 for i, r in enumerate(ranked_relevances) if r > 0]
            per_query["mrr"].append(1.0 / hit_ranks[0] if hit_ranks else 0.0)

            if total_relevant >= 3:
                within = [r for r in ranked_relevances if r > 0]
                within_ndcg5.append(ndcg_at_k(within, k=5))
                within_ndcg10.append(ndcg_at_k(within, k=10))

    summary = {name: round(float(np.mean(v)), 4) if v else None for name, v in per_query.items()}
    summary["within_industry_ndcg@5"] = round(float(np.mean(within_ndcg5)), 4) if within_ndcg5 else None
    summary["within_industry_ndcg@10"] = round(float(np.mean(within_ndcg10)), 4) if within_ndcg10 else None
    summary["n_evaluable_queries"] = len(per_query["ndcg@5"])
    summary["n_folds"] = n_splits_actual
    summary["avg_learned_weights"] = {
        b: round(float(np.mean([w[b] for w in fold_weights])), 3) for b in block_names
    }
    return summary
