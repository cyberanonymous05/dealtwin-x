"""Causal-temporal retrieval evaluation engine.

Protocol
  * Causal mask: query i may retrieve deal j only if j != i and t_j <= t_i (filing dates). Under constructs
    built from the realized multiple, the candidate pool is further restricted to deals with a multiple.
  * Folds: GroupKFold by two-digit SIC ("sic") or an expanding time window ("time"; the earliest block is
    only ever used for training).
  * Everything fitted -- imputation, scaling, size bins, block weights, the ranker -- is fitted on the
    training fold only. Learned retrievers are trained on (query, candidate) pairs whose query AND
    candidate both lie in the training fold, so no test-fold outcome ever enters training.
  * Test queries are ranked against every admissible candidate (training and test deals alike).

Metrics (per query, then averaged): NDCG@5 (linear gain), MAP, MRR, P@5, R@10 over binary relevance
(r > 0), and within-industry (WI) NDCG@5/@10 computed only over same-two-digit-SIC candidates. A query is
WI-evaluable when it has >= 3 same-industry admissible candidates spanning >= 2 relevance levels.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import GroupKFold

METRICS = ("ndcg@5", "map", "mrr", "precision@5", "recall@10", "wi_ndcg@5", "wi_ndcg@10", "hits@10")
TESTED_METRICS = METRICS[:-1]  # hits@10 is descriptive (practical significance), not tested


def causal_mask(date_ord: np.ndarray, causal: bool = True, candidate_pool: np.ndarray | None = None) -> np.ndarray:
    """Admissible (query, candidate) pairs. `candidate_pool` restricts which deals may be retrieved at all:
    under constructs built from the realized multiple, only deals WITH a computable multiple can be
    relevant, so retrieving the others would reward a retriever for predicting label availability
    (e.g. "cash deals have multiples") rather than comparability."""
    n = len(date_ord)
    mask = date_ord[None, :] <= date_ord[:, None] if causal else np.ones((n, n), dtype=bool)
    if candidate_pool is not None:
        mask = mask & candidate_pool[None, :]
    mask[np.arange(n), np.arange(n)] = False
    return mask


def make_folds(attrs: dict, strategy: str = "sic", n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(attrs["sic2"])
    if strategy == "sic":
        return [(tr, te) for tr, te in GroupKFold(n_splits=n_splits).split(np.zeros(n), groups=attrs["sic2"])]
    if strategy == "time":
        order = np.argsort(attrs["date_ord"], kind="stable")
        blocks = np.array_split(order, n_splits + 1)
        return [(np.concatenate(blocks[: k + 1]), blocks[k + 1]) for k in range(n_splits)]
    raise ValueError(strategy)


def _dcg(rels: np.ndarray, k: int) -> float:
    r = rels[:k]
    return float(np.sum(r / np.log2(np.arange(2, r.size + 2)))) if r.size else 0.0


def _ndcg(rels_in_rank_order: np.ndarray, k: int) -> float:
    ideal = _dcg(np.sort(rels_in_rank_order)[::-1], k)
    return _dcg(rels_in_rank_order, k) / ideal if ideal > 0 else 0.0


def query_metrics(scores: np.ndarray, rel: np.ndarray, admissible: np.ndarray, same_ind: np.ndarray,
                  tiebreak: np.ndarray) -> dict | None:
    cand = np.flatnonzero(admissible)
    if cand.size == 0:
        return None
    r = rel[cand]
    total = int((r > 0).sum())
    if total == 0:
        return None
    order = np.lexsort((tiebreak[cand], -scores[cand]))
    ranked = r[order]
    binary = ranked > 0
    hits = np.flatnonzero(binary)
    out = {
        "ndcg@5": _ndcg(ranked, 5),
        "map": float(np.mean((np.arange(hits.size) + 1) / (hits + 1))),
        "mrr": 1.0 / (hits[0] + 1),
        "precision@5": float(binary[:5].sum() / 5),
        "recall@10": float(binary[:10].sum() / total),
        "hits@10": float(binary[:10].sum()),
    }
    wi = same_ind[cand][order]
    wi_rels = ranked[wi]
    if wi_rels.size >= 3 and np.unique(wi_rels).size >= 2:
        out["wi_ndcg@5"], out["wi_ndcg@10"] = _ndcg(wi_rels, 5), _ndcg(wi_rels, 10)
    return out


@dataclass
class RunResult:
    per_query: dict = field(default_factory=dict)   # metric -> {query_idx: value}
    extras: dict = field(default_factory=dict)       # e.g. learned weights per fold

    def summary(self) -> dict:
        s = {m: round(float(np.mean(list(v.values()))), 4) if v else None for m, v in self.per_query.items()}
        s["n_queries"] = len(self.per_query.get("ndcg@5", {}))
        s["n_wi_queries"] = len(self.per_query.get("wi_ndcg@5", {}))
        return s


def evaluate(retriever, attrs: dict, rel: np.ndarray, folds, causal: bool = True, seed: int = 0,
             query_filter: np.ndarray | None = None, candidate_pool: np.ndarray | None = None) -> RunResult:
    """`retriever(train_idx, test_idx) -> (scores[n_test, n], extras)` returns a score row per test query
    over all n deals; admissibility is applied here, not by the retriever."""
    mask = causal_mask(attrs["date_ord"], causal, candidate_pool)
    same_ind = attrs["sic2"][:, None] == attrs["sic2"][None, :]
    tiebreak = np.random.default_rng(seed).random(len(attrs["sic2"]))
    result = RunResult(per_query={m: {} for m in METRICS})
    for fold_no, (train_idx, test_idx) in enumerate(folds):
        scores, extras = retriever(np.asarray(train_idx), np.asarray(test_idx))
        if extras:
            result.extras.setdefault("folds", []).append(extras)
        for row, q in enumerate(test_idx):
            if query_filter is not None and not query_filter[q]:
                continue
            m = query_metrics(scores[row], rel[q], mask[q], same_ind[q], tiebreak)
            if m is None:
                continue
            for name, value in m.items():
                result.per_query[name][int(q)] = value
    return result
