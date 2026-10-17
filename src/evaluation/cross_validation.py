"""Deal-level group K-fold cross-validation and relevance labeling, to avoid leakage between related
transactions (grouped by 2-digit SIC industry code) and to give retrieval evaluation something to score against."""

import math
from datetime import date

import numpy as np
from sklearn.model_selection import GroupKFold


def deal_level_splits(items, groups: list[str], n_splits: int = 5):
    """GroupKFold splits over `items` (any indexable, e.g. a feature matrix or a plain index range),
    grouped so that deals sharing a 2-digit SIC code never straddle the train/test boundary."""
    n_splits = min(n_splits, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    return n_splits, list(gkf.split(items, groups=groups))


def build_relevance_labels(deals: list[dict], size_log_window: float = 1.0, recency_days: int = 3 * 365) -> np.ndarray:
    """Graded relevance heuristic modeled on standard precedent-transaction screening criteria (no free
    expert-labeled ground truth exists, so this is a bootstrap proxy, not a banker's judgment call):
    a real comps screen filters candidates by (1) industry, (2) transaction size, (3) recency, and
    (4) consideration structure. Industry match is a gate -- a similarly-sized deal in an unrelated
    industry is not a comp, so it scores 0 regardless of the other factors.

    Score in [0, 3]: +1.5 exact SIC match / +1.0 same 2-digit SIC group (mutually exclusive), then
    +1.0 if revenue scale is within `size_log_window` natural-log units (~2.7x), +0.5 if consideration
    structure matches, +0.5 if filing dates are within `recency_days` of each other.
    """
    n = len(deals)
    relevance = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            if deals[i]["sic"] == deals[j]["sic"]:
                score = 1.5
            elif deals[i]["sic"][:2] == deals[j]["sic"][:2]:
                score = 1.0
            else:
                continue  # different industry entirely -> not a comp regardless of size/structure/timing

            rev_i, rev_j = deals[i].get("revenues"), deals[j].get("revenues")
            if rev_i and rev_j and rev_i > 0 and rev_j > 0:
                if abs(math.log(rev_i) - math.log(rev_j)) <= size_log_window:
                    score += 1.0

            mix_i, mix_j = deals[i].get("consideration_mix"), deals[j].get("consideration_mix")
            if mix_i and mix_j and mix_i != "unknown" and mix_i == mix_j:
                score += 0.5

            date_i, date_j = deals[i].get("filing_date"), deals[j].get("filing_date")
            if date_i and date_j:
                try:
                    if abs((date.fromisoformat(date_i) - date.fromisoformat(date_j)).days) <= recency_days:
                        score += 0.5
                except ValueError:
                    pass

            relevance[i, j] = score

    return relevance


def build_outcome_based_relevance(deals: list[dict], log_multiple_tight: float = 0.2, log_multiple_loose: float = 0.5) -> np.ndarray:
    """Relevance grounded in REAL financial outcomes rather than a hand-built heuristic: two deals are
    good comps for each other to the extent they command SIMILAR REALIZED EV/Revenue multiples -- which
    is literally the premise of precedent transaction analysis (you infer a target's multiple from what
    similar deals actually got paid). Still gated on same 2-digit SIC, since cross-industry multiples
    aren't comparable even when numerically close (a 3x SaaS multiple and a 3x industrials multiple mean
    very different things).

    Score in [0, 3]: +1.5 exact SIC / +1.0 same 2-digit group (as in build_relevance_labels), then +1.5 if
    |log(multiple_i) - log(multiple_j)| <= log_multiple_tight (~22% relative), +0.75 if <= log_multiple_loose
    (~65% relative), 0 otherwise. Deals missing a computable multiple can't be scored and are left at 0.

    IMPORTANT: any embedding evaluated against this relevance must exclude ev_to_revenue/ev_to_ebitda from
    its own features, or the comparison is circular (the target would be an input to itself).
    """
    n = len(deals)
    relevance = np.zeros((n, n), dtype=float)

    multiples = [
        math.log(d["ev_estimate"] / d["revenues"])
        if d.get("ev_estimate") and d.get("revenues") and d["ev_estimate"] > 0 and d["revenues"] > 0
        else None
        for d in deals
    ]

    for i in range(n):
        for j in range(n):
            if i == j or multiples[i] is None or multiples[j] is None:
                continue

            if deals[i]["sic"] == deals[j]["sic"]:
                score = 1.5
            elif deals[i]["sic"][:2] == deals[j]["sic"][:2]:
                score = 1.0
            else:
                continue

            diff = abs(multiples[i] - multiples[j])
            if diff <= log_multiple_tight:
                score += 1.5
            elif diff <= log_multiple_loose:
                score += 0.75

            relevance[i, j] = score

    return relevance
