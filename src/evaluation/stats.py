"""Paired comparison statistics.

Deals are not independent: deals in the same industry and year share market conditions and often appear
in each other's candidate lists. Confidence intervals (percentile) and standard errors therefore come from a CLUSTER
bootstrap that resamples (two-digit SIC x filing year) blocks of queries with replacement. The per-deal
Wilcoxon signed-rank test is reported as a secondary check only. Holm's step-down correction is applied
across the entire family of reported comparisons (`holm`). Effect size is the paired Cliff's delta,
P(d > 0) - P(d < 0) over per-query differences d.
"""

import numpy as np
from scipy.stats import norm, wilcoxon


def paired(per_query_a: dict, per_query_b: dict, clusters: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Compare retriever A against B on the queries evaluable under both. `clusters[q]` is query q's block."""
    qs = np.array(sorted(set(per_query_a) & set(per_query_b)))
    d = np.array([per_query_a[q] - per_query_b[q] for q in qs])
    blocks = clusters[qs]
    uniq, inverse = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inverse, weights=d, minlength=uniq.size)
    counts = np.bincount(inverse, minlength=uniq.size)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, uniq.size, size=(n_boot, uniq.size))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    # p-value: cluster-robust z-test with the bootstrap standard error. (A raw percentile p-value is
    # floored at 1/n_boot, which after Holm over a family of hundreds of tests could never reach 0.05.)
    se = boot.std(ddof=1)
    p_boot = float(2 * norm.sf(abs(d.mean()) / se)) if se > 0 else (0.0 if d.mean() != 0 else 1.0)
    try:
        p_wil = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p_wil = 1.0
    return {
        "n": int(qs.size),
        "n_clusters": int(uniq.size),
        "mean_a": float(np.mean([per_query_a[q] for q in qs])),
        "mean_b": float(np.mean([per_query_b[q] for q in qs])),
        "diff": float(d.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "se_cluster_bootstrap": float(se),
        "p_cluster_bootstrap": float(p_boot),
        "p_wilcoxon": p_wil,
        "cliffs_delta": float(((d > 0).sum() - (d < 0).sum()) / d.size) if d.size else 0.0,
    }


def holm(pvalues: dict) -> dict:
    keys = sorted(pvalues, key=pvalues.get)
    m, adjusted, running = len(keys), {}, 0.0
    for rank, k in enumerate(keys):
        running = max(running, min(1.0, (m - rank) * pvalues[k]))
        adjusted[k] = running
    return adjusted


def effect_label(delta: float) -> str:
    a = abs(delta)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"
