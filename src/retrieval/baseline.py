"""Naive industry+size baseline retriever: what a banker gets for free by filtering on SIC code and
revenue scale alone, with no transaction-structure, legal-clause, or macro signal. DealDNA's retrieval
quality is only a meaningful contribution if it beats this for free -- see scripts/run_ablation_study.py.
"""

import math

import numpy as np
from sklearn.preprocessing import StandardScaler


def build_naive_baseline_embeddings(deals: list[dict]) -> np.ndarray:
    sic_groups = sorted({d["sic"][:2] for d in deals})
    group_index = {g: i for i, g in enumerate(sic_groups)}

    n = len(deals)
    sic_onehot = np.zeros((n, len(sic_groups)), dtype=float)
    log_revenue = np.full((n, 1), np.nan, dtype=float)
    for i, d in enumerate(deals):
        sic_onehot[i, group_index[d["sic"][:2]]] = 1.0
        if d.get("revenues") and d["revenues"] > 0:
            log_revenue[i, 0] = math.log1p(d["revenues"])

    median = np.nanmedian(log_revenue)
    log_revenue = np.where(np.isnan(log_revenue), median, log_revenue)
    log_revenue_scaled = StandardScaler().fit_transform(log_revenue)

    # Industry block weighted up relative to the single size dimension so industry match dominates
    # similarity first, matching how a banker actually screens: industry, then size.
    return np.hstack([sic_onehot * 2.0, log_revenue_scaled]).astype("float32")
