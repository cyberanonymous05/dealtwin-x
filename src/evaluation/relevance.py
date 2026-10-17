"""Proxy relevance constructs for M&A precedent retrieval (no expert ground truth is publicly available).

All constructs are industry-gated on the two-digit SIC group and return an (n x n) graded relevance
matrix with a zero diagonal. They are proxies for an analyst's comparability judgement, not ground truth.

  heuristic       screen over industry, size, consideration structure and recency
  rubric          practitioner rubric: industry gate, size tier, business-profile distance, tie-breakers
  concordance     binary: industry, size, realized multiple and consideration structure all agree
  outcome         graded by realized EV/Revenue-multiple similarity (the premise of precedent analysis)
  legal           NEW: graded by agreement on ten regex-extracted legal deal terms (go-shop, reverse
                  termination fee, HSR, CFIUS, majority-of-minority, support agreements, financing
                  condition, top-up option, Section 251(h), CVR), independent of the clause selector

Constructs that use the realized multiple (concordance, outcome) must be paired with an embedding from
which every transaction-value-derived feature has been removed (see features.VALUE_DERIVED).
"""

import numpy as np

CONSTRUCTS = ("heuristic", "rubric", "concordance", "outcome", "legal")
USES_MULTIPLE = {"concordance", "outcome"}


def _industry_scores(sic: np.ndarray) -> np.ndarray:
    sic2 = np.array([s[:2] for s in sic])
    same4 = sic[:, None] == sic[None, :]
    same2 = sic2[:, None] == sic2[None, :]
    return np.where(same4, 1.5, np.where(same2, 1.0, 0.0))


def _abs_diff(x: np.ndarray) -> np.ndarray:
    return np.abs(x[:, None] - x[None, :])


def build_relevance(name: str, a: dict, outcome_tight: float = 0.2, outcome_loose: float = 0.5) -> np.ndarray:
    """`a` is the attribute dict from features.deal_attributes()."""
    ind = _industry_scores(a["sic"])
    gate = ind > 0
    n = len(a["sic"])

    if name == "heuristic":
        d_size = _abs_diff(a["log_revenue"])
        size_ok = np.nan_to_num(d_size, nan=np.inf) <= 1.0
        mix = a["mix"]
        mix_ok = (mix[:, None] == mix[None, :]) & (mix[:, None] != "unknown")
        recent = _abs_diff(a["date_ord"]) <= 3 * 365
        rel = ind + size_ok * 1.0 + mix_ok * 0.5 + recent * 0.5

    elif name == "rubric":
        d_size = np.nan_to_num(_abs_diff(a["log_revenue"]), nan=np.inf)
        size_tier = np.where(d_size <= 0.5, 1.0, np.where(d_size <= 1.0, 0.5, 0.0))
        prof = np.stack([a["growth_z"], a["margin_z"]], axis=1)
        d_prof = np.sqrt(np.nansum((prof[:, None, :] - prof[None, :, :]) ** 2, axis=2))
        has_prof = ~np.isnan(prof).any(axis=1)
        d_prof = np.where(has_prof[:, None] & has_prof[None, :], d_prof, np.inf)
        prof_tier = np.where(d_prof <= 0.5, 1.0, np.where(d_prof <= 1.0, 0.5, 0.0))
        mix = a["mix"]
        mix_ok = (mix[:, None] == mix[None, :]) & (mix[:, None] != "unknown")
        recent = _abs_diff(a["date_ord"]) <= 3 * 365
        rel = ind + size_tier + prof_tier + 0.25 * mix_ok + 0.25 * recent

    elif name == "concordance":
        m = a["log_multiple"]
        d_m = np.nan_to_num(_abs_diff(m), nan=np.inf)
        d_size = np.nan_to_num(_abs_diff(a["log_revenue"]), nan=np.inf)
        mix = a["mix"]
        mix_ok = (mix[:, None] == mix[None, :]) & (mix[:, None] != "unknown")
        rel = (gate & (d_size <= 1.0) & (d_m <= 0.3) & mix_ok).astype(float)

    elif name == "outcome":
        m = a["log_multiple"]
        has = ~np.isnan(m)
        d_m = np.nan_to_num(_abs_diff(m), nan=np.inf)
        bonus = np.where(d_m <= outcome_tight, 1.5, np.where(d_m <= outcome_loose, 0.75, 0.0))
        rel = np.where(has[:, None] & has[None, :], ind + bonus, 0.0)

    elif name == "legal":
        flags = a["legal_flags"].astype(float)
        agree = 1.0 - _abs_diff_matrix_mean(flags)
        bonus = np.where(agree >= 0.9, 1.5, np.where(agree >= 0.8, 0.75, 0.0))
        rel = ind + bonus

    else:
        raise ValueError(name)

    rel = np.where(gate, rel, 0.0)
    rel[np.arange(n), np.arange(n)] = 0.0
    return rel


def _abs_diff_matrix_mean(flags: np.ndarray) -> np.ndarray:
    """Mean over flags of |f_i - f_j|, i.e. the fraction of legal terms on which two deals disagree."""
    on = flags @ flags.T
    off = (1 - flags) @ (1 - flags).T
    return 1.0 - (on + off) / flags.shape[1]
