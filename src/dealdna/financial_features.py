"""Numeric feature vector: size, growth, margins, leverage, valuation multiples, and deal premium."""

import math

import numpy as np

FINANCIAL_FEATURE_NAMES = [
    "log_revenue",
    "revenue_growth",
    "ebitda_margin",
    "net_margin",
    "leverage_ratio",
    "cash_ratio",
    "ev_to_revenue",
    "ev_to_ebitda",
    "deal_premium",
]


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return np.nan
    return numerator / denominator


def build_financial_feature_vector(deal: dict) -> np.ndarray:
    revenues = deal.get("revenues")
    log_revenue = math.log1p(revenues) if revenues and revenues > 0 else np.nan

    values = [
        log_revenue,
        deal.get("revenue_growth", np.nan) if deal.get("revenue_growth") is not None else np.nan,
        deal.get("ebitda_margin", np.nan) if deal.get("ebitda_margin") is not None else np.nan,
        _safe_ratio(deal.get("net_income"), revenues),
        _safe_ratio(deal.get("long_term_debt"), deal.get("total_assets")),
        _safe_ratio(deal.get("cash_and_equivalents"), deal.get("total_assets")),
        _safe_ratio(deal.get("ev_estimate"), revenues),
        _safe_ratio(deal.get("ev_estimate"), deal.get("ebitda_estimate")),
        deal.get("deal_premium", np.nan) if deal.get("deal_premium") is not None else np.nan,
    ]
    return np.array(values, dtype=float)
