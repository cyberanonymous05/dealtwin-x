"""Macro condition feature vector at deal announcement date: rates, credit spreads, equity market
regime, and filing recency (raw calendar position -- corpus-wide z-scoring then makes deals from a
similar era naturally closer together, which is what the relevance heuristic's recency bonus rewards)."""

from datetime import date

import numpy as np

from src.data_ingestion.fred_data import DEFAULT_SERIES

_RATE_FEATURE_NAMES = list(DEFAULT_SERIES.values())
MACRO_FEATURE_NAMES = _RATE_FEATURE_NAMES + ["filing_date_ordinal"]


def build_macro_feature_vector(deal: dict) -> np.ndarray:
    macro_snapshot = deal.get("macro_snapshot") or {}
    values = [
        macro_snapshot.get(name) if macro_snapshot.get(name) is not None else np.nan
        for name in _RATE_FEATURE_NAMES
    ]

    filing_date = deal.get("filing_date")
    try:
        values.append(float(date.fromisoformat(filing_date).toordinal()) if filing_date else np.nan)
    except ValueError:
        values.append(np.nan)

    return np.array(values, dtype=float)
