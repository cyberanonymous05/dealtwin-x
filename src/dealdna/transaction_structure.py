"""Deal-term feature vector: cash vs. stock mix, earn-outs, termination fee sizing, MAC clause intensity."""

import math

import numpy as np

STRUCTURE_FEATURE_NAMES = [
    "is_cash",
    "is_stock",
    "is_mixed_consideration",
    "has_earnout",
    "termination_fee_pct_of_ev",
    "mac_clause_intensity",
]


def build_transaction_structure_vector(deal: dict) -> np.ndarray:
    mix = deal.get("consideration_mix")
    is_cash = 1.0 if mix == "cash" else 0.0
    is_stock = 1.0 if mix == "stock" else 0.0
    is_mixed = 1.0 if mix == "cash_and_stock" else 0.0
    has_earnout = 1.0 if deal.get("has_earnout") else 0.0

    fee = deal.get("termination_fee_usd")
    ev = deal.get("ev_estimate")
    fee_pct = (fee / ev) if fee is not None and ev not in (None, 0) else np.nan

    mac_mentions = deal.get("mac_clause_mentions")
    mac_intensity = math.log1p(mac_mentions) if mac_mentions is not None else np.nan

    return np.array([is_cash, is_stock, is_mixed, has_earnout, fee_pct, mac_intensity], dtype=float)
