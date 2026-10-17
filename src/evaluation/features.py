"""DealDNA block features with fold-local preprocessing and leakage control.

Blocks: industry (SIC one-hot, with 3/4-digit match indicators for the ranker), financial, structure,
acquirer, macro, legal (clause-passage text embedding) and business (10-K business-description embedding).

Leakage control: when the evaluation target is built from the realized multiple, every feature computed
from the transaction value is removed -- EV/Revenue, EV/EBITDA, deal premium, and the termination fee
(which is negotiated as a percentage of equity value, so fee/EV and fee/revenue both encode the price).

Fold-local preprocessing: `NumericBlocks.fit(train_idx)` estimates medians, means and standard deviations
on training-fold deals only; `transform()` then applies them to every deal (candidates and queries).
"""

import math
from datetime import date

import numpy as np

FINANCIAL = ["log_revenue", "revenue_growth", "ebitda_margin", "net_margin", "leverage_ratio", "cash_ratio",
             "ev_to_revenue", "ev_to_ebitda", "deal_premium"]
STRUCTURE = ["is_cash", "is_stock", "is_mixed", "has_earnout", "termination_fee_pct_of_ev", "mac_clause_intensity"]
ACQUIRER = ["is_sponsor", "acquirer_listed", "cross_border"]
MACRO = ["fed_funds_rate", "treasury_10y", "yield_curve_10y2y", "credit_spread_baa10y", "vix", "filing_date_ordinal"]
NUMERIC_BLOCKS = {"financial": FINANCIAL, "structure": STRUCTURE, "acquirer": ACQUIRER, "macro": MACRO}
VALUE_DERIVED = {"ev_to_revenue", "ev_to_ebitda", "deal_premium", "termination_fee_pct_of_ev"}
LEGAL_FLAG_NAMES = ["go_shop", "reverse_termination_fee", "hsr_required", "cfius_review", "majority_of_minority",
                    "voting_or_support_agreement", "financing_condition", "top_up_option", "section_251h", "cvr"]


def _ratio(a, b):
    return a / b if a is not None and b not in (None, 0) else np.nan


def raw_numeric(deal: dict) -> dict:
    rev = deal.get("revenues")
    usd = deal.get("reports_in_usd", True)
    macro = deal.get("macro_snapshot") or {}
    ev = deal.get("ev_estimate")
    fee = deal.get("termination_fee_usd")
    mix = deal.get("consideration_mix")
    return {
        "log_revenue": math.log1p(rev) if rev and rev > 0 and usd else np.nan,
        "revenue_growth": deal["revenue_growth"] if deal.get("revenue_growth") is not None and abs(deal["revenue_growth"]) < 10 else np.nan,
        "ebitda_margin": deal["ebitda_margin"] if deal.get("ebitda_margin") is not None and abs(deal["ebitda_margin"]) < 10 else np.nan,
        "net_margin": float(np.clip(_ratio(deal.get("net_income"), rev), -10, 10)) if rev else np.nan,
        "leverage_ratio": _ratio(deal.get("long_term_debt"), deal.get("total_assets")),
        "cash_ratio": _ratio(deal.get("cash_and_equivalents"), deal.get("total_assets")),
        "ev_to_revenue": math.log(ev / rev) if ev and rev and rev > 0 and ev > 0 else np.nan,
        "ev_to_ebitda": float(np.clip(_ratio(ev, deal.get("ebitda_estimate")), -100, 200)) if ev else np.nan,
        "deal_premium": deal["deal_premium"] if deal.get("deal_premium") is not None else np.nan,
        "is_cash": float(mix == "cash"),
        "is_stock": float(mix == "stock"),
        "is_mixed": float(mix == "cash_and_stock"),
        "has_earnout": float(bool(deal.get("has_earnout"))),
        "termination_fee_pct_of_ev": float(np.clip(fee / ev, 0, 0.2)) if fee and ev else np.nan,
        "mac_clause_intensity": math.log1p(deal.get("mac_clause_mentions") or 0),
        "is_sponsor": float(deal.get("acquirer_type") == "sponsor"),
        "acquirer_listed": float(bool(deal.get("acquirer_listed"))),
        "cross_border": float(bool(deal.get("cross_border"))),
        **{k: (macro.get(k) if macro.get(k) is not None else np.nan) for k in MACRO[:-1]},
        "filing_date_ordinal": float(date.fromisoformat(deal["filing_date"]).toordinal()),
    }


def deal_attributes(deals: list[dict]) -> dict:
    """Arrays used by relevance constructs, baselines and the ranker."""
    raw = [raw_numeric(d) for d in deals]
    log_rev = np.array([r["log_revenue"] for r in raw])
    growth = np.array([r["revenue_growth"] for r in raw])
    margin = np.array([r["ebitda_margin"] for r in raw])

    def z(x):
        mu, sd = np.nanmean(x), np.nanstd(x)
        return (x - mu) / (sd if sd > 0 else 1.0)

    return {
        "sic": np.array([d["sic"].zfill(4) for d in deals]),
        "sic2": np.array([d["sic"].zfill(4)[:2] for d in deals]),
        "sic3": np.array([d["sic"].zfill(4)[:3] for d in deals]),
        "date": np.array([d["filing_date"] for d in deals]),
        "date_ord": np.array([date.fromisoformat(d["filing_date"]).toordinal() for d in deals], dtype=float),
        "year": np.array([int(d["filing_date"][:4]) for d in deals]),
        "log_revenue": log_rev,
        "log_multiple": np.array([r["ev_to_revenue"] for r in raw]),
        "growth_z": z(growth),
        "margin_z": z(margin),
        "mix": np.array([d.get("consideration_mix") or "unknown" for d in deals]),
        "deal_type": np.array([d.get("deal_type") for d in deals]),
        "acquirer_type": np.array([d.get("acquirer_type") for d in deals]),
        "legal_flags": np.array([[bool((d.get("legal_flags") or {}).get(f)) for f in LEGAL_FLAG_NAMES] for d in deals]),
        "raw": raw,
    }


class NumericBlocks:
    """Fold-local median imputation and z-scoring of the numeric DealDNA blocks."""

    def __init__(self, deals: list[dict], drop: set = frozenset(), add_missing_indicators: bool = False,
                 imputation: str = "median"):
        raw = [raw_numeric(d) for d in deals]
        self.blocks = {b: [f for f in names if f not in drop] for b, names in NUMERIC_BLOCKS.items()}
        self.matrices = {b: np.array([[r[f] for f in names] for r in raw], dtype=float) for b, names in self.blocks.items()}
        self.add_missing_indicators = add_missing_indicators
        self.imputation = imputation

    def fit_transform(self, train_idx: np.ndarray) -> dict:
        out = {}
        for b, m in self.matrices.items():
            if m.shape[1] == 0:
                out[b] = np.zeros((m.shape[0], 1))
                continue
            train = m[train_idx]
            med = np.nanmedian(train, axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            missing = np.isnan(m)
            if self.imputation == "median":
                filled = np.where(missing, med, m)
            elif self.imputation == "zero_after_scaling":
                filled = np.where(missing, np.nan, m)
            else:
                raise ValueError(self.imputation)
            mu = np.nanmean(np.where(np.isnan(train), med, train), axis=0)
            sd = np.nanstd(np.where(np.isnan(train), med, train), axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            scaled = (filled - mu) / sd
            scaled = np.where(np.isnan(scaled), 0.0, scaled)
            if self.add_missing_indicators:
                cols = missing[:, missing[train_idx].any(axis=0)].astype(float)
                if cols.size:
                    scaled = np.hstack([scaled, cols - cols[train_idx].mean(axis=0)])
            out[b] = scaled
        return out


def cosine_gram(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normed = mat / np.where(norms == 0, 1.0, norms)
    return normed @ normed.T
