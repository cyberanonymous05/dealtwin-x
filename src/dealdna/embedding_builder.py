"""Fit a DealDNA embedding space across a corpus of transactions: build sub-vectors, impute/scale the numeric
blocks corpus-wide, one-hot encode 2-digit SIC industry group, and concatenate with L2-normalized legal-clause
embeddings (down-weighted so ~20 numeric dims aren't swamped by 768 FinBERT dims in cosine similarity).

The industry block exists because retrieval evaluation (see src/evaluation/cross_validation.py) gates
relevance on industry match, mirroring how a banker actually screens comps -- industry first, then
differentiate on financials/structure/legal/macro. Without an explicit industry signal, DealDNA has no way
to satisfy that gate except by accident, which is a real design gap a naive industry+size baseline doesn't
have (see src/retrieval/baseline.py) -- this block closes it.

assemble_embeddings() takes a `blocks` subset so the same raw per-deal sub-vectors (expensive FinBERT pass
included) can be reassembled into industry-only / +financial / +structure / +macro / +legal configurations
for the ablation study, without recomputing legal embeddings for every configuration.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.dealdna.financial_features import FINANCIAL_FEATURE_NAMES, build_financial_feature_vector
from src.dealdna.transaction_structure import STRUCTURE_FEATURE_NAMES, build_transaction_structure_vector
from src.dealdna.macro_features import MACRO_FEATURE_NAMES, build_macro_feature_vector
from src.dealdna.legal_embeddings import embed_legal_clauses

LEGAL_BLOCK_WEIGHT = 0.35
NUMERIC_BLOCK_WEIGHT = 1.0
INDUSTRY_BLOCK_WEIGHT = 1.5

ALL_BLOCKS = ("industry", "financial", "structure", "macro", "legal")
_NUMERIC_BLOCK_FEATURE_NAMES = {
    "financial": FINANCIAL_FEATURE_NAMES,
    "structure": STRUCTURE_FEATURE_NAMES,
    "macro": MACRO_FEATURE_NAMES,
}


def build_raw_deal_record(deal: dict) -> dict:
    return {
        "deal_id": deal["deal_id"],
        "sic_group": deal["sic"][:2],
        "financial_vec": build_financial_feature_vector(deal),
        "structure_vec": build_transaction_structure_vector(deal),
        "macro_vec": build_macro_feature_vector(deal),
        "legal_vec": embed_legal_clauses(deal.get("clause_passages", [])),
    }


def assemble_embeddings(records: list[dict], blocks: tuple = ALL_BLOCKS):
    """Assemble DealDNA embeddings from precomputed raw sub-vectors, including only the given blocks.
    Returns (embeddings, feature_names, fitted_transforms)."""
    numeric_blocks = [b for b in ("financial", "structure", "macro") if b in blocks]
    feature_names = []
    embedding_parts = []
    fitted = {"numeric_feature_names": []}

    if "industry" in blocks:
        sic_groups = sorted({r["sic_group"] for r in records})
        group_index = {g: i for i, g in enumerate(sic_groups)}
        industry_onehot = np.zeros((len(records), len(sic_groups)), dtype=float)
        for i, r in enumerate(records):
            industry_onehot[i, group_index[r["sic_group"]]] = 1.0
        embedding_parts.append(industry_onehot * INDUSTRY_BLOCK_WEIGHT)
        feature_names += [f"industry_sic2_{g}" for g in sic_groups]
        fitted["industry_sic_groups"] = sic_groups

    if numeric_blocks:
        numeric_matrix = np.hstack([np.vstack([r[f"{b}_vec"] for r in records]) for b in numeric_blocks])
        imputer = SimpleImputer(strategy="median")
        numeric_imputed = imputer.fit_transform(numeric_matrix)
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric_imputed)

        numeric_feature_names = [name for b in numeric_blocks for name in _NUMERIC_BLOCK_FEATURE_NAMES[b]]
        feature_names += numeric_feature_names
        embedding_parts.append(numeric_scaled * NUMERIC_BLOCK_WEIGHT)
        fitted.update(imputer=imputer, scaler=scaler, numeric_matrix_raw=numeric_matrix, numeric_feature_names=numeric_feature_names)

    if "legal" in blocks:
        legal_matrix = np.vstack([r["legal_vec"] for r in records])
        legal_norms = np.linalg.norm(legal_matrix, axis=1, keepdims=True)
        legal_normalized = legal_matrix / np.where(legal_norms == 0, 1.0, legal_norms)
        embedding_parts.append(legal_normalized * LEGAL_BLOCK_WEIGHT)
        feature_names += [f"legal_dim_{i}" for i in range(legal_matrix.shape[1])]

    if not embedding_parts:
        raise ValueError("assemble_embeddings requires at least one block")

    embeddings = np.hstack(embedding_parts).astype("float32")
    return embeddings, feature_names, fitted


def fit_transform_corpus(deals: list[dict]):
    """Build full DealDNA embeddings (all five blocks) for an entire deal corpus."""
    records = [build_raw_deal_record(d) for d in deals]
    return assemble_embeddings(records, blocks=ALL_BLOCKS)


def transform_new_deal(deal: dict, fitted: dict) -> np.ndarray:
    """Embed a single new deal using transforms already fit on a reference corpus (for counterfactual queries).
    A SIC 2-digit group never seen in the reference corpus gets an all-zero industry one-hot (no match)."""
    record = build_raw_deal_record(deal)

    embedding_parts = []
    if "industry_sic_groups" in fitted:
        sic_groups = fitted["industry_sic_groups"]
        onehot = np.zeros(len(sic_groups), dtype=float)
        if record["sic_group"] in sic_groups:
            onehot[sic_groups.index(record["sic_group"])] = 1.0
        embedding_parts.append(onehot * INDUSTRY_BLOCK_WEIGHT)

    numeric_vec = np.hstack([record["financial_vec"], record["structure_vec"], record["macro_vec"]]).reshape(1, -1)
    numeric_imputed = fitted["imputer"].transform(numeric_vec)
    numeric_scaled = fitted["scaler"].transform(numeric_imputed)
    embedding_parts.append(numeric_scaled.flatten() * NUMERIC_BLOCK_WEIGHT)

    legal_vec = record["legal_vec"]
    legal_norm = np.linalg.norm(legal_vec)
    legal_normalized = legal_vec / legal_norm if legal_norm > 0 else legal_vec
    embedding_parts.append(legal_normalized * LEGAL_BLOCK_WEIGHT)

    return np.hstack(embedding_parts).astype("float32")
