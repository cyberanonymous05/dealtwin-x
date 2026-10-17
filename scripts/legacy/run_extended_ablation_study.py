"""Extended ablation beyond the additive progression in run_ablation_study.py:

  1. Leave-one-block-out: start from full DealDNA, remove each block in turn, measure the drop. Directly
     isolates each block's marginal contribution (the additive study confounds order with contribution).
  2. Rank-cutoff sweep: NDCG@{1,3,5,10} for full DealDNA vs. the naive baseline, so the comparison
     isn't hostage to a single k.
  3. Per-2-digit-SIC breakdown: within-industry NDCG@5 for full DealDNA minus the naive baseline, per
     industry group with >= 5 deals -- shows where the multi-modal signal helps and where it hurts.

Hand-weighted blocks (same constants as run_ablation_study.py), scored against the heuristic relevance.
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dotenv import load_dotenv

from src.dealdna.embedding_builder import ALL_BLOCKS, assemble_embeddings, build_raw_deal_record
from src.evaluation.cross_validation import build_relevance_labels
from src.evaluation.metrics import ndcg_at_k
from src.retrieval.baseline import build_naive_baseline_embeddings
from src.retrieval.faiss_index import build_index, search
from src.utils.logging_utils import get_logger

load_dotenv()
logger = get_logger("run_extended_ablation_study")

DEAL_UNIVERSE_PATH = os.path.join("data", "processed", "deal_universe.json")
RESULTS_PATH = os.path.join("data", "processed", "extended_ablation_results.json")


def _rank_matrix(embeddings):
    index = build_index(embeddings.copy())
    n = len(embeddings)
    orders = np.zeros((n, n), dtype=int)
    for i in range(n):
        _, idx = search(index, embeddings[i].copy(), top_k=n)
        orders[i] = idx
    return orders


def ndcg_sweep(deals, embeddings, relevance, ks=(1, 3, 5, 10)):
    orders = _rank_matrix(embeddings)
    out = {f"ndcg@{k}": [] for k in ks}
    for q in range(len(deals)):
        if (relevance[q] > 0).sum() == 0:
            continue
        ranked = [relevance[q, j] for j in orders[q] if j != q and j != -1]
        for k in ks:
            out[f"ndcg@{k}"].append(ndcg_at_k(ranked, k=k))
    return {m: round(float(np.mean(v)), 4) if v else None for m, v in out.items()}


def within_industry_ndcg5_per_group(deals, embeddings, relevance, min_deals=5):
    orders = _rank_matrix(embeddings)
    per_group = defaultdict(list)
    for q in range(len(deals)):
        if (relevance[q] > 0).sum() < 3:
            continue
        g = deals[q]["sic"][:2]
        within = [relevance[q, j] for j in orders[q] if j != q and j != -1 and relevance[q, j] > 0]
        per_group[g].append(ndcg_at_k(within, k=5))
    counts = defaultdict(int)
    for d in deals:
        counts[d["sic"][:2]] += 1
    return {
        g: round(float(np.mean(v)), 4)
        for g, v in per_group.items()
        if counts[g] >= min_deals and v
    }


def main():
    with open(DEAL_UNIVERSE_PATH) as f:
        deals = json.load(f)
    logger.info(f"{len(deals)} deals")

    relevance = build_relevance_labels(deals)
    records = [build_raw_deal_record(d) for d in deals]

    full_emb, _, _ = assemble_embeddings(records, blocks=ALL_BLOCKS)
    baseline_emb = build_naive_baseline_embeddings(deals)

    logger.info("Leave-one-block-out from full DealDNA...")
    full_sweep = ndcg_sweep(deals, full_emb, relevance)
    leave_one_out = {"full_dealdna": full_sweep}
    for block in ALL_BLOCKS:
        kept = tuple(b for b in ALL_BLOCKS if b != block)
        emb, _, _ = assemble_embeddings(records, blocks=kept)
        s = ndcg_sweep(deals, emb, relevance)
        leave_one_out[f"minus_{block}"] = {
            **s,
            "delta_ndcg@5_vs_full": round((s["ndcg@5"] or 0) - (full_sweep["ndcg@5"] or 0), 4),
        }

    logger.info("Rank-cutoff sweep: full DealDNA vs. naive baseline...")
    ndcg_cutoff_sweep = {
        "full_dealdna": full_sweep,
        "naive_industry_size_baseline": ndcg_sweep(deals, baseline_emb, relevance),
    }

    logger.info("Per-industry within-NDCG@5 (full DealDNA minus naive baseline)...")
    dna_by_group = within_industry_ndcg5_per_group(deals, full_emb, relevance)
    base_by_group = within_industry_ndcg5_per_group(deals, baseline_emb, relevance)
    per_industry = {
        g: {
            "full_dealdna": dna_by_group[g],
            "naive_baseline": base_by_group.get(g),
            "delta": round(dna_by_group[g] - base_by_group[g], 4) if g in base_by_group else None,
        }
        for g in sorted(dna_by_group)
    }

    results = {
        "n_deals": len(deals),
        "leave_one_block_out": leave_one_out,
        "ndcg_cutoff_sweep": ndcg_cutoff_sweep,
        "per_industry_within_ndcg5": per_industry,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved {RESULTS_PATH}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
