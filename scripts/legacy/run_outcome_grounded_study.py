"""The decisive test: is DealDNA's fuller feature set actually worth something, when relevance is
grounded in REAL financial outcomes (realized EV/Revenue multiple similarity) instead of a hand-built
heuristic -- and when the multiple itself is excluded from the embedding to avoid circularity?

Why this matters: the earlier industry/size/structure/recency heuristic directly mirrors what the naive
industry+size baseline encodes, so that comparison couldn't help but favor the baseline. Grounding
relevance in "did these two deals actually get paid similar multiples" is the real premise of precedent
transaction analysis, and isn't something the naive baseline is trivially built to satisfy -- similar
industry and size do NOT guarantee similar realized multiples, so this is a genuinely harder, more
meaningful test of whether structure/legal/macro signal helps predict which deals were priced alike.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dotenv import load_dotenv

from src.dealdna.embedding_builder import build_raw_deal_record
from src.dealdna.financial_features import FINANCIAL_FEATURE_NAMES
from src.evaluation.cross_validation import build_outcome_based_relevance
from src.evaluation.metrics import evaluate_retrieval, evaluate_within_industry_differentiation
from src.retrieval.baseline import build_naive_baseline_embeddings
from src.retrieval.faiss_index import build_index
from src.retrieval.metric_learning import evaluate_learned_metric
from src.utils.logging_utils import get_logger

load_dotenv()
logger = get_logger("run_outcome_grounded_study")

DEAL_UNIVERSE_PATH = os.path.join("data", "processed", "deal_universe.json")
RESULTS_PATH = os.path.join("data", "processed", "outcome_grounded_results.json")

ALL_BLOCKS = ["industry", "financial", "structure", "macro", "legal"]

# these directly encode the outcome relevance is built from -- must be excluded or the test is circular
_LEAKY_MULTIPLE_INDICES = [FINANCIAL_FEATURE_NAMES.index(f) for f in ("ev_to_revenue", "ev_to_ebitda")]


def load_deals() -> list[dict]:
    with open(DEAL_UNIVERSE_PATH) as f:
        return json.load(f)


def strip_multiples(records: list[dict]) -> list[dict]:
    """Return copies of records with ev_to_revenue/ev_to_ebitda zeroed out of the financial block."""
    stripped = []
    for r in records:
        r2 = dict(r)
        fv = r["financial_vec"].copy()
        fv[_LEAKY_MULTIPLE_INDICES] = np.nan
        r2["financial_vec"] = fv
        stripped.append(r2)
    return stripped


def main():
    deals = load_deals()
    if len(deals) < 5:
        logger.error("Not enough deals. Run scripts/build_deal_universe.py first.")
        return

    logger.info("Building outcome-grounded relevance labels (realized EV/Revenue multiple similarity, industry-gated)...")
    relevance = build_outcome_based_relevance(deals)
    n_scorable = int((relevance.sum(axis=1) > 0).sum())
    logger.info(f"{n_scorable}/{len(deals)} deals have at least one same-industry peer with a comparable multiple")

    logger.info("Building raw DealDNA sub-vectors and stripping valuation multiples to avoid circularity...")
    records = strip_multiples([build_raw_deal_record(d) for d in deals])

    logger.info("Evaluating naive industry+size baseline against outcome-grounded relevance...")
    baseline_embeddings = build_naive_baseline_embeddings(deals)
    baseline_index = build_index(baseline_embeddings.copy())
    baseline_results = {
        **evaluate_retrieval(deals, baseline_embeddings, baseline_index, relevance=relevance),
        **evaluate_within_industry_differentiation(deals, baseline_embeddings, baseline_index, relevance=relevance),
    }

    logger.info("Running nested-CV learned-metric DealDNA (multiples excluded) against outcome-grounded relevance...")
    dealdna_results = evaluate_learned_metric(deals, records, ALL_BLOCKS, n_splits=5, relevance=relevance)

    results = {
        "n_deals_with_scorable_multiple": n_scorable,
        "naive_industry_size_baseline": baseline_results,
        "full_dealdna_learned_weights_no_multiples": dealdna_results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {RESULTS_PATH}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
