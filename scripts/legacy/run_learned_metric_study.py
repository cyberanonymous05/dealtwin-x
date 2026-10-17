"""Nested cross-validated evaluation of DealDNA with LEARNED block weights (industry/financial/structure/
macro/legal), replacing the hand-picked constants used in the main pipeline/ablation study, against the
naive industry+size baseline.

This is the fair version of the baseline comparison: instead of asking "does DealDNA with arbitrary
weights beat naive industry+size filtering," it asks "does DealDNA's fuller feature set beat naive
industry+size filtering when each block's contribution is learned from data (out-of-fold, so it's not
circular with the reported metric)."
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.dealdna.embedding_builder import build_raw_deal_record
from src.retrieval.metric_learning import evaluate_learned_metric
from src.utils.logging_utils import get_logger

load_dotenv()
logger = get_logger("run_learned_metric_study")

DEAL_UNIVERSE_PATH = os.path.join("data", "processed", "deal_universe.json")
RESULTS_PATH = os.path.join("data", "processed", "learned_metric_results.json")

ALL_BLOCKS = ["industry", "financial", "structure", "macro", "legal"]


def load_deals() -> list[dict]:
    with open(DEAL_UNIVERSE_PATH) as f:
        return json.load(f)


def main():
    deals = load_deals()
    if len(deals) < 5:
        logger.error("Not enough deals. Run scripts/build_deal_universe.py first.")
        return

    logger.info(f"Building raw DealDNA sub-vectors once for {len(deals)} deals...")
    records = [build_raw_deal_record(d) for d in deals]

    logger.info("Running nested-CV learned-metric evaluation (5 folds, grouped by 2-digit SIC)...")
    results = evaluate_learned_metric(deals, records, ALL_BLOCKS, n_splits=5)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {RESULTS_PATH}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
