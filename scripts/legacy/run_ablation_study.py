"""Ablation + baseline study: does DealDNA's fuller embedding actually beat a naive industry+size
filter, and which blocks (financial/structure/macro/legal) contribute to retrieval quality?

Answers two of the paper's open questions directly:
  - "Does DealDNA beat naive industry+size filtering?" -> naive_industry_size_baseline row below.
  - "Does legal/macro signal add retrieval value over fundamentals alone?" -> the block-by-block rows.

All configurations are scored against the SAME multi-factor relevance labels (src/evaluation/
cross_validation.py: industry + size + consideration structure + recency), so the comparison is fair.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.dealdna.embedding_builder import build_raw_deal_record, assemble_embeddings
from src.retrieval.baseline import build_naive_baseline_embeddings
from src.retrieval.faiss_index import build_index
from src.evaluation.metrics import evaluate_retrieval, evaluate_within_industry_differentiation
from src.utils.logging_utils import get_logger

load_dotenv()
logger = get_logger("run_ablation_study")

DEAL_UNIVERSE_PATH = os.path.join("data", "processed", "deal_universe.json")
RESULTS_PATH = os.path.join("data", "processed", "ablation_and_baseline_results.json")

ABLATION_CONFIGS = [
    ("financial_only", ("financial",)),
    ("industry_only", ("industry",)),
    ("industry_plus_financial", ("industry", "financial")),
    ("industry_financial_structure", ("industry", "financial", "structure")),
    ("industry_financial_structure_macro", ("industry", "financial", "structure", "macro")),
    ("full_dealdna", ("industry", "financial", "structure", "macro", "legal")),
]


def load_deals() -> list[dict]:
    with open(DEAL_UNIVERSE_PATH) as f:
        return json.load(f)


def evaluate_embeddings(deals, embeddings) -> dict:
    index = build_index(embeddings.copy())
    return {
        **evaluate_retrieval(deals, embeddings, index),
        **evaluate_within_industry_differentiation(deals, embeddings, index),
    }


def main():
    deals = load_deals()
    if len(deals) < 5:
        logger.error("Not enough deals. Run scripts/build_deal_universe.py first.")
        return

    logger.info(f"Building raw DealDNA sub-vectors once for {len(deals)} deals (reused across all ablation configs)...")
    records = [build_raw_deal_record(d) for d in deals]

    results = {}
    for name, blocks in ABLATION_CONFIGS:
        logger.info(f"Evaluating config '{name}' (blocks={blocks})...")
        embeddings, feature_names, _ = assemble_embeddings(records, blocks=blocks)
        results[name] = {
            "blocks": list(blocks),
            "embedding_dim": int(embeddings.shape[1]),
            **evaluate_embeddings(deals, embeddings),
        }

    logger.info("Evaluating naive industry+size baseline...")
    baseline_embeddings = build_naive_baseline_embeddings(deals)
    results["naive_industry_size_baseline"] = {
        "blocks": ["sic_one_hot", "log_revenue"],
        "embedding_dim": int(baseline_embeddings.shape[1]),
        **evaluate_embeddings(deals, baseline_embeddings),
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {RESULTS_PATH}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
