"""Text-similarity baselines (TF-IDF, BM25, generic MiniLM sentence encoder) over the SAME legal-clause
passages DealDNA's FinBERT block consumes, evaluated against both relevance constructs:
  - the industry/size/structure/recency heuristic (build_relevance_labels)
  - the outcome-grounded realized-EV/Revenue-multiple construct (build_outcome_based_relevance)

Purpose: show whether the domain-adapted contextual embedding (FinBERT) of the selected passages beats
a plain lexical model or a general-purpose encoder on those passages -- the comparison the naive
industry+size baseline can't provide. Complements run_ablation_study.py and run_outcome_grounded_study.py.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.evaluation.cross_validation import build_outcome_based_relevance, build_relevance_labels
from src.evaluation.metrics import evaluate_similarity_matrix
from src.retrieval.text_baselines import TEXT_BASELINES
from src.utils.logging_utils import get_logger

load_dotenv()
logger = get_logger("run_text_baseline_study")

DEAL_UNIVERSE_PATH = os.path.join("data", "processed", "deal_universe.json")
RESULTS_PATH = os.path.join("data", "processed", "text_baseline_results.json")


def main():
    with open(DEAL_UNIVERSE_PATH) as f:
        deals = json.load(f)
    logger.info(f"{len(deals)} deals loaded")

    heuristic_rel = build_relevance_labels(deals)
    outcome_rel = build_outcome_based_relevance(deals)
    n_with_passages = sum(1 for d in deals if d.get("clause_passages"))
    logger.info(f"{n_with_passages}/{len(deals)} deals have >=1 selected legal-clause passage")

    results = {"n_deals": len(deals), "n_deals_with_clause_passages": n_with_passages, "baselines": {}}
    for name, fn in TEXT_BASELINES.items():
        logger.info(f"Building '{name}' similarity matrix...")
        sim = fn(deals)
        results["baselines"][name] = {
            "heuristic_relevance": evaluate_similarity_matrix(deals, sim, relevance=heuristic_rel),
            "outcome_grounded_relevance": evaluate_similarity_matrix(deals, sim, relevance=outcome_rel),
        }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {RESULTS_PATH}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
