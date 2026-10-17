"""End-to-end DealTwin-X pipeline over the discovered deal universe:
build DealDNA embeddings -> FAISS index -> counterfactual precedent retrieval demo ->
SHAP-explained valuation-multiple surrogate -> retrieval evaluation (NDCG@5, MAP, MRR, P@5, R@10).

Run scripts/build_deal_universe.py first to populate data/processed/deal_universe.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dotenv import load_dotenv

from src.dealdna.embedding_builder import fit_transform_corpus
from src.retrieval.faiss_index import build_index
from src.retrieval.search import find_precedent_transactions
from src.explainability.shap_explainer import train_valuation_surrogate, explain_single_deal
from src.evaluation.metrics import evaluate_retrieval
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

load_dotenv()
logger = get_logger("run_pipeline")

RETRIEVAL_TOP_K = load_config()["retrieval"]["top_k"]
DEAL_UNIVERSE_PATH = os.path.join("data", "processed", "deal_universe.json")
EMBEDDINGS_PATH = os.path.join("data", "embeddings", "dealdna_embeddings.npy")
DEAL_IDS_PATH = os.path.join("data", "embeddings", "deal_ids.json")
RESULTS_PATH = os.path.join("data", "processed", "pipeline_results.json")


def load_deals() -> list[dict]:
    with open(DEAL_UNIVERSE_PATH) as f:
        return json.load(f)


def build_embeddings(deals: list[dict]):
    logger.info(f"Building DealDNA embeddings for {len(deals)} deals (financial + structure + macro + FinBERT legal)...")
    embeddings, feature_names, fitted = fit_transform_corpus(deals)
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(DEAL_IDS_PATH, "w") as f:
        json.dump([d["deal_id"] for d in deals], f)
    logger.info(f"Embeddings shape: {embeddings.shape} ({len(feature_names)} feature dims)")
    return embeddings, feature_names, fitted


def demo_counterfactual_retrieval(deals, embeddings, index, query_idx: int = 0, top_k: int = 5):
    query_deal = deals[query_idx]
    deal_id_lookup = [d["deal_id"] for d in deals]
    deals_by_id = {d["deal_id"]: d for d in deals}

    hits = find_precedent_transactions(
        index, deal_id_lookup, embeddings[query_idx].copy(), top_k=top_k, exclude_deal_id=query_deal["deal_id"]
    )
    results = []
    for hit in hits:
        candidate = deals_by_id[hit["deal_id"]]
        results.append(
            {
                "company_name": candidate["company_name"],
                "similarity_score": round(hit["similarity_score"], 4),
                "sic": candidate["sic"],
                "consideration_mix": candidate.get("consideration_mix"),
                "ev_to_revenue": (
                    round(candidate["ev_estimate"] / candidate["revenues"], 2)
                    if candidate.get("ev_estimate") and candidate.get("revenues")
                    else None
                ),
                "filing_date": candidate["filing_date"],
            }
        )
    return {
        "query_company": query_deal["company_name"],
        "query_sic": query_deal["sic"],
        "query_ev_to_revenue": (
            round(query_deal["ev_estimate"] / query_deal["revenues"], 2)
            if query_deal.get("ev_estimate") and query_deal.get("revenues")
            else None
        ),
        "retrieved_precedents": results,
    }


def main():
    deals = load_deals()
    if len(deals) < 5:
        logger.error("Not enough deals in deal universe. Run scripts/build_deal_universe.py first.")
        return

    embeddings, feature_names, fitted = build_embeddings(deals)
    index = build_index(embeddings.copy())

    logger.info("Running counterfactual precedent retrieval demo...")
    query_idx = max(range(len(deals)), key=lambda i: deals[i]["filing_date"])
    retrieval_demo = demo_counterfactual_retrieval(deals, embeddings, index, query_idx=query_idx, top_k=RETRIEVAL_TOP_K)

    logger.info("Training XGBoost valuation-multiple surrogate with deal-level GroupKFold CV + SHAP...")
    surrogate_summary, model, explainer, num_feature_names, valid_idx = train_valuation_surrogate(deals, fitted)
    single_explanation = explain_single_deal(deals, fitted, model, explainer, num_feature_names, valid_idx, query_idx)

    logger.info("Evaluating retrieval quality (NDCG@5, MAP, MRR, Precision@5, Recall@10)...")
    eval_summary = evaluate_retrieval(deals, embeddings, index)

    results = {
        "n_deals": len(deals),
        "embedding_dim": int(embeddings.shape[1]),
        "counterfactual_retrieval_demo": retrieval_demo,
        "valuation_surrogate": surrogate_summary,
        "explained_query_deal": single_explanation,
        "retrieval_evaluation": eval_summary,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {RESULTS_PATH}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
