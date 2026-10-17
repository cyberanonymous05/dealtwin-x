"""High-level counterfactual precedent search: given a target company/deal embedding, retrieve and rank
comparable historical transactions, optionally excluding the query deal itself (when the query is already
a member of the indexed corpus, e.g. a retrospective "what would have justified deal X's multiple" demo)."""

from src.retrieval.faiss_index import search


def find_precedent_transactions(
    index, deal_id_lookup: list[str], query_embedding, top_k: int = 10, exclude_deal_id: str | None = None
) -> list[dict]:
    fetch_k = top_k + 1 if exclude_deal_id is not None else top_k
    scores, indices = search(index, query_embedding, top_k=fetch_k)
    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        deal_id = deal_id_lookup[idx]
        if deal_id == exclude_deal_id:
            continue
        results.append({"deal_id": deal_id, "similarity_score": float(score)})
        if len(results) == top_k:
            break
    return results
