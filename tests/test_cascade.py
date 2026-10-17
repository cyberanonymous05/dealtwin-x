"""The cascade ranker may use the realized multiples of earlier (public) candidate deals, but never the
query's own realized multiple."""

import numpy as np

from src.evaluation.features import VALUE_DERIVED, NumericBlocks, deal_attributes
from src.evaluation.relevance import build_relevance
from src.retrieval import retrievers as R


def _deals(n=120, seed=0):
    rng = np.random.default_rng(seed)
    deals = []
    for i in range(n):
        rev = float(np.exp(rng.normal(19, 1.5)))
        deals.append({
            "sic": rng.choice(["2834", "2836", "7372", "7371", "6022"]),
            "filing_date": f"{2013 + i // 12}-{1 + i % 12:02d}-15",
            "revenues": rev, "ev_estimate": rev * float(np.exp(rng.normal(1, 0.8))),
            "revenue_growth": float(rng.normal(0.1, 0.2)), "ebitda_margin": float(rng.normal(0.1, 0.1)),
            "net_income": rev * float(rng.normal(0.05, 0.1)), "total_assets": rev * 2,
            "consideration_mix": "cash", "deal_type": "merger", "acquirer_type": "strategic",
        })
    return deals


def test_cascade_ranking_ignores_the_querys_own_multiple():
    deals = _deals()
    attrs = deal_attributes(deals)
    rng = np.random.default_rng(1)
    n = len(deals)
    sims = {"legal": np.eye(n), "business": np.eye(n)}
    ctx = R.Context(deals, attrs, build_relevance("outcome", attrs), NumericBlocks(deals, drop=VALUE_DERIVED), sims,
                    candidate_pool=~np.isnan(attrs["log_multiple"]), business_emb=rng.normal(size=(n, 16)))
    train, test = np.arange(90), np.arange(90, n)
    params = dict(n_estimators=40, deterministic=True, force_row_wise=True, n_jobs=1)
    q = n - 1                                   # latest deal: it is no other query's candidate
    before, _ = R.lambdamart_cascade(ctx, top_n=20, params=params)(train, test)
    attrs["log_multiple"][q] += 3.0             # change the query's own realized multiple
    deals[q]["ev_estimate"] *= np.e ** 3
    after, _ = R.lambdamart_cascade(ctx, top_n=20, params=params)(train, test)
    np.testing.assert_allclose(before[-1], after[-1])
