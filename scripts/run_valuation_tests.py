"""Paired significance of k-NN valuation vs. the causal industry-median predictor, at the PRE-SPECIFIED
configuration (top-8 neighbours, rank-weighted mean: the submitted paper's setting), so no (k, aggregator)
choice is made on the evaluation data. Error = |predicted - realized| in log(1 + EV/Revenue) space;
lower is better, so a positive difference (industry median minus retriever) favours the retriever.
Holm correction across the retrievers. Output: results/valuation_tests.json
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_valuation_study import valuation_objective_neighbors
from src.evaluation.engine import causal_mask
from src.evaluation.features import deal_attributes
from src.evaluation.stats import effect_label, holm, paired

RESULTS = os.environ.get("DTX_RESULTS", "results")
K = 8


def main():
    deals = json.load(open(os.path.join("data", "processed", "deal_universe.json")))
    attrs = deal_attributes(deals)
    biz = np.load(os.path.join("data", "embeddings", "business_mpnet.npy"))
    neighbors = json.load(open(os.path.join(RESULTS, "retrieval", "neighbors_outcome_sic.json")))
    path = os.path.join(RESULTS, "retrieval", "neighbors_valuation_objective.json")
    if not os.path.exists(path):
        json.dump(valuation_objective_neighbors(deals, attrs, biz), open(path, "w"))
    neighbors["valuation_objective"] = json.load(open(path))

    y = np.log1p(np.exp(attrs["log_multiple"]))
    has = ~np.isnan(attrs["log_multiple"])
    mask = causal_mask(attrs["date_ord"])
    queries = sorted(set.intersection(*[{int(q) for q, v in t.items() if len(v["idx"]) >= 3} for t in neighbors.values()])
                     & set(np.flatnonzero(has).tolist()))
    w = 1.0 / np.arange(1, K + 1)

    def knn_err(table):
        out = {}
        for q in queries:
            idx = table[str(q)]["idx"][:K]
            out[q] = abs(float(np.sum(w[: len(idx)] * y[idx]) / w[: len(idx)].sum()) - y[q])
        return out

    ind_err = {}
    for q in queries:
        earlier = np.flatnonzero(mask[q] & has)
        same = earlier[attrs["sic2"][earlier] == attrs["sic2"][q]]
        ind_err[q] = abs(float(np.median(y[same] if same.size else y[earlier])) - y[q])

    clusters = np.array([f"{s}_{yr}" for s, yr in zip(attrs["sic2"], attrs["year"])])
    tests = {r: paired(ind_err, knn_err(t), clusters) for r, t in neighbors.items()}
    adjusted = holm({r: t["p_cluster_bootstrap"] for r, t in tests.items()})
    for r in tests:
        tests[r]["p_holm"] = adjusted[r]
        tests[r]["effect"] = effect_label(tests[r]["cliffs_delta"])
        tests[r]["mean_abs_log_error_retriever"] = tests[r].pop("mean_b")
        tests[r]["mean_abs_log_error_industry_median"] = tests[r].pop("mean_a")
    json.dump({"k": K, "aggregator": "rank_weighted_mean", "n": len(queries), "tests": tests},
              open(os.path.join(RESULTS, "valuation_tests.json"), "w"), indent=1)
    for r, t in tests.items():
        print(f"{r:20s} err {t['mean_abs_log_error_retriever']:.4f} vs ind.median {t['mean_abs_log_error_industry_median']:.4f} "
              f"gain {t['diff']:+.4f} holm p={t['p_holm']:.3g} delta={t['cliffs_delta']:+.3f}")


if __name__ == "__main__":
    main()
