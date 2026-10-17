"""Significance, practical significance and subgroup analyses over the stored per-query retrieval results.

Family for the Holm correction: EVERY comparison reported -- each non-naive retriever vs. the naive
baseline, on every tested metric (including within-industry metrics), under every relevance construct,
plus the LambdaMART-vs-strongest-baseline comparisons and the multiple-dispersion split. p-values are
from the (SIC2 x filing-year) cluster bootstrap; per-deal Wilcoxon p-values are reported alongside.

Output: results/significance.json
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.engine import TESTED_METRICS
from src.evaluation.features import deal_attributes
from src.evaluation.relevance import CONSTRUCTS
from src.evaluation.stats import effect_label, holm, paired

RESULTS = os.environ.get("DTX_RESULTS", "results")
RUNS = os.path.join(RESULTS, "retrieval")
COMPARED = ("sic_backoff", "size_bins", "text_peers", "sic2_text", "linear", "lambdamart", "lambdamart_cascade")
PROPOSED = ("lambdamart", "lambdamart_cascade")
BASELINES = ("naive", "sic_backoff", "size_bins", "text_peers", "sic2_text")


def load(name):
    path = os.path.join(RUNS, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        run = json.load(f)
    run["per_query"] = {m: {int(k): v for k, v in d.items()} for m, d in run["per_query"].items()}
    return run


def main():
    with open(os.path.join("data", "processed", "deal_universe.json")) as f:
        deals = json.load(f)
    attrs = deal_attributes(deals)
    clusters = np.array([f"{s}_{y}" for s, y in zip(attrs["sic2"], attrs["year"])])

    tests, pvals = {}, {}

    def add(key, a, b, metric):
        if a is None or b is None or not a["per_query"].get(metric) or not b["per_query"].get(metric):
            return
        res = paired(a["per_query"][metric], b["per_query"][metric], clusters)
        tests[key] = res
        pvals[key] = res["p_cluster_bootstrap"]

    best_baseline = {}
    for construct in CONSTRUCTS:
        naive = load(f"main__{construct}__sic__naive")
        for r in COMPARED:
            run = load(f"main__{construct}__sic__{r}")
            for m in TESTED_METRICS:
                add(f"{construct}|{r}_vs_naive|{m}", run, naive, m)
        for r in ("linear", "lambdamart"):
            run = load(f"original__{construct}__sic__{r}")
            for m in TESTED_METRICS:
                add(f"{construct}|{r}_original_vs_naive|{m}", run, naive, m)
        lm_run = load(f"main__{construct}__sic__lambdamart")
        cascade = load(f"main__{construct}__sic__lambdamart_cascade")
        for name in PROPOSED:
            prop = load(f"main__{construct}__sic__{name}")
            for base in BASELINES[1:]:
                base_run = load(f"main__{construct}__sic__{base}")
                for m in TESTED_METRICS:
                    add(f"{construct}|{name}_vs_{base}|{m}", prop, base_run, m)
        orig = load(f"original__{construct}__sic__lambdamart")
        for m in TESTED_METRICS:
            add(f"{construct}|lambdamart_vs_lambdamart_original|{m}", lm_run, orig, m)
            add(f"{construct}|lambdamart_cascade_vs_lambdamart|{m}", cascade, lm_run, m)
            add(f"{construct}|lambdamart_cascade_vs_lambdamart_original|{m}", cascade, orig, m)
        # The same headline comparisons under expanding-window temporal folds.
        for name in PROPOSED:
            for m in TESTED_METRICS:
                add(f"{construct}|time|{name}_vs_sic_backoff|{m}", load(f"main__{construct}__time__{name}"),
                    load(f"main__{construct}__time__sic_backoff"), m)
        # Strongest baseline under this construct = best mean MAP among the non-learned retrievers.
        scored = {b: load(f"main__{construct}__sic__{b}") for b in BASELINES}
        scored = {b: r for b, r in scored.items() if r is not None}
        if scored:
            # strongest = best mean graded within-industry ranking (binary metrics only reflect industry gating)
            best = max(scored, key=lambda b: scored[b]["summary"]["wi_ndcg@5"] or 0)
            best_baseline[construct] = best
            for name in PROPOSED:
                prop = load(f"main__{construct}__sic__{name}")
                for m in TESTED_METRICS:
                    add(f"{construct}|{name}_vs_best_baseline({best})|{m}", prop, scored[best], m)

    # Block ablations (full LambdaMART vs. LambdaMART without one feature group), outcome and legal-aware.
    for construct in ("outcome", "legal"):
        full = load(f"main__{construct}__sic__lambdamart")
        for g in ("industry", "financial", "structure", "acquirer", "macro", "legal", "business"):
            ablated = load(f"ablation__{construct}__sic__lambdamart_minus_{g}")
            for m in ("ndcg@5", "wi_ndcg@5"):
                add(f"{construct}|lambdamart_vs_minus_{g}|{m}", full, ablated, m)

    cascade_full = load("main__outcome__sic__lambdamart_cascade")
    cascade_abl = load("ablation__outcome__sic__lambdamart_cascade_minus_multiples")
    for m in ("ndcg@5", "wi_ndcg@5"):
        add(f"outcome|lambdamart_cascade_vs_minus_multiples|{m}", cascade_full, cascade_abl, m)
        add(f"outcome|cascade_minus_multiples_vs_sic_backoff|{m}", cascade_abl, load("main__outcome__sic__sic_backoff"), m)

    # Multiple-dispersion split (outcome construct): industries whose realized log-multiple dispersion is
    # above vs. below the median across industries with >= 5 multiples.
    lm, naive = load("main__outcome__sic__lambdamart"), load("main__outcome__sic__naive")
    dispersion = {}
    if lm and naive:
        m = attrs["log_multiple"]
        sd = {g: np.nanstd(m[attrs["sic2"] == g]) for g in np.unique(attrs["sic2"]) if np.sum(~np.isnan(m[attrs["sic2"] == g])) >= 5}
        cut = np.median(list(sd.values()))
        for side, keep in (("dispersed", lambda g: sd.get(g, -1) > cut), ("clustered", lambda g: 0 <= sd.get(g, -1) <= cut)):
            for metric in TESTED_METRICS:
                a = {q: v for q, v in lm["per_query"][metric].items() if keep(attrs["sic2"][q])}
                b = {q: v for q, v in naive["per_query"][metric].items() if keep(attrs["sic2"][q])}
                if a and b:
                    res = paired(a, b, clusters)
                    tests[f"outcome|lambdamart_vs_naive|{metric}|{side}"] = res
                    pvals[f"outcome|lambdamart_vs_naive|{metric}|{side}"] = res["p_cluster_bootstrap"]
        dispersion = {"median_industry_log_multiple_sd": float(cut), "n_industries": len(sd)}

    adjusted = holm(pvals)
    for k in tests:
        tests[k]["p_holm"] = adjusted[k]
        tests[k]["effect"] = effect_label(tests[k]["cliffs_delta"])

    # Practical significance: extra relevant precedents in a top-10 list.
    practical = {}
    for construct in CONSTRUCTS:
        lmr, nv = load(f"main__{construct}__sic__lambdamart"), load(f"main__{construct}__sic__naive")
        if lmr and nv and lmr["per_query"].get("hits@10"):
            qs = sorted(set(lmr["per_query"]["hits@10"]) & set(nv["per_query"]["hits@10"]))
            practical[construct] = {
                "lambdamart_hits@10": float(np.mean([lmr["per_query"]["hits@10"][q] for q in qs])),
                "naive_hits@10": float(np.mean([nv["per_query"]["hits@10"][q] for q in qs])),
                "map_relative_gain": (lmr["summary"]["map"] - nv["summary"]["map"]) / nv["summary"]["map"],
            }

    # Subgroups (outcome construct): LambdaMART vs naive within deal type / acquirer type / status.
    subgroups = {}
    if lm and naive:
        status = np.array([d.get("completion_status") for d in deals])
        for label, arr in (("deal_type", attrs["deal_type"]), ("acquirer_type", attrs["acquirer_type"]), ("completion_status", status)):
            for value in np.unique(arr):
                for metric in ("map", "ndcg@5", "wi_ndcg@5"):
                    a = {q: v for q, v in lm["per_query"][metric].items() if arr[q] == value}
                    b = {q: v for q, v in naive["per_query"][metric].items() if arr[q] == value}
                    if len(set(a) & set(b)) >= 30:
                        subgroups[f"{label}={value}|{metric}"] = paired(a, b, clusters)

    summaries = {os.path.basename(p)[:-5]: json.load(open(p))["summary"] for p in sorted(glob.glob(os.path.join(RUNS, "*__*.json")))}
    extras = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*linear*.json"))):
        run = json.load(open(p))
        w = [f["weights"] for f in run["extras"].get("folds", []) if "weights" in f]
        if w:
            blocks = list(w[0])
            extras[os.path.basename(p)[:-5]] = {b: {"mean": float(np.mean([x[b] for x in w])),
                                                    "min": float(np.min([x[b] for x in w])),
                                                    "max": float(np.max([x[b] for x in w]))} for b in blocks}

    with open(os.path.join(RESULTS, "significance.json"), "w") as f:
        json.dump({"tests": tests, "family_size": len(pvals), "best_baseline": best_baseline, "dispersion": dispersion,
                   "practical": practical, "subgroups": subgroups, "summaries": summaries, "linear_weights": extras}, f, indent=1)
    print(f"family size {len(pvals)}; wrote results/significance.json")


if __name__ == "__main__":
    main()
