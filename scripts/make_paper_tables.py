"""Collect every number used in the paper into results/paper_numbers.json and draw the paper figures
(paper/figures/). The manuscript is written from this file only."""

import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.features import deal_attributes
from src.evaluation.relevance import CONSTRUCTS, build_relevance

RESULTS = os.environ.get("DTX_RESULTS", "results")
FIG = os.path.join("paper", "figures")
RET = os.path.join(RESULTS, "retrieval")


def summary(name):
    path = os.path.join(RET, f"{name}.json")
    return json.load(open(path))["summary"] if os.path.exists(path) else None


def kendall_between_constructs(attrs, n_sample=400, seed=0):
    """Kendall tau between constructs' graded relevance over sampled same-industry pairs."""
    from scipy.stats import kendalltau

    rels = {c: build_relevance(c, attrs) for c in CONSTRUCTS}
    same = attrs["sic2"][:, None] == attrs["sic2"][None, :]
    np.fill_diagonal(same, False)
    i, j = np.nonzero(same)
    rng = np.random.default_rng(seed)
    pick = rng.choice(i.size, size=min(200_000, i.size), replace=False)
    i, j = i[pick], j[pick]
    out = {}
    for a in CONSTRUCTS:
        for b in CONSTRUCTS:
            if a < b:
                out[f"{a}~{b}"] = round(float(kendalltau(rels[a][i, j], rels[b][i, j]).statistic), 3)
    pairs = {c: int((rels[c] > 0).sum()) for c in CONSTRUCTS}
    return out, pairs


def corpus_numbers(deals, attrs):
    n = len(deals)

    def cov(f):
        return round(sum(1 for d in deals if d.get(f) is not None) / n, 3)

    return {
        "n_deals": n,
        "by_type": dict(Counter(d["deal_type"] for d in deals)),
        "by_form": dict(Counter(d["form"] for d in deals)),
        "n_sic2": int(len(set(attrs["sic2"]))),
        "years": [min(d["filing_date"] for d in deals)[:4], max(d["filing_date"] for d in deals)[:4]],
        "computable_multiple": int(np.sum(~np.isnan(attrs["log_multiple"]))),
        "acquirer": dict(Counter(d["acquirer_type"] for d in deals)),
        "cross_border": int(sum(bool(d["cross_border"]) for d in deals)),
        "completion": dict(Counter(d["completion_status"] for d in deals)),
        "coverage": {f: cov(f) for f in ("revenues", "ebitda_estimate", "revenue_growth", "ev_estimate", "cash_per_share",
                                         "deal_premium", "termination_fee_usd", "shares_outstanding")},
        "clause_passage_share": round(sum(1 for d in deals if d.get("clause_passages")) / n, 3),
        "business_description_share": round(sum(1 for d in deals if len(d.get("business_description") or "") > 500) / n, 3),
        "xbrl_revenue_provenance": dict(Counter((d.get("xbrl_provenance") or {}).get("revenues", "none").split(":")[-1] for d in deals)),
        "median_ev_to_revenue": round(float(np.exp(np.nanmedian(attrs["log_multiple"]))), 3),
    }


def figures(sig, val):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    os.makedirs(FIG, exist_ok=True)

    # Fig: paired difference vs naive with 95% cluster-bootstrap CIs (outcome construct).
    metrics = ["ndcg@5", "map", "mrr", "precision@5", "recall@10", "wi_ndcg@5", "wi_ndcg@10"]
    labels = ["NDCG@5", "MAP", "MRR", "P@5", "R@10", "WI-NDCG@5", "WI-NDCG@10"]
    panels = [("lambdamart_vs_naive", "Improved LambdaMART vs. naive"),
              ("lambdamart_vs_lambdamart_original", "Improved vs. submitted LambdaMART"),
              ("lambdamart_vs_sic_backoff", "Improved LambdaMART vs. SIC back-off")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, sharex=True)
    for ax, (key, title) in zip(axes, panels):
        for row, m in enumerate(metrics):
            t = sig["tests"].get(f"outcome|{key}|{m}")
            if not t:
                continue
            color = "#1a9e77" if t["p_holm"] < 0.05 else "#555555"
            ax.plot(t["ci95"], [row, row], color=color, lw=4, solid_capstyle="butt")
            ax.plot(t["diff"], row, "o", color="black", ms=4)
        ax.axvline(0, ls="--", color="#444", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(labels)
        ax.grid(axis="x", color="#ddd")
        ax.set_xlabel("mean per-deal difference")
    fig.suptitle("Outcome-grounded construct: paired per-deal difference (95% cluster-bootstrap CI; green = Holm-significant)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_significance.png"), dpi=300)
    plt.close(fig)

    # Fig: surrogate model comparison under both CV schemes.
    models = ["xgboost", "hist_gb", "ridge", "industry_median", "global_median"]
    names = ["XGBoost", "HistGB", "Ridge", "Industry\nmedian", "Global\nmedian"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    width = 0.38
    for k, (strategy, label, color) in enumerate((("sic", "GroupKFold by SIC", "#3b6ea8"), ("time", "Expanding time window", "#c4572a"))):
        res = val["surrogate"][strategy]["models"]
        x = np.arange(len(models)) + (k - 0.5) * width
        axes[0].bar(x, [res[m]["r2_log"] for m in models], width, label=label, color=color)
        axes[1].bar(x, [res[m]["mae_raw"] for m in models], width, label=label, color=color)
    for ax, title in zip(axes, ("Cross-validated log-space $R^2$", "MAE on EV/Revenue (x)")):
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="#444", lw=0.6)
        ax.grid(axis="y", color="#eee")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_model_comparison.png"), dpi=300)
    plt.close(fig)


def main():
    deals = json.load(open(os.path.join("data", "processed", "deal_universe.json")))
    attrs = deal_attributes(deals)
    sig = json.load(open(os.path.join(RESULTS, "significance.json")))
    val = json.load(open(os.path.join(RESULTS, "valuation.json")))
    parse = json.load(open(os.path.join("data", "processed", "parse_stats.json")))
    discovery = json.load(open(os.path.join("data", "processed", "discovery_stats.json")))
    tau, pairs = kendall_between_constructs(attrs)

    numbers = {
        "discovery": discovery,
        "parse": {k: v for k, v in parse.items() if k != "clause_selector"},
        "clause_selector": parse["clause_selector"],
        "corpus": corpus_numbers(deals, attrs),
        "construct_kendall_tau": tau,
        "construct_positive_pairs": pairs,
        "retrieval": {name: s for name, s in sig["summaries"].items()},
        "tests": sig["tests"],
        "family_size": sig["family_size"],
        "best_baseline": sig["best_baseline"],
        "practical": sig["practical"],
        "dispersion": sig["dispersion"],
        "subgroups": sig["subgroups"],
        "linear_weights": sig["linear_weights"],
        "valuation": {k: v for k, v in val.items()},
        "valuation_tests": json.load(open(os.path.join(RESULTS, "valuation_tests.json"))),
    }
    for s in numbers["valuation"]["surrogate"].values():
        s.pop("oof_predictions", None)
    with open(os.path.join(RESULTS, "paper_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1)
    figures(sig, val)
    print("wrote results/paper_numbers.json and paper/figures/")


if __name__ == "__main__":
    main()
