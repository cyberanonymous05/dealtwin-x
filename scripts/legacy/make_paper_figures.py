"""Generate publication-quality figures for the ACISM 2026 paper from the pipeline's real result JSONs.

Uses the validated categorical palette from the dataviz skill (fixed hue order, CVD-safe):
blue #2a78d6 (DealDNA / ours), orange #eb6834 (naive baseline / comparison), aqua #1baf7a (secondary).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#1a1a19"
MUTED = "#52514e"
GRID = "#dedcd5"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RESULTS_DIR = os.path.join("data", "processed")
# self-contained: writes inside this project regardless of where it's placed, so re-running after
# extracting/moving the project folder elsewhere still works without a sibling folder assumption
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_assets", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


def bar_labels(ax, bars, fmt="{:.3f}", avoid_y=None):
    """Labels sit just inside the top of tall-enough bars (white bold text, never collides with a
    reference line or an adjacent label); short bars get a small dark label just outside instead,
    since white text wouldn't fit inside them."""
    ax.figure.canvas.draw()
    y_lo, y_hi = ax.get_ylim()
    threshold = 0.12 * (y_hi - y_lo)
    for b in bars:
        h = b.get_height()
        if abs(h) >= threshold:
            sign = 1 if h >= 0 else -1
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                        xytext=(0, -12 * sign), textcoords="offset points",
                        ha="center", va="top" if h >= 0 else "bottom",
                        fontsize=9, color="white", fontweight="bold")
        else:
            sign = 1 if h >= 0 else -1
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3 * sign), textcoords="offset points",
                        ha="center", va="bottom" if h >= 0 else "top",
                        fontsize=9, color=INK)


def fig1_ablation_progression():
    ablation = load("ablation_and_baseline_results.json")
    order = [
        ("financial_only", "Financial only"),
        ("industry_only", "Industry only"),
        ("industry_plus_financial", "Industry\n+ Financial"),
        ("industry_financial_structure", "+ Structure"),
        ("industry_financial_structure_macro", "+ Macro"),
        ("full_dealdna", "Full DealDNA\n(+ Legal)"),
    ]
    labels = [lab for _, lab in order]
    values = [ablation[key]["within_industry_ndcg@5"] for key, _ in order]
    baseline = ablation["naive_industry_size_baseline"]["within_industry_ndcg@5"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.55, color=BLUE, zorder=3, label="DealDNA (hand-weighted blocks)")
    ax.axhline(baseline, color=ORANGE, linestyle="--", linewidth=1.8, zorder=4,
               label=f"Naive industry+size baseline ({baseline:.3f})")
    bar_labels(ax, bars, avoid_y=baseline)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Within-industry NDCG@5")
    ax.set_ylim(0, 1.12)
    ax.set_title("Block ablation: within-industry ranking quality vs. naive baseline")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False, fontsize=9)
    save(fig, "fig1_ablation_progression.png")


def fig2_learned_weights():
    heuristic = load("learned_metric_results.json")["avg_learned_weights"]
    outcome = load("outcome_grounded_results.json")["full_dealdna_learned_weights_no_multiples"]["avg_learned_weights"]
    blocks = ["industry", "financial", "structure", "macro", "legal"]
    labels = [b.capitalize() for b in blocks]

    x = np.arange(len(blocks))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - width / 2, [heuristic[b] for b in blocks], width, color=BLUE, zorder=3,
                label="Heuristic relevance (industry/size/structure/recency)")
    b2 = ax.bar(x + width / 2, [outcome[b] for b in blocks], width, color=ORANGE, zorder=3,
                label="Outcome-grounded relevance (realized multiple similarity)")
    bar_labels(ax, b1, fmt="{:.2f}")
    bar_labels(ax, b2, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Learned block weight (budget = 5.0)")
    ax.set_title("Cross-validated learned block weights, two independent relevance constructs")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    save(fig, "fig2_learned_weights.png")


def fig3_outcome_grounded_comparison():
    r = load("outcome_grounded_results.json")
    base = r["naive_industry_size_baseline"]
    dna = r["full_dealdna_learned_weights_no_multiples"]
    metrics = [
        ("ndcg@5", "NDCG@5"), ("map", "MAP"), ("mrr", "MRR"),
        ("precision@5", "Precision@5"), ("recall@10", "Recall@10"),
        ("within_industry_ndcg@5", "Within-Ind.\nNDCG@5"), ("within_industry_ndcg@10", "Within-Ind.\nNDCG@10"),
    ]
    labels = [lab for _, lab in metrics]
    base_vals = [base[k] for k, _ in metrics]
    dna_vals = [dna[k] for k, _ in metrics]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    b1 = ax.bar(x - width / 2, base_vals, width, color=ORANGE, zorder=3, label="Naive industry+size baseline")
    b2 = ax.bar(x + width / 2, dna_vals, width, color=BLUE, zorder=3, label="Full DealDNA (learned weights, multiples excluded)")
    bar_labels(ax, b1, fmt="{:.2f}")
    bar_labels(ax, b2, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Outcome-grounded evaluation: DealDNA vs. naive baseline (realized-multiple relevance)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=9)
    save(fig, "fig3_outcome_grounded_comparison.png")


_FEATURE_LABELS = {
    "mac_clause_intensity": "MAC clause intensity",
    "yield_curve_10y2y": "Yield curve (10y-2y)",
    "treasury_10y": "Treasury 10y yield",
    "ebitda_margin": "EBITDA margin",
    "log_revenue": "Log revenue",
    "net_margin": "Net margin",
    "revenue_growth": "Revenue growth",
    "leverage_ratio": "Leverage ratio",
    "termination_fee_pct_of_ev": "Termination fee (% of EV)",
}


def fig4_shap_drivers():
    surrogate = load("pipeline_results.json")["valuation_surrogate"]
    drivers = surrogate["top_shap_drivers_global"]
    names = [_FEATURE_LABELS.get(d["feature"], d["feature"].replace("_", " ").capitalize()) for d in drivers][::-1]
    values = [d["mean_abs_shap"] for d in drivers][::-1]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(names, values, color=BLUE, zorder=3, height=0.6)
    for b in bars:
        w = b.get_width()
        ax.annotate(f"{w:.3f}", (w, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Mean |SHAP value| (log EV/Revenue)")
    ax.set_title("Global SHAP feature importance: XGBoost valuation-multiple surrogate")
    ax.set_xlim(0, max(values) * 1.25)
    save(fig, "fig4_shap_drivers.png")


_SURROGATE_MODEL_ORDER = [
    ("industry_median_baseline", "Industry\nmedian"),
    ("ridge", "Ridge"),
    ("random_forest", "Random\nForest"),
    ("gradient_boosting", "Gradient\nBoosting"),
    ("lightgbm", "LightGBM"),
    ("xgboost", "XGBoost"),
]


def fig5_model_comparison():
    surrogate = load("pipeline_results.json")["valuation_surrogate"]
    comp = surrogate.get("model_comparison")
    if comp:
        n = surrogate.get("n_training_deals", "?")
        keys = [(k, lab) for k, lab in _SURROGATE_MODEL_ORDER if k in comp]
        labels = [lab for _, lab in keys]
        mae = [comp[k]["cv_mean_mae_ev_to_revenue"] for k, _ in keys]
        r2 = [comp[k]["cv_mean_r2_log_space"] for k, _ in keys]
        colors = [BLUE if k == "xgboost" else ORANGE for k, _ in keys]
    else:  # legacy 2-model shape
        ridge = surrogate["model_comparison_ridge_baseline"]
        n = 158
        labels = ["XGBoost", "Ridge"]
        mae = [surrogate["cv_mean_mae_ev_to_revenue"], ridge["cv_mean_mae_ev_to_revenue"]]
        r2 = [surrogate["cv_mean_r2_log_space"], ridge["cv_mean_r2_log_space"]]
        colors = [BLUE, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    b1 = axes[0].bar(labels, mae, color=colors, width=0.6, zorder=3)
    bar_labels(axes[0], b1, fmt="{:.2f}")
    axes[0].set_ylabel("Held-out MAE (EV/Revenue, x)")
    axes[0].set_title("(a) Cross-validated MAE")
    axes[0].tick_params(axis="x", labelsize=8)

    b2 = axes[1].bar(labels, r2, color=colors, width=0.6, zorder=3)
    bar_labels(axes[1], b2, fmt="{:.2f}")
    axes[1].axhline(0, color=MUTED, linewidth=0.8)
    axes[1].set_ylabel("Held-out R² (log space)")
    axes[1].set_title("(b) Cross-validated R²")
    axes[1].tick_params(axis="x", labelsize=8)

    fig.suptitle(f"Valuation-multiple surrogate: model-class comparison (n={n}, GroupKFold)", y=1.03)
    fig.tight_layout()
    save(fig, "fig5_model_comparison.png")


def fig6_text_baselines():
    """FinBERT legal block (learned weight, from outcome-grounded study) vs. lexical / generic-encoder
    baselines over the same clause passages, on outcome-grounded NDCG@5 and MAP."""
    try:
        tb = load("text_baseline_results.json")["baselines"]
    except FileNotFoundError:
        print("skip fig6: text_baseline_results.json not found")
        return
    outcome = load("outcome_grounded_results.json")["full_dealdna_learned_weights_no_multiples"]

    names = ["TF-IDF", "BM25", "MiniLM", "Full DealDNA\n(learned)"]
    key = ["tfidf_legal_text", "bm25_legal_text", "minilm_legal_text"]
    ndcg = [tb[k]["outcome_grounded_relevance"]["ndcg@5"] for k in key] + [outcome["ndcg@5"]]
    mapv = [tb[k]["outcome_grounded_relevance"]["map"] for k in key] + [outcome["map"]]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors_n = [ORANGE, ORANGE, ORANGE, BLUE]
    b1 = ax.bar(x - width / 2, ndcg, width, color=colors_n, zorder=3)
    b2 = ax.bar(x + width / 2, mapv, width, color=colors_n, alpha=0.55, zorder=3)
    bar_labels(ax, b1, fmt="{:.2f}")
    bar_labels(ax, b2, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score (outcome-grounded relevance)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Legal-text retrieval baselines vs. full DealDNA\n(darker = NDCG@5, lighter = MAP)")
    save(fig, "fig6_text_baselines.png")


def fig7_per_industry():
    try:
        pi = load("extended_ablation_results.json")["per_industry_within_ndcg5"]
    except FileNotFoundError:
        print("skip fig7: extended_ablation_results.json not found")
        return
    groups = [g for g in pi if pi[g]["delta"] is not None]
    groups.sort(key=lambda g: pi[g]["delta"])
    deltas = [pi[g]["delta"] for g in groups]
    colors = [AQUA if d >= 0 else ORANGE for d in deltas]

    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.32 * len(groups))))
    ax.barh([f"SIC {g}" for g in groups], deltas, color=colors, zorder=3, height=0.6)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    ax.set_xlabel("Within-industry NDCG@5: full DealDNA − naive baseline")
    ax.set_title("Where the multi-modal signal helps vs. hurts, by industry group")
    save(fig, "fig7_per_industry.png")


if __name__ == "__main__":
    fig1_ablation_progression()
    fig2_learned_weights()
    fig3_outcome_grounded_comparison()
    fig4_shap_drivers()
    fig5_model_comparison()
    fig6_text_baselines()
    fig7_per_industry()
