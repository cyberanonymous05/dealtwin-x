"""Retrieval experiments under the causal-temporal protocol. Every run's per-query metrics are written to
results/retrieval/<run>.json (skipped if already present), so significance testing, subgroup analyses
and the paper tables are computed from exactly these files.

Usage: python scripts/run_retrieval_study.py [section ...]
Sections: main, original, ablation, encoders, text, protocol, threshold, missing, drift, neighbors
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.engine import causal_mask, evaluate, make_folds
from src.evaluation.features import VALUE_DERIVED, NumericBlocks, cosine_gram, deal_attributes
from src.evaluation.relevance import CONSTRUCTS, USES_MULTIPLE, build_relevance
from src.retrieval import retrievers as R
from src.utils.logging_utils import get_logger

RESULTS = os.environ.get("DTX_RESULTS", "results")

logger = get_logger("retrieval_study")
OUT = os.path.join(RESULTS, "retrieval")
BLOCKS = ("industry", "financial", "structure", "acquirer", "macro", "legal", "business")
ORIGINAL_BLOCKS = ("industry", "financial", "structure", "acquirer", "macro", "legal")


def load():
    with open(os.path.join("data", "processed", "deal_universe.json")) as f:
        deals = json.load(f)
    emb = {name: np.load(os.path.join("data", "embeddings", f"{name}.npy"))
           for name in ("legal_finbert", "legal_legalbert", "legal_mpnet", "business_mpnet")}
    return deals, deal_attributes(deals), emb


def context(deals, attrs, emb, construct, legal="legal_finbert", causal=True, rel=None, drop=None, **numeric_kw):
    rel = build_relevance(construct, attrs) if rel is None else rel
    drop = (VALUE_DERIVED if construct in USES_MULTIPLE else set()) if drop is None else drop
    sims = {"legal": cosine_gram(emb[legal]), "business": cosine_gram(emb["business_mpnet"])}
    pool = pool_for(construct, attrs)
    return R.Context(deals, attrs, rel, NumericBlocks(deals, drop=drop, **numeric_kw), sims, causal=causal,
                     candidate_pool=pool, business_emb=emb["business_mpnet"]), rel


def pool_for(construct, attrs):
    return ~np.isnan(attrs["log_multiple"]) if construct in USES_MULTIPLE else None


def save(name, result, meta):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"meta": meta, "summary": result.summary(), "extras": result.extras,
                   "per_query": {m: {str(k): v for k, v in d.items()} for m, d in result.per_query.items()}}, f)
    logger.info(f"{name}: {result.summary()}")


def done(name):
    return os.path.exists(os.path.join(OUT, f"{name}.json"))


def run(name, retriever, attrs, rel, folds, meta, causal=True):
    if done(name):
        return
    t0 = time.time()
    pool = pool_for(meta.get("construct"), attrs)
    save(name, evaluate(retriever, attrs, rel, folds, causal=causal, candidate_pool=pool),
         {**meta, "candidate_pool": "deals_with_multiple" if pool is not None else "all", "seconds": round(time.time() - t0, 1)})


def retriever_set(ctx, include=("naive", "sic_backoff", "size_bins", "text_peers", "sic2_text", "linear", "lambdamart",
                                "lambdamart_cascade")):
    table = {
        "naive": lambda: R.naive(ctx),
        "sic_backoff": lambda: R.sic_backoff(ctx),
        "size_bins": lambda: R.size_bins(ctx),
        "text_peers": lambda: R.text_peers(ctx),
        "sic2_text": lambda: R.sic2_text(ctx),
        "linear": lambda: R.linear(ctx, BLOCKS),
        "lambdamart": lambda: R.lambdamart(ctx, BLOCKS),
        "lambdamart_cascade": lambda: R.lambdamart_cascade(ctx, BLOCKS),
    }
    return {k: table[k] for k in include}


def main(sections):
    deals, attrs, emb = load()
    logger.info(f"{len(deals)} deals")
    folds = {"sic": make_folds(attrs, "sic"), "time": make_folds(attrs, "time")}

    if "main" in sections:
        for construct in CONSTRUCTS:
            ctx, rel = context(deals, attrs, emb, construct)
            for strategy in ("sic", "time"):
                for rname, factory in retriever_set(ctx).items():
                    run(f"main__{construct}__{strategy}__{rname}", factory(), attrs, rel, folds[strategy],
                        {"construct": construct, "folds": strategy, "retriever": rname})

    if "original" in sections:
        # The submitted paper's retrievers: six blocks, LambdaMART on per-block similarities only.
        for construct in CONSTRUCTS:
            ctx, rel = context(deals, attrs, emb, construct)
            run(f"original__{construct}__sic__linear", R.linear(ctx, ORIGINAL_BLOCKS), attrs, rel, folds["sic"],
                {"construct": construct, "retriever": "linear_original"})
            run(f"original__{construct}__sic__lambdamart", R.lambdamart(ctx, ORIGINAL_BLOCKS, extras=False), attrs, rel,
                folds["sic"], {"construct": construct, "retriever": "lambdamart_original"})

    if "ablation" in sections:
        ctx, rel = context(deals, attrs, emb, "outcome")
        for drop in BLOCKS:
            groups = tuple(b for b in BLOCKS if b != drop)
            run(f"ablation__outcome__sic__lambdamart_minus_{drop}", R.lambdamart(ctx, groups), attrs, rel, folds["sic"],
                {"construct": "outcome", "dropped_group": drop})

    if "ablation" in sections:
        # Cascade: is the gain from the two-stage design itself or from the precedent-multiple features?
        ctx, rel = context(deals, attrs, emb, "outcome")
        run("ablation__outcome__sic__lambdamart_cascade_minus_multiples", R.lambdamart_cascade(ctx, BLOCKS, use_multiples=False),
            attrs, rel, folds["sic"], {"construct": "outcome", "dropped_group": "precedent_multiples"})

    if "ablation" in sections:
        # Where does the legal-aware construct's gain come from: the clause-text block or structure?
        ctx, rel = context(deals, attrs, emb, "legal")
        for drop in ("legal", "structure", "acquirer", "business"):
            groups = tuple(b for b in BLOCKS if b != drop)
            run(f"ablation__legal__sic__lambdamart_minus_{drop}", R.lambdamart(ctx, groups), attrs, rel, folds["sic"],
                {"construct": "legal", "dropped_group": drop})

    if "encoders" in sections:
        for construct in ("heuristic", "outcome", "legal"):
            for legal in ("legal_finbert", "legal_legalbert", "legal_mpnet"):
                ctx, rel = context(deals, attrs, emb, construct, legal=legal)
                run(f"encoders__{construct}__sic__linear_{legal}", R.linear(ctx, ORIGINAL_BLOCKS), attrs, rel, folds["sic"],
                    {"construct": construct, "legal_encoder": legal})

    if "text" in sections:
        from src.retrieval.text_baselines import TEXT_BASELINES

        ctx, rel = context(deals, attrs, emb, "outcome")
        for tname, fn in TEXT_BASELINES.items():
            run(f"text__outcome__sic__{tname}", R.text_only(ctx, fn(deals)), attrs, rel, folds["sic"],
                {"construct": "outcome", "text_baseline": tname, "industry_first": True})
        run("text__outcome__sic__finbert_legal", R.text_only(ctx, ctx.text_sims["legal"]), attrs, rel, folds["sic"],
            {"construct": "outcome", "text_baseline": "finbert_legal", "industry_first": True})

    if "protocol" in sections:
        for construct in ("concordance", "outcome"):
            ctx, rel = context(deals, attrs, emb, construct, causal=False)
            for rname, factory in retriever_set(ctx, ("naive", "linear", "lambdamart")).items():
                run(f"protocol__{construct}__sic__{rname}__noncausal", factory(), attrs, rel, folds["sic"],
                    {"construct": construct, "retriever": rname, "causal": False}, causal=False)

    if "threshold" in sections:
        for tight in (0.15, 0.20, 0.25, 0.30):
            rel = build_relevance("outcome", attrs, outcome_tight=tight)
            ctx, _ = context(deals, attrs, emb, "outcome", rel=rel)
            run(f"threshold__outcome{tight:.2f}__sic__linear", R.linear(ctx, ORIGINAL_BLOCKS), attrs, rel, folds["sic"],
                {"construct": "outcome", "tight": tight})

    if "missing" in sections:
        variants = {
            "median": dict(),
            "median_plus_indicators": dict(add_missing_indicators=True),
            "drop_premium_and_ev_features": dict(drop=VALUE_DERIVED),
        }
        for construct in ("heuristic", "rubric"):
            for vname, kw in variants.items():
                ctx, rel = context(deals, attrs, emb, construct, **kw)
                for rname, factory in retriever_set(ctx, ("linear", "lambdamart")).items():
                    run(f"missing__{construct}__sic__{rname}__{vname}", factory(), attrs, rel, folds["sic"],
                        {"construct": construct, "retriever": rname, "missing_data": vname})

    if "drift" in sections:
        ctx_full, rel_full = context(deals, attrs, emb, "outcome")
        for start in (2013, 2016, 2019, 2022):
            era = np.flatnonzero((attrs["year"] >= start) & (attrs["year"] < start + 3))
            era_folds = [(np.intersect1d(tr, era), np.intersect1d(te, era)) for tr, te in folds["sic"]]
            run(f"drift__outcome{start}__sic__linear", R.linear(ctx_full, ORIGINAL_BLOCKS), attrs, rel_full,
                [f for f in era_folds if f[0].size and f[1].size], {"construct": "outcome", "era_start": start})

    if "neighbors" in sections:
        path = os.path.join(OUT, "neighbors_outcome_sic.json")
        wanted = ("naive", "sic_backoff", "sic2_text", "linear", "lambdamart", "lambdamart_cascade")
        neighbors = json.load(open(path)) if os.path.exists(path) else {}
        if any(r not in neighbors for r in wanted):
            ctx, rel = context(deals, attrs, emb, "outcome")
            mask = causal_mask(attrs["date_ord"])
            has_multiple = ~np.isnan(attrs["log_multiple"])
            for rname, factory in retriever_set(ctx, tuple(r for r in wanted if r not in neighbors)).items():
                fn, out = factory(), {}
                for train_idx, test_idx in folds["sic"]:
                    scores, _ = fn(train_idx, test_idx)
                    for row, q in enumerate(test_idx):
                        cand = np.flatnonzero(mask[q] & has_multiple)
                        top = cand[np.argsort(-scores[row, cand], kind="stable")[:50]]
                        out[str(int(q))] = {"idx": top.tolist(), "score": scores[row, top].round(6).tolist()}
                neighbors[rname] = out
                logger.info(f"neighbors: {rname}")
            with open(path, "w") as f:
                json.dump(neighbors, f)


if __name__ == "__main__":
    main(sys.argv[1:] or ["main", "original", "ablation", "encoders", "text", "protocol", "threshold", "missing", "drift", "neighbors"])
