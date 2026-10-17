"""Valuation experiments: the feature surrogate and retrieval-based (k-NN) valuation.

Target: y = log(1 + EV/Revenue) on deals with a computable multiple. No transaction-value-derived feature
is ever an input. Everything fitted (imputation, scaling, industry target encoding, PCA of the business-
description embedding, hyperparameters) is fitted inside each outer training fold.

Outputs results/valuation.json.
"""

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.engine import causal_mask, make_folds
from src.evaluation.features import NUMERIC_BLOCKS, VALUE_DERIVED, deal_attributes, raw_numeric
from src.utils.logging_utils import get_logger

RESULTS = os.environ.get("DTX_RESULTS", "results")

warnings.filterwarnings("ignore")
logger = get_logger("valuation_study")
FEATURES = [f for b in ("financial", "structure", "acquirer", "macro") for f in NUMERIC_BLOCKS[b] if f not in VALUE_DERIVED]


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def raw_mae(y, p):
    return float(np.mean(np.abs(np.expm1(y) - np.expm1(p))))


def design(X_raw, train, ind, sic2, y, biz, imputation="median", n_pca=10):
    """Fold-local design matrix: imputed numeric features, industry target encoding, business-text PCA."""
    from sklearn.decomposition import PCA

    X = X_raw.copy()
    med = np.nanmedian(X[train], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    cols = [X]
    if imputation in ("median", "median_plus_indicators"):
        miss = np.isnan(X)
        X = np.where(miss, med, X)
        cols = [X]
        if imputation == "median_plus_indicators":
            keep = miss[train].any(axis=0)
            cols.append(miss[:, keep].astype(float))
    # Industry target encoding from TRAINING deals only, shrunk toward the training mean (m = 5).
    gmean = y[train].mean()
    enc = np.full(len(X), gmean)
    for g in np.unique(sic2):
        tr = train[sic2[train] == g]
        if tr.size:
            enc[sic2 == g] = (y[tr].sum() + 5 * gmean) / (tr.size + 5)
    cols.append(enc[:, None])
    has_biz = np.linalg.norm(biz, axis=1) > 0
    pca = PCA(n_components=n_pca, random_state=0).fit(biz[train][has_biz[train]])
    cols.append(np.where(has_biz[:, None], pca.transform(biz), 0.0))
    return np.hstack(cols)


def tune_xgb(X, y, n_trials=30, seed=0):
    import optuna
    import xgboost as xgb
    from sklearn.model_selection import KFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(n_estimators=trial.suggest_int("n_estimators", 100, 800), max_depth=trial.suggest_int("max_depth", 2, 7),
                      learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                      subsample=trial.suggest_float("subsample", 0.5, 1.0), colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                      min_child_weight=trial.suggest_float("min_child_weight", 1, 20, log=True),
                      reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10, log=True))
        errs = []
        for tr, va in KFold(3, shuffle=True, random_state=seed).split(X):
            m = xgb.XGBRegressor(**params, tree_method="hist", random_state=seed, n_jobs=8).fit(X[tr], y[tr])
            errs.append(np.mean((m.predict(X[va]) - y[va]) ** 2))
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def surrogate(deals, attrs, biz, strategy, imputation="median", models=("xgboost", "hist_gb", "ridge", "industry_median", "global_median")):
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    idx = np.flatnonzero(~np.isnan(attrs["log_multiple"]))
    y_all = np.log1p(np.exp(attrs["log_multiple"]))
    X_raw = np.array([[raw_numeric(d)[f] for f in FEATURES] for d in deals], dtype=float)
    sub = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (len(deals),) else v) for k, v in attrs.items()}
    folds = make_folds(sub, strategy)
    y, sic2 = y_all[idx], attrs["sic2"][idx]
    oof = {m: np.full(idx.size, np.nan) for m in models}
    importance = []
    for tr, te in folds:
        X = design(X_raw[idx], tr, None, sic2, y, biz[idx], imputation)
        for m in models:
            if m == "xgboost":
                params = tune_xgb(X[tr], y[tr])
                model = xgb.XGBRegressor(**params, tree_method="hist", random_state=0, n_jobs=8).fit(X[tr], y[tr])
                oof[m][te] = model.predict(X[te])
                importance.append(model.feature_importances_[: len(FEATURES) + 1])
            elif m == "hist_gb":
                oof[m][te] = HistGradientBoostingRegressor(random_state=0).fit(X[tr], y[tr]).predict(X[te])
            elif m == "ridge":
                Xs = np.nan_to_num(X)
                sc = StandardScaler().fit(Xs[tr])
                oof[m][te] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(sc.transform(Xs[tr]), y[tr]).predict(sc.transform(Xs[te]))
            elif m == "industry_median":
                gmed = np.median(y[tr])
                for q in te:
                    same = tr[sic2[tr] == sic2[q]]
                    oof[m][q] = np.median(y[same]) if same.size else gmed
            elif m == "global_median":
                oof[m][te] = np.median(y[tr])
    evaluated = np.flatnonzero(~np.isnan(oof[models[0]]))
    out = {"n": int(evaluated.size), "folds": strategy, "imputation": imputation, "models": {}}
    for m in models:
        p = oof[m][evaluated]
        out["models"][m] = {"r2_log": round(r2(y[evaluated], p), 4), "mae_raw": round(raw_mae(y[evaluated], p), 3)}
    # Under GroupKFold by SIC the test industry never appears in training, so the "industry median"
    # falls back to the global training median for every test deal; record how often that happened.
    fallbacks = sum(int(not np.any(sic2[tr] == sic2[q])) for tr, te in folds for q in te)
    out["industry_median_fallback_share"] = round(fallbacks / max(1, sum(len(te) for _, te in folds)), 3)
    if "xgboost" in models:
        yy, pp = y[evaluated], oof["xgboost"][evaluated]
        resid_sd = np.std(yy - pp)
        out["xgboost_calibration"] = calibration(yy, pp, resid_sd)
        names = FEATURES + ["industry_target_encoding"]
        imp = np.mean(importance, axis=0)
        out["xgboost_top_features"] = [[names[i], round(float(imp[i]), 4)] for i in np.argsort(-imp)[:8]]
        out["oof_predictions"] = {"deal_index": idx[evaluated].tolist(), "y": yy.round(5).tolist(), "pred": pp.round(5).tolist()}
    return out


def calibration(y, p, sd):
    q_pred = np.quantile(p, [0.2, 0.4, 0.6, 0.8])
    top = p >= q_pred[-1]
    by_realized = []
    edges = np.quantile(y, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for k in range(5):
        sel = (y >= edges[k]) & ((y <= edges[k + 1]) if k == 4 else (y < edges[k + 1]))
        by_realized.append({
            "quintile": k + 1,
            "realized_range_x": [round(float(np.expm1(edges[k])), 2), round(float(np.expm1(edges[k + 1])), 2)],
            "n": int(sel.sum()),
            "mae_x": round(float(np.mean(np.abs(np.expm1(y[sel]) - np.expm1(p[sel])))), 3),
            "bias_x": round(float(np.mean(np.expm1(p[sel]) - np.expm1(y[sel]))), 3),
            "bias_log": round(float(np.mean(p[sel] - y[sel])), 3),
            "coverage_1sd": round(float(np.mean(np.abs(y[sel] - p[sel]) <= sd)), 3),
        })
    return {
        "top_pred_quintile_mean_pred_x": round(float(np.mean(np.expm1(p[top]))), 3),
        "top_pred_quintile_mean_realized_x": round(float(np.mean(np.expm1(y[top]))), 3),
        "coverage_1sd": round(float(np.mean(np.abs(y - p) <= sd)), 3),
        "by_realized_quintile": by_realized,
    }


# ------------------------------------------------------------------------------------------- k-NN valuation
def knn_valuation(attrs, neighbors):
    y_all = np.log1p(np.exp(attrs["log_multiple"]))
    mask = causal_mask(attrs["date_ord"])
    has = ~np.isnan(attrs["log_multiple"])
    queries = sorted({int(q) for q in neighbors["naive"]} & set(np.flatnonzero(has).tolist()))
    queries = [q for q in queries if all(len(neighbors[r][str(q)]["idx"]) >= 3 for r in neighbors)]
    y = y_all[queries]
    out = {"n": len(queries), "retrievers": {}}

    def agg(vals, scores, how):
        if how == "mean":
            return vals.mean()
        if how == "median":
            return np.median(vals)
        if how == "rank_weighted":
            w = 1.0 / np.arange(1, vals.size + 1)
            return np.sum(w * vals) / w.sum()
        raise ValueError(how)

    for rname, table in neighbors.items():
        res = {}
        for k in (3, 5, 8, 10, 20):
            for how in ("mean", "median", "rank_weighted"):
                pred = np.array([agg(y_all[table[str(q)]["idx"][:k]], np.array(table[str(q)]["score"][:k]), how) for q in queries])
                res[f"k{k}_{how}"] = {"r2_log": round(r2(y, pred), 4), "mae_raw": round(raw_mae(y, pred), 3)}
        out["retrievers"][rname] = res

    # Causal industry-median and global-median predictors on the same queries.
    preds_ind, preds_glob = [], []
    for q in queries:
        earlier = np.flatnonzero(mask[q] & has)
        same = earlier[attrs["sic2"][earlier] == attrs["sic2"][q]]
        preds_ind.append(np.median(y_all[same]) if same.size else np.median(y_all[earlier]))
        preds_glob.append(np.median(y_all[earlier]))
    out["industry_median_causal"] = {"r2_log": round(r2(y, np.array(preds_ind)), 4), "mae_raw": round(raw_mae(y, np.array(preds_ind)), 3)}
    out["global_median_causal"] = {"r2_log": round(r2(y, np.array(preds_glob)), 4), "mae_raw": round(raw_mae(y, np.array(preds_glob)), 3)}
    return out, queries


def valuation_objective_neighbors(deals, attrs, emb_business, k_max=50, seed=0):
    """A retriever trained for neighbour-multiple homogeneity: a LightGBM regressor on (query, candidate)
    pair features predicting |y_q - y_c|, trained on training-fold pairs only; neighbours are the
    candidates with the smallest predicted distance."""
    import lightgbm as lgb

    from src.evaluation.features import NumericBlocks, cosine_gram
    from src.retrieval.retrievers import PAIR_DELTA_FEATURES, Context, _pair_matrix

    y_all = np.log1p(np.exp(attrs["log_multiple"]))
    has = ~np.isnan(attrs["log_multiple"])
    sims = {"legal": np.zeros((len(deals), len(deals))), "business": cosine_gram(emb_business)}
    ctx = Context(deals, attrs, np.zeros((len(deals), len(deals))), NumericBlocks(deals, drop=VALUE_DERIVED), sims)
    groups = ("industry", "financial", "structure", "acquirer", "macro", "business")
    rng = np.random.default_rng(seed)
    out = {}
    for train_idx, test_idx in make_folds(attrs, "sic"):
        scaled, fsims = ctx.fold_blocks(train_idx)
        fsims["legal"] = sims["legal"]
        names = ctx.numeric.blocks["financial"]
        delta_cols = [names.index(f) for f in PAIR_DELTA_FEATURES if f in names]
        in_train = np.zeros(len(deals), dtype=bool)
        in_train[train_idx] = True
        X, t = [], []
        for q in train_idx[has[train_idx]]:
            cand = np.flatnonzero(ctx.mask[q] & in_train & has)
            if cand.size == 0:
                continue
            cand = rng.choice(cand, min(cand.size, 200), replace=False)
            X.append(_pair_matrix(ctx, fsims, scaled, delta_cols, q, cand, groups))
            t.append(np.abs(y_all[q] - y_all[cand]))
        model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        model.fit(np.concatenate(X), np.concatenate(t))
        for q in test_idx:
            cand = np.flatnonzero(ctx.mask[q] & has)
            if cand.size == 0:
                continue
            d = model.predict(_pair_matrix(ctx, fsims, scaled, delta_cols, q, cand, groups))
            order = np.argsort(d, kind="stable")[:k_max]
            out[str(int(q))] = {"idx": cand[order].tolist(), "score": (-d[order]).round(6).tolist()}
    return out


def main():
    with open(os.path.join("data", "processed", "deal_universe.json")) as f:
        deals = json.load(f)
    attrs = deal_attributes(deals)
    biz = np.load(os.path.join("data", "embeddings", "business_mpnet.npy"))
    results = {"surrogate": {}}
    for strategy in ("sic", "time"):
        results["surrogate"][strategy] = surrogate(deals, attrs, biz, strategy)
        logger.info(f"surrogate {strategy}: {results['surrogate'][strategy]['models']}")
    results["surrogate_missing_data"] = {}
    for imputation in ("median_plus_indicators", "native_nan"):
        results["surrogate_missing_data"][imputation] = surrogate(deals, attrs, biz, "sic", imputation=imputation,
                                                                  models=("xgboost",))
        results["surrogate_missing_data"][imputation].pop("oof_predictions", None)
        logger.info(f"surrogate missing={imputation}: {results['surrogate_missing_data'][imputation]['models']}")

    with open(os.path.join(RESULTS, "retrieval", "neighbors_outcome_sic.json")) as f:
        neighbors = json.load(f)
    neighbors["valuation_objective"] = valuation_objective_neighbors(deals, attrs, biz)
    results["knn"], _ = knn_valuation(attrs, neighbors)
    logger.info(json.dumps({r: v["k8_rank_weighted"] for r, v in results["knn"]["retrievers"].items()}))
    with open(os.path.join(RESULTS, "valuation.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
