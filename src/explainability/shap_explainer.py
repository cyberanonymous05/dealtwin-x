"""Train an XGBoost surrogate for a deal's EV/Revenue multiple and explain predictions with SHAP.

Used to answer the "what valuation multiple would be justified" half of the counterfactual question:
FAISS finds the precedents, this module explains what's driving the multiple those precedents suggest.
"""

import numpy as np
import shap
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.base import clone as _clone_like
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from src.evaluation.cross_validation import deal_level_splits

# Excluded from the surrogate's inputs: these ARE (or directly determine) the prediction target
# (ev_to_revenue), so including them would leak the answer into the features.
LEAKY_TARGET_FEATURES = {"ev_to_revenue", "ev_to_ebitda"}

_MODEL_KWARGS = dict(
    n_estimators=60, max_depth=2, learning_rate=0.1, subsample=0.9,
    reg_lambda=2.0, min_child_weight=2, random_state=0,
)


def _non_leaky_feature_indices(all_feature_names: list[str]) -> list[int]:
    return [i for i, name in enumerate(all_feature_names) if name not in LEAKY_TARGET_FEATURES]


def _competitor_models() -> dict:
    """Model-class field beyond XGBoost vs. Ridge: does any other standard tabular regressor do better
    on this small, heterogeneous sample, or is the ceiling a property of the data, not the model?"""
    return {
        "ridge": Ridge(alpha=5.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=3, random_state=0, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.9, random_state=0
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=200, max_depth=3, num_leaves=7, learning_rate=0.05,
            subsample=0.9, reg_lambda=2.0, min_child_samples=5, random_state=0, verbose=-1
        ),
    }


def _cv_score_model(model_factory, X, y, splits) -> dict:
    fold_mae, fold_r2 = [], []
    for train_idx, test_idx in splits:
        if len(test_idx) == 0 or len(train_idx) < 5:
            continue
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        preds_log = model.predict(X[test_idx])
        fold_mae.append(mean_absolute_error(np.expm1(y[test_idx]), np.expm1(preds_log)))
        if len(test_idx) > 1:
            fold_r2.append(r2_score(y[test_idx], preds_log))
    return {
        "cv_mean_mae_ev_to_revenue": float(np.mean(fold_mae)) if fold_mae else None,
        "cv_mean_r2_log_space": float(np.mean(fold_r2)) if fold_r2 else None,
    }


def _industry_median_baseline_score(deals, y_raw, groups, valid_idx, splits) -> dict:
    """Dumbest defensible baseline: predict the train-fold median realized EV/Revenue for the query's
    2-digit SIC group (global train median when the group is unseen). Anchors what 'no model' looks like."""
    y_raw = np.asarray(y_raw)
    groups = np.asarray(groups)
    fold_mae, fold_r2 = [], []
    for train_idx, test_idx in splits:
        if len(test_idx) == 0 or len(train_idx) < 5:
            continue
        global_median = np.median(y_raw[train_idx])
        group_median = {}
        for g in np.unique(groups[train_idx]):
            group_median[g] = np.median(y_raw[train_idx][groups[train_idx] == g])
        preds = np.array([group_median.get(groups[i], global_median) for i in test_idx])
        actual = y_raw[test_idx]
        fold_mae.append(mean_absolute_error(actual, preds))
        if len(test_idx) > 1:
            fold_r2.append(r2_score(np.log1p(actual), np.log1p(preds)))
    return {
        "cv_mean_mae_ev_to_revenue": float(np.mean(fold_mae)) if fold_mae else None,
        "cv_mean_r2_log_space": float(np.mean(fold_r2)) if fold_r2 else None,
    }


def train_valuation_surrogate(deals: list[dict], fitted: dict):
    """Fit an XGBoost regressor predicting log1p(EV/Revenue) from non-leaky DealDNA numeric features,
    with deal-level GroupKFold CV (grouped by 2-digit SIC) and a SHAP explainer over the final model."""
    numeric_matrix = fitted["numeric_matrix_raw"]
    numeric_imputed = fitted["imputer"].transform(numeric_matrix)
    all_feature_names = fitted["numeric_feature_names"]
    keep_cols = _non_leaky_feature_indices(all_feature_names)
    feature_names = [all_feature_names[i] for i in keep_cols]
    numeric_imputed = numeric_imputed[:, keep_cols]

    # EV/Revenue is right-skewed (a handful of low-revenue, high-strategic-value deals trade at very
    # high multiples) -- regress in log space, standard practice for valuation multiples, so those
    # legitimate outliers don't dominate the loss on a ~30-row sample. Predictions are back-transformed.
    y_raw, groups, valid_idx = [], [], []
    for i, d in enumerate(deals):
        if d.get("ev_estimate") and d.get("revenues"):
            y_raw.append(d["ev_estimate"] / d["revenues"])
            groups.append(d["sic"][:2])
            valid_idx.append(i)
    X = numeric_imputed[valid_idx]
    y = np.log1p(np.array(y_raw))

    # Model-class comparison: is XGBoost's extra complexity even earning its keep on ~30 rows, or
    # would a plain linear model do just as well (or better)? Both are scored on identical folds.
    n_splits, splits = deal_level_splits(X, groups, n_splits=5)

    xgb_scores = _cv_score_model(lambda: xgb.XGBRegressor(**_MODEL_KWARGS), X, y, splits)
    fold_mae = [xgb_scores["cv_mean_mae_ev_to_revenue"]]  # kept for the summary fields below
    fold_r2 = [xgb_scores["cv_mean_r2_log_space"]] if xgb_scores["cv_mean_r2_log_space"] is not None else []

    model_comparison = {"xgboost": xgb_scores}
    for name, estimator in _competitor_models().items():
        model_comparison[name] = _cv_score_model(lambda e=estimator: _clone_like(e), X, y, splits)
    model_comparison["industry_median_baseline"] = _industry_median_baseline_score(
        deals, y_raw, groups, valid_idx, splits
    )

    ridge_fold_mae = [model_comparison["ridge"]["cv_mean_mae_ev_to_revenue"]]
    ridge_fold_r2 = (
        [model_comparison["ridge"]["cv_mean_r2_log_space"]]
        if model_comparison["ridge"]["cv_mean_r2_log_space"] is not None else []
    )

    final_model = xgb.XGBRegressor(**_MODEL_KWARGS)
    final_model.fit(X, y)

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_drivers = sorted(zip(feature_names, mean_abs_shap.tolist()), key=lambda t: t[1], reverse=True)[:8]

    summary = {
        "n_training_deals": int(len(y)),
        "n_group_folds": int(n_splits),
        "target": "log1p(ev_to_revenue), back-transformed for MAE reporting",
        "cv_mean_mae_ev_to_revenue": float(np.mean(fold_mae)) if fold_mae else None,
        "cv_mean_r2_log_space": float(np.mean(fold_r2)) if fold_r2 else None,
        "model_comparison_ridge_baseline": {
            "cv_mean_mae_ev_to_revenue": float(np.mean(ridge_fold_mae)) if ridge_fold_mae else None,
            "cv_mean_r2_log_space": float(np.mean(ridge_fold_r2)) if ridge_fold_r2 else None,
            "note": "Ridge(alpha=5.0) on the same folds/features -- sanity check for whether XGBoost's "
                    "extra model capacity is justified at this sample size, or a plain linear model does as well.",
        },
        "model_comparison": {
            m: {
                "cv_mean_mae_ev_to_revenue": (
                    round(s["cv_mean_mae_ev_to_revenue"], 3) if s["cv_mean_mae_ev_to_revenue"] is not None else None
                ),
                "cv_mean_r2_log_space": (
                    round(s["cv_mean_r2_log_space"], 3) if s["cv_mean_r2_log_space"] is not None else None
                ),
            }
            for m, s in model_comparison.items()
        },
        "top_shap_drivers_global": [{"feature": f, "mean_abs_shap": round(v, 4)} for f, v in top_drivers],
    }
    return summary, final_model, explainer, feature_names, valid_idx


def explain_single_deal(deals: list[dict], fitted: dict, model, explainer, feature_names, valid_idx, query_idx: int):
    """SHAP-explain the surrogate's prediction for one deal already in the fitted training set."""
    if query_idx not in valid_idx:
        return None
    local_idx = valid_idx.index(query_idx)
    keep_cols = _non_leaky_feature_indices(fitted["numeric_feature_names"])
    numeric_matrix = fitted["imputer"].transform(fitted["numeric_matrix_raw"])[:, keep_cols]
    X_row = numeric_matrix[valid_idx][local_idx : local_idx + 1]
    shap_row = explainer.shap_values(X_row)[0]
    contributions = sorted(zip(feature_names, shap_row.tolist()), key=lambda t: abs(t[1]), reverse=True)[:6]
    return {
        "company_name": deals[query_idx]["company_name"],
        "predicted_ev_to_revenue": round(float(np.expm1(model.predict(X_row)[0])), 3),
        "base_value_log_space": round(float(explainer.expected_value), 3),
        "top_feature_contributions_log_space": [{"feature": f, "shap_value": round(v, 4)} for f, v in contributions],
    }
