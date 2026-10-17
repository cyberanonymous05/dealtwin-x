"""Retrievers evaluated under the causal-temporal protocol (see src/evaluation/engine.py).

Each factory returns `retriever(train_idx, test_idx) -> (scores[len(test_idx), n], extras)`.

Baselines (no learning):
  naive            original DealTwin-X baseline: inner product of [2 x SIC2 one-hot, z(log revenue)]
  sic_backoff      lexicographic: same SIC2, then same SIC3, then same SIC4, then closest revenue
  size_bins        same SIC2, then same within-industry revenue quintile, then closest revenue
  text_peers       10-K business-description cosine only (Hoberg-Phillips-style text peers)
  sic2_text        same SIC2, then business-description cosine
Learned (fitted on training-fold pairs only):
  linear           non-negative per-block weights from pairwise-ranking logistic regression
  lambdamart       LightGBM LambdaMART over per-block similarities, fine-grained industry matches,
                   absolute differences of financial features and categorical agreement indicators
"""

from dataclasses import dataclass

import numpy as np

from src.evaluation.engine import causal_mask
from src.evaluation.features import NumericBlocks, cosine_gram

PAIR_DELTA_FEATURES = ["log_revenue", "revenue_growth", "ebitda_margin", "net_margin", "leverage_ratio", "cash_ratio"]


@dataclass
class Context:
    deals: list
    attrs: dict
    rel: np.ndarray            # training labels (same construct as the evaluation target)
    numeric: NumericBlocks     # already configured with the leakage-control drop set
    text_sims: dict            # {"legal": n x n, "business": n x n} cosine similarity matrices
    causal: bool = True
    candidate_pool: np.ndarray | None = None  # deals that may be retrieved (and used as training candidates)
    business_emb: np.ndarray | None = None    # raw 10-K business-description embeddings (cascade surrogate only)

    def __post_init__(self):
        a = self.attrs
        self.same2 = (a["sic2"][:, None] == a["sic2"][None, :]).astype(float)
        self.same3 = (a["sic3"][:, None] == a["sic3"][None, :]).astype(float)
        self.same4 = (a["sic"][:, None] == a["sic"][None, :]).astype(float)
        self.mask = causal_mask(a["date_ord"], self.causal, self.candidate_pool)
        self.n = len(a["sic"])

    def biz_pcs(self, train_idx, k: int = 10):
        """Top-k principal components of the business embedding, fitted on the training fold."""
        if self.business_emb is None:
            return np.zeros((self.n, 0))
        from sklearn.decomposition import PCA

        return PCA(k, random_state=0).fit(self.business_emb[train_idx]).transform(self.business_emb)

    def fold_blocks(self, train_idx):
        scaled = self.numeric.fit_transform(train_idx)
        sims = {"industry": self.same2}
        for b, m in scaled.items():
            sims[b] = cosine_gram(m)
        sims.update(self.text_sims)
        return scaled, sims


def _imputed_logrev(ctx: Context, train_idx):
    x = ctx.attrs["log_revenue"].copy()
    med = np.nanmedian(x[train_idx])
    return np.where(np.isnan(x), med, x)


# ----------------------------------------------------------------------------------------- baselines
def naive(ctx: Context):
    def run(train_idx, test_idx):
        x = _imputed_logrev(ctx, train_idx)
        z = (x - x[train_idx].mean()) / (x[train_idx].std() or 1.0)
        return 4.0 * ctx.same2[test_idx] + np.outer(z[test_idx], z), None
    return run


def sic_backoff(ctx: Context):
    def run(train_idx, test_idx):
        x = _imputed_logrev(ctx, train_idx)
        d = np.abs(x[test_idx][:, None] - x[None, :])
        return 1e6 * ctx.same2[test_idx] + 1e4 * ctx.same3[test_idx] + 1e2 * ctx.same4[test_idx] - d, None
    return run


def size_bins(ctx: Context, n_bins: int = 5):
    def run(train_idx, test_idx):
        x = _imputed_logrev(ctx, train_idx)
        sic2 = ctx.attrs["sic2"]
        global_edges = np.quantile(x[train_idx], np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.digitize(x, global_edges)
        for g in np.unique(sic2):
            tr = train_idx[sic2[train_idx] == g]
            if tr.size >= 2 * n_bins:
                members = sic2 == g
                bins[members] = np.digitize(x[members], np.quantile(x[tr], np.linspace(0, 1, n_bins + 1)[1:-1]))
        same_bin = (bins[test_idx][:, None] == bins[None, :]).astype(float)
        d = np.abs(x[test_idx][:, None] - x[None, :])
        return 1e6 * ctx.same2[test_idx] + 1e3 * same_bin - d, None
    return run


def text_peers(ctx: Context):
    def run(train_idx, test_idx):
        return ctx.text_sims["business"][test_idx], None
    return run


def sic2_text(ctx: Context):
    def run(train_idx, test_idx):
        return 1e3 * ctx.same2[test_idx] + ctx.text_sims["business"][test_idx], None
    return run


def text_only(ctx: Context, sim: np.ndarray, industry_first: bool = True):
    def run(train_idx, test_idx):
        return (1e3 * ctx.same2[test_idx] if industry_first else 0.0) + sim[test_idx], None
    return run


# ----------------------------------------------------------------------------------------- linear
def _train_candidates(ctx: Context, train_idx):
    in_train = np.zeros(ctx.n, dtype=bool)
    in_train[train_idx] = True
    return in_train


def _sample_pairs(ctx: Context, train_idx, rng, per_query: int = 25):
    in_train = _train_candidates(ctx, train_idx)
    out = []
    for i in train_idx:
        cand = np.flatnonzero(ctx.mask[i] & in_train)
        if cand.size < 2:
            continue
        r = ctx.rel[i, cand]
        if np.unique(r).size < 2:
            continue
        a = rng.choice(cand, size=per_query * 4)
        b = rng.choice(cand, size=per_query * 4)
        keep = ctx.rel[i, a] != ctx.rel[i, b]
        a, b = a[keep][:per_query], b[keep][:per_query]
        out.append(np.stack([np.full(a.size, i), a, b], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 3), dtype=int)


def linear(ctx: Context, blocks: tuple, seeds=(0, 1, 2), budget: float | None = None):
    from sklearn.linear_model import LogisticRegression

    budget = budget or float(len(blocks))

    def run(train_idx, test_idx):
        _, sims = ctx.fold_blocks(train_idx)
        weights = []
        for seed in seeds:
            pairs = _sample_pairs(ctx, train_idx, np.random.default_rng(seed))
            i, j, k = pairs[:, 0], pairs[:, 1], pairs[:, 2]
            X = np.stack([sims[b][i, j] - sims[b][i, k] for b in blocks], axis=1)
            y = (ctx.rel[i, j] > ctx.rel[i, k]).astype(int)
            w = np.clip(LogisticRegression(fit_intercept=False, max_iter=2000).fit(X, y).coef_.ravel(), 0, None)
            weights.append(w / w.sum() * budget if w.sum() > 0 else np.full(len(blocks), budget / len(blocks)))
        w = np.mean(weights, axis=0)
        scores = sum(w[n] * sims[b][test_idx] for n, b in enumerate(blocks))
        return scores, {"weights": dict(zip(blocks, w.round(4).tolist()))}

    return run


# ----------------------------------------------------------------------------------------- LambdaMART
def _pair_matrix(ctx, sims, scaled, delta_cols, q, cand, groups, extras=True):
    feats = []
    if "industry" in groups:
        feats += [ctx.same2[q, cand], ctx.same3[q, cand], ctx.same4[q, cand]] if extras else [ctx.same2[q, cand]]
    for b in ("financial", "structure", "acquirer", "macro", "legal", "business"):
        if b in groups and b in sims:
            feats.append(sims[b][q, cand])
    if not extras:
        return np.stack(feats, axis=1).astype(np.float32)
    if "financial" in groups:
        fin = scaled["financial"]
        for c in delta_cols:
            feats.append(np.abs(fin[q, c] - fin[cand, c]))
    if "structure" in groups:
        a = ctx.attrs
        feats += [(a["mix"][q] == a["mix"][cand]).astype(float), (a["deal_type"][q] == a["deal_type"][cand]).astype(float)]
    if "acquirer" in groups:
        feats.append((ctx.attrs["acquirer_type"][q] == ctx.attrs["acquirer_type"][cand]).astype(float))
    if "macro" in groups:
        feats.append(np.abs(ctx.attrs["date_ord"][q] - ctx.attrs["date_ord"][cand]) / 365.0)
    return np.stack(feats, axis=1).astype(np.float32)


def lambdamart(ctx: Context, groups: tuple = ("industry", "financial", "structure", "acquirer", "macro", "legal", "business"),
               max_pos: int = 150, max_neg: int = 150, seed: int = 0, params: dict | None = None,
               extras: bool = True):
    """`extras=False` reproduces the submitted DealTwin-X ranker: one similarity feature per block only."""
    import lightgbm as lgb

    levels = np.unique(ctx.rel)
    hp = dict(objective="lambdarank", n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1,
              lambdarank_truncation_level=20)
    hp.update(params or {})

    def run(train_idx, test_idx):
        rng = np.random.default_rng(seed)
        scaled, sims = ctx.fold_blocks(train_idx)
        names = ctx.numeric.blocks["financial"]
        delta_cols = [names.index(f) for f in PAIR_DELTA_FEATURES if f in names]
        in_train = _train_candidates(ctx, train_idx)

        def build(queries):
            X, y, g = [], [], []
            for q in queries:
                cand = np.flatnonzero(ctx.mask[q] & in_train)
                r = ctx.rel[q, cand]
                if cand.size < 2 or np.unique(r).size < 2:
                    continue
                pos, neg = cand[r > 0], cand[r == 0]
                if pos.size > max_pos:
                    pos = rng.choice(pos, max_pos, replace=False)
                if neg.size > max_neg:
                    same = neg[ctx.same2[q, neg] > 0]
                    other = neg[ctx.same2[q, neg] == 0]
                    take_same = min(same.size, max_neg // 2)
                    neg = np.concatenate([rng.choice(same, take_same, replace=False),
                                          rng.choice(other, min(other.size, max_neg - take_same), replace=False)])
                sel = np.concatenate([pos, neg])
                X.append(_pair_matrix(ctx, sims, scaled, delta_cols, q, sel, groups, extras))
                y.append(np.searchsorted(levels, ctx.rel[q, sel]))
                g.append(sel.size)
            return np.concatenate(X), np.concatenate(y), g

        tr_q = rng.permutation(train_idx)
        n_val = max(1, int(0.15 * tr_q.size))
        Xt, yt, gt = build(tr_q[n_val:])
        Xv, yv, gv = build(tr_q[:n_val])
        model = lgb.LGBMRanker(**hp)
        model.fit(Xt, yt, group=gt, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[5],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        scores = np.full((len(test_idx), ctx.n), -1e9, dtype=float)
        for row, q in enumerate(test_idx):
            cand = np.flatnonzero(ctx.mask[q])
            if cand.size:
                scores[row, cand] = model.predict(_pair_matrix(ctx, sims, scaled, delta_cols, q, cand, groups, extras),
                                                  num_iteration=model.best_iteration_)
        gain = model.booster_.feature_importance(importance_type="gain")
        return scores, {"best_iteration": int(model.best_iteration_ or hp["n_estimators"]),
                        "gain": (gain / gain.sum()).round(4).tolist()}

    return run


# ----------------------------------------------------------------------------------------- cascade
def _surrogate_predictions(ctx: Context, train_idx, seed: int = 0, inner_folds: int = 5):
    """Predicted log multiple for every deal from PRE-DEAL features only (value-derived features always
    dropped), fitted on training-fold deals with a multiple. Training-fold deals get inner out-of-fold
    predictions, so the ranker never trains on a feature fitted to its own label. The query's realized
    multiple is never an input at test time."""
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    from src.evaluation.features import VALUE_DERIVED, NumericBlocks

    blocks = NumericBlocks(ctx.deals, drop=VALUE_DERIVED).fit_transform(train_idx)
    X = np.hstack([blocks[b] for b in ("financial", "structure", "acquirer", "macro")])
    X = np.hstack([X, ctx.biz_pcs(train_idx)])
    y = ctx.attrs["log_multiple"]
    lab = train_idx[~np.isnan(y[train_idx])]
    hp = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    pred = lgb.LGBMRegressor(**hp).fit(X[lab], y[lab]).predict(X)
    for tr, te in KFold(inner_folds, shuffle=True, random_state=seed).split(lab):
        pred[lab[te]] = lgb.LGBMRegressor(**hp).fit(X[lab[tr]], y[lab[tr]]).predict(X[lab[te]])
    return pred


def lambdamart_cascade(ctx: Context, groups: tuple = ("industry", "financial", "structure", "acquirer", "macro", "legal", "business"),
                       top_n: int = 50, seed: int = 0, params: dict | None = None,
                       use_multiples: bool = True):
    """Two-stage retrieval, fixed in advance (top_n = 50, no tuning on evaluation data).
    Stage 1: the SIC back-off screen proposes its top-`top_n` admissible candidates.
    Stage 2: LambdaMART re-ranks them with the standard pair features plus (a) the candidate's screen rank,
    (b) the candidate's realized multiple (a past, public deal), (c) the query's predicted multiple from a
    fold-local surrogate, (d) the median realized multiple of the query's screened candidates, and the
    absolute gaps |(c)-(b)| and |(d)-(b)|. Candidates beyond `top_n` keep the screen's order below them.
    `use_multiples=False` (ablation) keeps only the two-stage design and the screen rank."""
    import lightgbm as lgb

    levels = np.unique(ctx.rel)
    hp = dict(objective="lambdarank", n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1,
              lambdarank_truncation_level=20)
    hp.update(params or {})
    m_all = ctx.attrs["log_multiple"]

    def run(train_idx, test_idx):
        rng = np.random.default_rng(seed)
        scaled, sims = ctx.fold_blocks(train_idx)
        names = ctx.numeric.blocks["financial"]
        delta_cols = [names.index(f) for f in PAIR_DELTA_FEATURES if f in names]
        in_train = _train_candidates(ctx, train_idx)
        x = _imputed_logrev(ctx, train_idx)
        m_hat = _surrogate_predictions(ctx, train_idx, seed) if use_multiples else np.zeros(ctx.n)

        def screen(q, allowed):
            cand = np.flatnonzero(allowed)
            s = 1e6 * ctx.same2[q, cand] + 1e4 * ctx.same3[q, cand] + 1e2 * ctx.same4[q, cand] - np.abs(x[q] - x[cand])
            return cand[np.argsort(-s, kind="stable")]

        def features(q, top):
            base = _pair_matrix(ctx, sims, scaled, delta_cols, q, top, groups, True)
            mc = m_all[top]
            med = np.nanmedian(mc) if np.any(~np.isnan(mc)) else np.nan
            extra = np.stack([np.log1p(np.arange(top.size)), mc, np.full(top.size, m_hat[q]), np.full(top.size, med),
                              np.abs(m_hat[q] - mc), np.abs(med - mc)], axis=1)
            if not use_multiples:
                extra = extra[:, :1]
            return np.hstack([base, extra]).astype(np.float32)

        def build(queries):
            X, y, g = [], [], []
            for q in queries:
                top = screen(q, ctx.mask[q] & in_train)[:top_n]
                if top.size < 2 or np.unique(ctx.rel[q, top]).size < 2:
                    continue
                X.append(features(q, top))
                y.append(np.searchsorted(levels, ctx.rel[q, top]))
                g.append(top.size)
            return np.concatenate(X), np.concatenate(y), g

        tr_q = rng.permutation(train_idx)
        n_val = max(1, int(0.15 * tr_q.size))
        Xt, yt, gt = build(tr_q[n_val:])
        Xv, yv, gv = build(tr_q[:n_val])
        model = lgb.LGBMRanker(**hp)
        model.fit(Xt, yt, group=gt, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[5],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        scores = np.full((len(test_idx), ctx.n), -1e9, dtype=float)
        for row, q in enumerate(test_idx):
            order = screen(q, ctx.mask[q])
            if not order.size:
                continue
            top, rest = order[:top_n], order[top_n:]
            s = model.predict(features(q, top), num_iteration=model.best_iteration_)
            scores[row, top] = 1e6 + (s - s.min())          # stage-2 order inside the screened set
            scores[row, rest] = -np.arange(1, rest.size + 1)  # screen order below it
        return scores, {"best_iteration": int(model.best_iteration_ or hp["n_estimators"]), "top_n": top_n}

    return run
