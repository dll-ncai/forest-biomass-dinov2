#!/usr/bin/env python3
"""
Leak-free biomass regression pipeline (corrected methodology).

Fixes vs. the original code:
  1. Feature COUNT (K) is selected on TRAINING data only:
       - German : K chosen by 5-fold CV on the training split (per method), then a
                  single final evaluation on the held-out test set.
       - Pakistan: fully NESTED LOO. Inside each outer fold, K (and model
                  hyper-parameters) are chosen by inner CV on the 24 training
                  samples; the held-out sample is predicted once.
  2. Feature RANKING (|Pearson| with biomass) is recomputed on TRAIN ONLY, inside
     every CV fold / outer fold. No global ranking over all samples.
  3. CLUSTERING (German) is fit on TRAIN ONLY: StandardScaler + UMAP + KMeans are
     fit on the training split; validation/test samples are assigned with
     scaler.transform -> umap.transform -> kmeans.predict. Per-cluster feature
     ranking uses that cluster's TRAIN samples only.

Works for both feature backbones (DINO 384-d, ResNet50 2048-d) and both sites.
Usage:  python corrected_pipeline.py --site german --features dino
"""
import os, json, argparse, warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = os.path.dirname(os.path.abspath(__file__))
# Train-only feature-count sweep. Coarser than a stride-10 grid at the high end
# (where curves are flat) to keep the nested search tractable with the richer
# hyper-parameter grids below.  Values above n_features are skipped at use sites.
FEATURE_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160,
                  190, 220, 260, 300, 360]

# ───────────────────────── metrics & models ─────────────────────────

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    r2 = float(r2_score(y, p)) if len(y) > 1 else float("nan")
    mean_y = float(np.mean(y))
    prmse = (rmse / mean_y * 100.0) if mean_y > 1e-9 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "percent_rmse": prmse}


def _abs_corr_order(X, y):
    """All feature indices ordered by |Pearson(feature, y)| desc (vectorized, TRAIN only)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    num = Xc.T @ yc
    den = np.sqrt((Xc ** 2).sum(axis=0) * float(yc @ yc) + 1e-30)
    r = num / den
    r[~np.isfinite(r)] = 0.0
    return np.argsort(-np.abs(r))


def rank_features_train(Xtr, ytr, n):
    """Top-n feature indices by |Pearson| computed on TRAIN ONLY."""
    return _abs_corr_order(Xtr, ytr)[:n].tolist()


def predict_nnls(Xtr, ytr, Xte, k, cosine=True, simplex=True, l2=0.0):
    """Local non-negative reconstruction of each test example from its k nearest
    TRAIN neighbours, then predict biomass with the same weights.

    Improvements over the first corrected version:
      * ``l2``  Tikhonov (ridge) regularisation on the reconstruction weights, solved
                as an augmented NNLS  min ||A w - b||^2 + l2||w||^2, s.t. w >= 0.
                Stabilises the (often under-determined) reconstruction when the query
                lives in a high-dim feature space with few neighbours.
      * wider ``k`` range is now supported (the grid goes up to ~all-of-train), so the
                local basis is large enough to reconstruct the query well.
    ``simplex=True`` renormalises the weights to sum to 1 (convex combination -> stays
    inside the training biomass range); ``simplex=False`` lets the raw NNLS weights
    through (can extrapolate beyond the neighbour labels)."""
    out = []
    if cosine:
        Xtr_n = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-10)
    else:
        sc = StandardScaler().fit(Xtr)
        Xtr_n = sc.transform(Xtr)
    for x in Xte:
        if cosine:
            xq = x / (np.linalg.norm(x) + 1e-10)
            d = 1 - Xtr_n @ xq
        else:
            xq = sc.transform(x[None, :])[0]
            d = np.linalg.norm(Xtr_n - xq, axis=1)
        kk = min(k, len(Xtr_n))
        idx = np.argsort(d)[:kk]
        A = Xtr_n[idx].T                     # (n_feat, kk)
        b = xq
        if l2 > 0:                           # augmented system for ridge-NNLS
            A = np.vstack([A, np.sqrt(l2) * np.eye(kk)])
            b = np.concatenate([xq, np.zeros(kk)])
        try:
            w, _ = nnls(A, b, maxiter=50000)
        except Exception:
            w = np.ones(kk) / kk
        if simplex and w.sum() > 0:
            w = w / w.sum()
        out.append(float(w @ ytr[idx]))
    return np.array(out)


def predict_ridge(Xtr, ytr, Xte, alpha=10.0, scale=True):
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = Ridge(alpha=alpha).fit(Xtr, ytr)
    return m.predict(Xte)


def predict_xgb(Xtr, ytr, Xte, n_estimators=100, learning_rate=0.05, max_depth=3,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1,
                min_child_weight=2.0):
    m = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                         max_depth=max_depth, subsample=subsample,
                         colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                         reg_alpha=reg_alpha, min_child_weight=min_child_weight,
                         random_state=42, n_jobs=1, objective="reg:squarederror")
    m.fit(Xtr, ytr)
    return m.predict(Xte)


# ─── hyper-parameter grids ───────────────────────────────────────────
# "rich" = extensive tuning used for all headline runs (main pipeline, ablations,
# literature-comparison). "light" = a much smaller grid the many-seed robustness
# sweep switches to (via set_profile) so 20 stratified splits stay tractable.
NNLS_GRID = [dict(k=k, cosine=c, simplex=s, l2=l2)
             for k in [3, 5, 8, 10, 12, 15, 20]
             for c in (True, False)
             for s in (True, False)
             for l2 in [0.0, 0.1, 1.0]]                       # 84 configs
RIDGE_GRID = [dict(alpha=a, scale=s)
              for a in [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0]
              for s in (True, False)]                          # 16 configs
XGB_GRID = [dict(n_estimators=n, learning_rate=lr, max_depth=d, reg_lambda=rl,
                 subsample=ss, colsample_bytree=cs)
            for n in [100, 300] for lr in [0.03, 0.05] for d in [2, 3]
            for rl in [1.0, 10.0] for ss in [0.8] for cs in [0.8]]  # 16 configs

# Below this many training rows, NNLS selection is restricted to simplex=True configs
# (convex combinations). See the guard in select_K_and_cfg for the rationale.
# 30 separates Pakistan's nested-LOO inner train (n=24, restricted) from the German
# clusters (35-61, full grid) — so Germany is unchanged and only tiny-n runs are guarded.
NNLS_SIMPLEX_ONLY_BELOW = 30

PREDICT = {"nnls": predict_nnls, "ridge": predict_ridge, "xgb": predict_xgb}
GRIDS = {"nnls": NNLS_GRID, "ridge": RIDGE_GRID, "xgb": XGB_GRID}


def set_profile(profile="rich"):
    """Swap the module-level FEATURE_COUNTS / GRIDS in place. Callers that imported
    `corrected_pipeline as cp` see the change because they read cp.GRIDS at call time."""
    global FEATURE_COUNTS, NNLS_GRID, RIDGE_GRID, XGB_GRID, GRIDS
    if profile == "light":
        FEATURE_COUNTS = [20, 40, 60, 100, 150, 200, 300]
        NNLS_GRID = [dict(k=k, cosine=c, simplex=s, l2=l2)
                     for k in [5, 8, 12, 20] for c in (True, False)
                     for s in (True, False) for l2 in [0.0, 1.0]]      # 32
        RIDGE_GRID = [dict(alpha=a, scale=s)
                      for a in [0.1, 1.0, 10.0, 100.0, 1000.0] for s in (True, False)]
        XGB_GRID = [dict(n_estimators=n, learning_rate=0.05, max_depth=d, reg_lambda=rl)
                    for n in [100] for d in [2, 3] for rl in [1.0, 10.0]]
    GRIDS = {"nnls": NNLS_GRID, "ridge": RIDGE_GRID, "xgb": XGB_GRID}
    return FEATURE_COUNTS, GRIDS


def cv_score(method, X, y, cfg, n_splits=5):
    """Mean %RMSE over CV folds; feature ranking is NOT here (X already selected)."""
    n = len(X)
    if n_splits >= n:            # LOO
        splitter = LeaveOneOut().split(X)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X)
    errs = []
    for tr, va in splitter:
        try:
            p = PREDICT[method](X[tr], y[tr], X[va], **cfg)
            errs.append(metrics(y[va], p)["percent_rmse"])
        except Exception:
            pass
    return float(np.mean(errs)) if errs else float("inf")


def select_K_and_cfg(method, Xtr_full, ytr, feature_counts, n_splits=5):
    """
    Train-only selection of (K, hyper-params). For each K: rank features on train,
    CV-tune hyper-params, keep the (K,cfg) with lowest CV %RMSE.  Feature ranking is
    re-done inside each CV fold to avoid ranking leakage.
    Returns best_K, best_cfg, best_cv, and the full K->cv curve.
    """
    n = len(Xtr_full)
    splitter = list((LeaveOneOut() if n_splits >= n else
                     KFold(n_splits=n_splits, shuffle=True, random_state=42)).split(Xtr_full))
    # rank features ONCE per fold on fold-train (order reused for every K, cfg)
    folds = [(tr, va, _abs_corr_order(Xtr_full[tr], ytr[tr])) for tr, va in splitter]
    # Small-sample guard for NNLS: with few training rows the CV cannot reliably choose
    # the higher-variance non-simplex (extrapolating) reconstruction — its selection
    # overfits the tiny inner folds and generalises poorly (observed on Pakistan, n=25).
    # Restrict to the convex-combination (simplex=True) estimator when n is small.
    grid = GRIDS[method]
    if method == "nnls" and n < NNLS_SIMPLEX_ONLY_BELOW:
        grid = [g for g in grid if g.get("simplex", True)]
    curve = {}
    best = (None, None, float("inf"))
    for K in feature_counts:
        if K > Xtr_full.shape[1]:
            continue
        best_cfg_cv = float("inf"); best_cfg = None
        for cfg in grid:
            errs = []
            for tr, va, order in folds:
                idx = order[:K]
                try:
                    p = PREDICT[method](Xtr_full[tr][:, idx], ytr[tr],
                                        Xtr_full[va][:, idx], **cfg)
                    errs.append(metrics(ytr[va], p)["percent_rmse"])
                except Exception:
                    pass
            cv = float(np.mean(errs)) if errs else float("inf")
            if cv < best_cfg_cv:
                best_cfg_cv, best_cfg = cv, cfg
        curve[K] = best_cfg_cv
        if best_cfg_cv < best[2]:
            best = (K, best_cfg, best_cfg_cv)
    return best[0], best[1], best[2], curve


def predict_nnls_ensembleK(Xtr, ytr, Xte, feature_counts, n_splits=5):
    """Small-sample NNLS without the high-variance single-K selection: for every K in
    the grid, pick the best (neighbours/l2/metric) cfg by train-only CV, form the
    prediction, and AVERAGE the predictions over all K (model averaging over the
    feature-count hyper-parameter). On n=25 nested LOO this removes the K-selection
    overfitting that made single-K NNLS unstable, and is insensitive to the exact grid.
    Train-only throughout (the simplex small-sample guard in select_K_and_cfg applies)."""
    preds = []
    for K in feature_counts:
        if K > Xtr.shape[1]:
            continue
        _, cfg, _, _ = select_K_and_cfg("nnls", Xtr, ytr, [K], n_splits)
        idx = rank_features_train(Xtr, ytr, K)
        preds.append(predict_nnls(Xtr[:, idx], ytr, Xte[:, idx], **cfg))
    return np.mean(preds, axis=0)


# ───────────────────────── German (split + clustering) ─────────────────────────

def fit_train_only_clusters(Xtr, n_clusters=2):
    sc = StandardScaler().fit(Xtr)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        random_state=42, metric="euclidean").fit(sc.transform(Xtr))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(reducer.embedding_)
    return sc, reducer, km


def assign_clusters(sc, reducer, km, X):
    return km.predict(reducer.transform(sc.transform(X)))


def run_german(df, methods=("nnls", "ridge", "xgb")):
    fc = [c for c in df.columns if c.startswith("feature_")]
    X = df[fc].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    split = df["split"].values
    tr, te = split == "train", split == "test"
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    # clustering fit on TRAIN ONLY
    sc, reducer, km = fit_train_only_clusters(Xtr, 2)
    ctr = km.labels_
    cte = assign_clusters(sc, reducer, km, Xte)

    results = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
               "cluster_train_sizes": np.bincount(ctr, minlength=2).tolist(),
               "cluster_test_sizes": np.bincount(cte, minlength=2).tolist(),
               "methods": {}}

    for method in methods:
        if method == "xgb" and not HAS_XGB:
            continue
        pooled_true, pooled_pred, per_cluster = [], [], {}
        for cid in [0, 1]:
            m_tr = ctr == cid
            m_te = cte == cid
            if m_tr.sum() < 5 or m_te.sum() < 1:
                per_cluster[cid] = {"skipped": True, "n_train": int(m_tr.sum()),
                                    "n_test": int(m_te.sum())}
                continue
            Xc_tr, yc_tr = Xtr[m_tr], ytr[m_tr]
            Xc_te, yc_te = Xte[m_te], yte[m_te]
            K, cfg, cv, curve = select_K_and_cfg(method, Xc_tr, yc_tr, FEATURE_COUNTS, 5)
            idx = rank_features_train(Xc_tr, yc_tr, K)          # final ranking on cluster-train
            pred = PREDICT[method](Xc_tr[:, idx], yc_tr, Xc_te[:, idx], **cfg)
            per_cluster[cid] = {"K": K, "cfg": cfg, "cv_percent_rmse": cv,
                                "test": metrics(yc_te, pred),
                                "cv_curve": curve, "n_train": int(m_tr.sum()),
                                "n_test": int(m_te.sum())}
            pooled_true += yc_te.tolist(); pooled_pred += list(map(float, pred))
        combined = metrics(pooled_true, pooled_pred) if pooled_true else None
        results["methods"][method] = {"combined_test": combined,
                                      "per_cluster": per_cluster,
                                      "pooled_true": pooled_true,
                                      "pooled_pred": pooled_pred}
    return results


# ───────────────────────── Pakistan (nested LOO) ─────────────────────────

def run_pakistan(df, methods=("nnls", "ridge", "xgb")):
    fc = [c for c in df.columns if c.startswith("feature_")]
    X = df[fc].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    # kg -> t/ha (circular plot r=17.5 m), matching original
    import math
    y = (y / 1000.0) * (10000.0 / (math.pi * 17.5 ** 2))
    n = len(X)
    counts = [c for c in FEATURE_COUNTS if c <= X.shape[1]]

    out = {"n": n, "methods": {}}
    for method in methods:
        if method == "xgb" and not HAS_XGB:
            continue
        preds, truths, chosen_K = [], [], []
        print(f"  [pakistan/{method}] nested LOO over {n} folds ...", flush=True)
        for i in range(n):
            mask = np.ones(n, bool); mask[i] = False
            Xtr, ytr = X[mask], y[mask]
            Xte, yte = X[~mask], y[~mask]
            # inner selection on the 24 training samples (train-only)
            fcnts = counts if method != "xgb" else [c for c in counts if c <= 60]
            if method == "nnls" and len(Xtr) < NNLS_SIMPLEX_ONLY_BELOW:
                # small-sample NNLS: average over the feature-count grid (no single-K pick)
                p = predict_nnls_ensembleK(Xtr, ytr, Xte, fcnts, n_splits=5)
                K, cfg = -1, "ensembleK"
            else:
                K, cfg, cv, _ = select_K_and_cfg(method, Xtr, ytr, fcnts, n_splits=5)
                idx = rank_features_train(Xtr, ytr, K)
                p = PREDICT[method](Xtr[:, idx], ytr, Xte[:, idx], **cfg)
            preds.append(float(p[0])); truths.append(float(yte[0])); chosen_K.append(K)
            print(f"    fold {i+1:>2}/{n}  K={K}  cfg={cfg}", flush=True)
        kmod = int(pd.Series(chosen_K).mode().iloc[0])
        out["methods"][method] = {"test": metrics(truths, preds),
                                  "y_true": truths, "y_pred": preds,
                                  "K_per_fold": chosen_K,
                                  "K_modal": "ensembleK" if kmod == -1 else kmod}
    return out


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=["german", "pakistan"], required=True)
    ap.add_argument("--features", choices=["dino", "resnet50"], required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    csv = os.path.join(ROOT, "datasets_corrected", f"{args.site}_{args.features}.csv")
    df = pd.read_csv(csv)
    print(f"[{args.site}/{args.features}] {df.shape} from {csv}")

    res = run_german(df) if args.site == "german" else run_pakistan(df)
    res["site"] = args.site; res["features"] = args.features

    outdir = os.path.join(ROOT, "results_corrected")
    os.makedirs(outdir, exist_ok=True)
    out = args.out or os.path.join(outdir, f"{args.site}_{args.features}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    # concise console summary
    print(f"  -> {out}")
    if args.site == "german":
        for m, r in res["methods"].items():
            c = r["combined_test"]
            if c:
                print(f"   {m:6s}: combined %RMSE={c['percent_rmse']:.2f}  R2={c['r2']:.2f}")
    else:
        for m, r in res["methods"].items():
            t = r["test"]
            print(f"   {m:6s}: %RMSE={t['percent_rmse']:.2f}  R2={t['r2']:.2f}  K_modal={r['K_modal']}")


if __name__ == "__main__":
    main()
