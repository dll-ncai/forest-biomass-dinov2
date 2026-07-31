#!/usr/bin/env python3
"""
Germany robustness: repeat the full evaluation over MANY stratified
85/15 splits (default 20 seeds) so conclusions don't rest on one noisy split.

For each seed and each feature set we evaluate, all train-only:
  * proposed     : train-only UMAP+KMeans clustering + per-cluster train-only K-select
  * no_cluster   : single global model + train-only K-select
  * all_features : train-only clustering, all features (no selection)
And the hand-crafted GLCM literature baseline (RF/GBT/XGB/Ridge, CV-tuned on train).

Reports mean +/- std of combined test %RMSE and R^2 per (config, model).
"""
import os, json, argparse, math
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import xgboost as xgb
import pipeline as cp

ROOT = os.path.dirname(os.path.abspath(__file__))
DEEP_METHODS = ("nnls", "ridge", "xgb")
LIT_MODELS = {
    "rf":  (RandomForestRegressor, [dict(n_estimators=100, max_depth=d) for d in [3, 5, None]]),
    "gbt": (GradientBoostingRegressor, [dict(n_estimators=100, max_depth=d, learning_rate=lr) for d in [2, 3] for lr in [0.05, 0.1]]),
    "xgb": (xgb.XGBRegressor, [dict(n_estimators=100, max_depth=d, learning_rate=0.05, reg_lambda=lam) for d in [2, 3] for lam in [1.0, 10.0]]),
    "ridge": (Ridge, [dict(alpha=a) for a in [1.0, 10.0, 100.0, 1000.0]]),
}


def strat_train_mask(y, seed, test_frac=0.15):
    bins = pd.qcut(y, q=min(5, len(np.unique(y))), labels=False, duplicates="drop")
    idx = np.arange(len(y))
    tr, _ = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=bins)
    m = np.zeros(len(y), bool); m[tr] = True
    return m


def eval_proposed(Xtr, ytr, Xte, yte):
    sc, red, km = cp.fit_train_only_clusters(Xtr, 2)
    ctr, cte = km.labels_, cp.assign_clusters(sc, red, km, Xte)
    out = {}
    for m in DEEP_METHODS:
        if m == "xgb" and not cp.HAS_XGB:
            continue
        pt, pp = [], []
        for cid in [0, 1]:
            a, b = ctr == cid, cte == cid
            if a.sum() < 5 or b.sum() < 1:
                continue
            K, cfg, _, _ = cp.select_K_and_cfg(m, Xtr[a], ytr[a], cp.FEATURE_COUNTS, 5)
            idx = cp.rank_features_train(Xtr[a], ytr[a], K)
            pred = cp.PREDICT[m](Xtr[a][:, idx], ytr[a], Xte[b][:, idx], **cfg)
            pt += yte[b].tolist(); pp += list(map(float, pred))
        out[m] = cp.metrics(pt, pp) if pt else None
    return out


def eval_no_cluster(Xtr, ytr, Xte, yte):
    out = {}
    for m in DEEP_METHODS:
        if m == "xgb" and not cp.HAS_XGB:
            continue
        K, cfg, _, _ = cp.select_K_and_cfg(m, Xtr, ytr, cp.FEATURE_COUNTS, 5)
        idx = cp.rank_features_train(Xtr, ytr, K)
        out[m] = cp.metrics(yte, cp.PREDICT[m](Xtr[:, idx], ytr, Xte[:, idx], **cfg))
    return out


def eval_all_features(Xtr, ytr, Xte, yte):
    sc, red, km = cp.fit_train_only_clusters(Xtr, 2)
    ctr, cte = km.labels_, cp.assign_clusters(sc, red, km, Xte)
    nf = Xtr.shape[1]; out = {}
    for m in DEEP_METHODS:
        if m == "xgb" and not cp.HAS_XGB:
            continue
        pt, pp = [], []
        for cid in [0, 1]:
            a, b = ctr == cid, cte == cid
            if a.sum() < 5 or b.sum() < 1:
                continue
            _, cfg, _, _ = cp.select_K_and_cfg(m, Xtr[a], ytr[a], [nf], 5)
            pred = cp.PREDICT[m](Xtr[a], ytr[a], Xte[b], **cfg)
            pt += yte[b].tolist(); pp += list(map(float, pred))
        out[m] = cp.metrics(pt, pp) if pt else None
    return out


def eval_literature(Xtr, ytr, Xte, yte):
    def cv_tune(cls, grid, X, y):
        best, bcv = None, 1e18
        for cfg in grid:
            errs = []
            for tr, va in KFold(5, shuffle=True, random_state=42).split(X):
                kw = dict(cfg)
                if cls is not Ridge: kw["random_state"] = 42
                if cls is xgb.XGBRegressor: kw["verbosity"] = 0
                if cls is RandomForestRegressor: kw["n_jobs"] = -1
                mdl = cls(**kw).fit(X[tr], y[tr])
                errs.append(cp.metrics(y[va], mdl.predict(X[va]))["percent_rmse"])
            if np.mean(errs) < bcv:
                bcv, best = np.mean(errs), cfg
        return best
    out = {}
    for name, (cls, grid) in LIT_MODELS.items():
        cfg = cv_tune(cls, grid, Xtr, ytr)
        kw = dict(cfg)
        if cls is not Ridge: kw["random_state"] = 42
        if cls is xgb.XGBRegressor: kw["verbosity"] = 0
        if cls is RandomForestRegressor: kw["n_jobs"] = -1
        out[name] = cp.metrics(yte, cls(**kw).fit(Xtr, ytr).predict(Xte))
    return out


def load(name):
    df = pd.read_csv(os.path.join(ROOT, "datasets", f"german_{name}.csv"))
    fc = [c for c in df.columns if c.startswith("feature_")]
    return df[fc].values.astype(np.float32), df["label"].values.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--profile", choices=["rich", "light"], default="light",
                    help="light = smaller grid so many seeds stay tractable")
    args = ap.parse_args()

    cp.set_profile(args.profile)
    print(f"[robustness] profile={args.profile} feature_counts={cp.FEATURE_COUNTS} "
          f"nnls_grid={len(cp.NNLS_GRID)} ridge_grid={len(cp.RIDGE_GRID)} "
          f"xgb_grid={len(cp.XGB_GRID)}", flush=True)

    data = {n: load(n) for n in ["dino", "resnet50", "glcm"]}
    y = data["dino"][1]
    acc = {}  # (feat, config, model) -> list of (prmse, r2)

    def add(feat, cfgname, res):
        for m, v in res.items():
            if v is None:
                continue
            acc.setdefault((feat, cfgname, m), []).append((v["percent_rmse"], v["r2"]))

    for s in range(args.seeds):
        tr = strat_train_mask(y, seed=1000 + s)
        for feat in ["dino", "resnet50"]:
            X, yy = data[feat]
            Xtr, ytr, Xte, yte = X[tr], yy[tr], X[~tr], yy[~tr]
            add(feat, "proposed", eval_proposed(Xtr, ytr, Xte, yte))
            add(feat, "no_cluster", eval_no_cluster(Xtr, ytr, Xte, yte))
            add(feat, "all_features", eval_all_features(Xtr, ytr, Xte, yte))
        Xg, yg = data["glcm"]
        add("glcm", "literature", eval_literature(Xg[tr], yg[tr], Xg[~tr], yg[~tr]))
        print(f"seed {s+1}/{args.seeds} done", flush=True)

    summary = {}
    for (feat, cfg, m), vals in acc.items():
        a = np.array(vals)
        summary[f"{feat}|{cfg}|{m}"] = {
            "prmse_mean": float(a[:, 0].mean()), "prmse_std": float(a[:, 0].std()),
            "r2_mean": float(a[:, 1].mean()), "r2_std": float(a[:, 1].std()),
            "n_seeds": len(vals)}
    outp = os.path.join(ROOT, "results", "robustness_german.json")
    json.dump({"seeds": args.seeds, "summary": summary}, open(outp, "w"), indent=2)
    print("=== GERMANY ROBUSTNESS (mean±std over %d splits) ===" % args.seeds)
    for k in sorted(summary):
        s = summary[k]
        print(f"  {k:32s} %RMSE={s['prmse_mean']:5.2f}±{s['prmse_std']:4.2f}  "
              f"R2={s['r2_mean']:.2f}±{s['r2_std']:.2f}")
    print("ROBUSTNESS_DONE")


if __name__ == "__main__":
    main()
