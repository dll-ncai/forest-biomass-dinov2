#!/usr/bin/env python3
"""
Corrected (leak-free) ablations for the German site, mirroring the paper's two
ablations but with train-only clustering / ranking / K-selection:

  A1  all-features  : train-only UMAP+KMeans clustering, but use ALL features per
                      cluster (no correlation selection). Hyper-params tuned by
                      5-fold CV on cluster-train; evaluated on cluster-test (pooled).
  A2  no-clustering : a single global model (no clustering). Train-only feature
                      ranking + train-only K selection (5-fold CV on train);
                      evaluated once on the held-out test set.

Run for both feature backbones.  Results -> results_corrected/ablations_*.json
"""
import os, json
import numpy as np, pandas as pd
import corrected_pipeline as cp

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(site, feat):
    return pd.read_csv(os.path.join(ROOT, "datasets_corrected", f"{site}_{feat}.csv"))


def german_all_features(df, methods=("nnls", "ridge", "xgb")):
    fc = [c for c in df.columns if c.startswith("feature_")]
    X = df[fc].values.astype(np.float32); y = df["label"].values.astype(np.float32)
    sp = df["split"].values; tr, te = sp == "train", sp == "test"
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    sc, red, km = cp.fit_train_only_clusters(Xtr, 2)
    ctr, cte = km.labels_, cp.assign_clusters(sc, red, km, Xte)
    out = {}
    for m in methods:
        if m == "xgb" and not cp.HAS_XGB:
            continue
        pt, pp = [], []
        for cid in [0, 1]:
            mtr, mte = ctr == cid, cte == cid
            if mtr.sum() < 5 or mte.sum() < 1:
                continue
            nf = X.shape[1]
            # K fixed to ALL features -> select_K_and_cfg only tunes hyper-params
            _, cfg, cv, _ = cp.select_K_and_cfg(m, Xtr[mtr], ytr[mtr], [nf], 5)
            pred = cp.PREDICT[m](Xtr[mtr], ytr[mtr], Xte[mte], **cfg)
            pt += yte[mte].tolist(); pp += list(map(float, pred))
        out[m] = cp.metrics(pt, pp) if pt else None
    return out


def german_no_clustering(df, methods=("nnls", "ridge", "xgb")):
    fc = [c for c in df.columns if c.startswith("feature_")]
    X = df[fc].values.astype(np.float32); y = df["label"].values.astype(np.float32)
    sp = df["split"].values; tr, te = sp == "train", sp == "test"
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    out = {}
    for m in methods:
        if m == "xgb" and not cp.HAS_XGB:
            continue
        K, cfg, cv, curve = cp.select_K_and_cfg(m, Xtr, ytr, cp.FEATURE_COUNTS, 5)
        idx = cp.rank_features_train(Xtr, ytr, K)
        pred = cp.PREDICT[m](Xtr[:, idx], ytr, Xte[:, idx], **cfg)
        out[m] = {"K": K, "cfg": cfg, "cv_percent_rmse": cv, "test": cp.metrics(yte, pred)}
    return out


def main():
    outdir = os.path.join(ROOT, "results_corrected"); os.makedirs(outdir, exist_ok=True)
    for feat in ["dino", "resnet50"]:
        df = load("german", feat)
        res = {"features": feat,
               "all_features": german_all_features(df),
               "no_clustering": german_no_clustering(df)}
        with open(os.path.join(outdir, f"ablations_german_{feat}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"=== german / {feat} ablations ===")
        for m, v in res["all_features"].items():
            if v: print(f"  all-features {m:6s}: %RMSE={v['percent_rmse']:.2f} R2={v['r2']:.2f}")
        for m, v in res["no_clustering"].items():
            t = v["test"]; print(f"  no-cluster  {m:6s}: %RMSE={t['percent_rmse']:.2f} R2={t['r2']:.2f} K={v['K']}")
    print("ABLATIONS_DONE")


if __name__ == "__main__":
    main()
