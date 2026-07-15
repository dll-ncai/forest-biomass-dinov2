#!/usr/bin/env python3
"""
Leak-free hand-crafted (GLCM + vegetation-index + spectral) literature baselines,
evaluated with the SAME protocol as the corrected DINO/ResNet50 pipeline:
  * Pakistan : nested LOO (hyper-params tuned by 5-fold CV on the inner training set)
  * Germany  : the identical 85/15 split used by the DINO pipeline; hyper-params
               tuned by 5-fold CV on train; single evaluation on test.

Features (97-d) are extracted from the SAME image crops used for the deep features,
so the comparison is on identical patches. No test information is used for tuning
and there is no feature-count selected on test.
"""
import os, sys, glob, re, json, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pakistani_data_analysis"))
from run_paper_methods_nested_loo import extract_all_paper_features
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneOut
import xgboost as xgb
import corrected_pipeline as cp

ROOT = os.path.dirname(os.path.abspath(__file__))
PAK_CROPS = os.path.join(ROOT, "dataset_crops_152")
GER_CROPS = os.path.join(ROOT, "german_data_results", "complete_pipeline_dataset", "crops")

MODELS = {
    "rf":    (RandomForestRegressor, [dict(n_estimators=n, max_depth=d) for n in [100] for d in [3, 5, None]]),
    "gbt":   (GradientBoostingRegressor, [dict(n_estimators=n, max_depth=d, learning_rate=lr) for n in [100] for d in [2, 3] for lr in [0.05, 0.1]]),
    "xgb":   (xgb.XGBRegressor, [dict(n_estimators=n, max_depth=d, learning_rate=lr, reg_lambda=lam) for n in [100] for d in [2, 3] for lr in [0.05] for lam in [1.0, 10.0]]),
    "ridge": (Ridge, [dict(alpha=a) for a in [1.0, 10.0, 100.0, 1000.0]]),
}


def build(model_cls, cfg):
    kw = dict(cfg)
    if model_cls is not Ridge:
        kw["random_state"] = 42
    if model_cls is xgb.XGBRegressor:
        kw["verbosity"] = 0
    if model_cls is RandomForestRegressor:
        kw["n_jobs"] = -1
    return model_cls(**kw)


def cv_tune(model_cls, grid, X, y, n_splits=5):
    n = len(y)
    sp = list((LeaveOneOut() if n_splits >= n else KFold(n_splits, shuffle=True, random_state=42)).split(X))
    best, best_cv = None, float("inf")
    for cfg in grid:
        errs = []
        for tr, va in sp:
            m = build(model_cls, cfg).fit(X[tr], y[tr])
            errs.append(cp.metrics(y[va], m.predict(X[va]))["percent_rmse"])
        cv = float(np.mean(errs))
        if cv < best_cv:
            best_cv, best = cv, cfg
    return best


def extract_matrix(items):
    """items: list of (path, label). Returns X (n x d), y, feature_names."""
    rows, ys = [], []
    for path, lab in items:
        rows.append(extract_all_paper_features(path)); ys.append(lab)
    df = pd.DataFrame(rows).fillna(0.0)
    return df.values.astype(np.float64), np.array(ys, float), list(df.columns)


def pak_items():
    it = []
    for f in glob.glob(os.path.join(PAK_CROPS, "crop_*_label_*.png")):
        lab = float(re.search(r"label_(\d+)", os.path.basename(f)).group(1))
        it.append((int(re.search(r"crop_(\d+)", os.path.basename(f)).group(1)), f, lab))
    it.sort()
    return [(f, l) for _, f, l in it]


def ger_items():
    df = pd.read_csv(os.path.join(ROOT, "datasets_corrected", "german_dino.csv"))
    items, splits = [], []
    for _, r in df.iterrows():
        items.append((os.path.join(GER_CROPS, r["source_file"]), float(r["label"])))
        splits.append(r["split"])
    return items, np.array(splits)


def run_pakistan():
    items = pak_items()
    X, y, names = extract_matrix(items)
    y = (y / 1000.0) * (10000.0 / (math.pi * 17.5 ** 2))   # kg -> t/ha
    n = len(y); out = {}
    for name, (cls, grid) in MODELS.items():
        preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, bool); mask[i] = False
            cfg = cv_tune(cls, grid, X[mask], y[mask], 5)
            preds[i] = build(cls, cfg).fit(X[mask], y[mask]).predict(X[~mask])[0]
        out[name] = cp.metrics(y, preds)
        print(f"  PAK {name:5s}: %RMSE={out[name]['percent_rmse']:.2f} R2={out[name]['r2']:.2f}")
    return out


def run_german():
    items, splits = ger_items()
    X, y, names = extract_matrix(items)
    tr, te = splits == "train", splits == "test"
    out = {}
    for name, (cls, grid) in MODELS.items():
        cfg = cv_tune(cls, grid, X[tr], y[tr], 5)
        pred = build(cls, cfg).fit(X[tr], y[tr]).predict(X[te])
        out[name] = cp.metrics(y[te], pred)
        print(f"  GER {name:5s}: %RMSE={out[name]['percent_rmse']:.2f} R2={out[name]['r2']:.2f}")
    return out


def main():
    outdir = os.path.join(ROOT, "results_corrected"); os.makedirs(outdir, exist_ok=True)
    print("=== corrected literature baselines: Germany (85/15) ===")
    ger = run_german()
    print("=== corrected literature baselines: Pakistan (nested LOO) ===")
    pak = run_pakistan()
    json.dump({"german": ger, "pakistan": pak},
              open(os.path.join(outdir, "literature_baselines.json"), "w"), indent=2)
    print("LITERATURE_DONE")


if __name__ == "__main__":
    main()
