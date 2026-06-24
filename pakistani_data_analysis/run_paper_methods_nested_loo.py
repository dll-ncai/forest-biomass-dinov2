#!/usr/bin/env python3
"""
Literature Methods Comparison on Pakistani (Balakot) Dataset
============================================================
Implements hand-crafted feature pipelines from two reference papers and
evaluates them with Nested Leave-One-Out (LOO) cross-validation, matching
the same rigorous protocol used for our DINO-based models.

Papers:
  1. Liu et al. (Forests 2025, 16, 1777) – Texture + Vegetation Indices + RGB spectral
     with Random Forest, Gradient Boosting, XGBoost, and Stacking ensemble
  2. Eckert (Remote Sens. 2012, 4, 810) – GLCM texture with Stepwise multiple linear regression

Output:
  - paper_methods_nested_loo_results.csv
  - paper_methods_nested_loo_scatter.png
  - paper_methods_nested_loo_comparison.csv
"""

import os, json, glob, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy import stats

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────
# 1.  Feature extraction (identical to German pipeline)
# ──────────────────────────────────────────────────────────────────────

def extract_glcm_features(image: np.ndarray,
                           distances: List[int] = [1, 2, 3],
                           angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels: int = 256) -> Dict[str, float]:
    """Multi-scale GLCM texture features (remotesensing-04-00810 / forests-16-01777)."""
    if len(image.shape) == 3:
        gray = image[:, :, 1].astype(np.uint8)  # green band
    else:
        gray = image.astype(np.uint8)

    features: Dict[str, float] = {}
    try:
        for dist in distances:
            glcm = graycomatrix(gray, distances=[dist], angles=angles,
                                levels=levels, symmetric=True, normed=True)
            for prop in ["correlation", "homogeneity", "contrast", "energy"]:
                vals = graycoprops(glcm, prop)
                features[f"glcm_{prop}_d{dist}_mean"] = float(np.mean(vals))
                features[f"glcm_{prop}_d{dist}_std"]  = float(np.std(vals))
                features[f"glcm_{prop}_d{dist}_max"]  = float(np.max(vals))
                features[f"glcm_{prop}_d{dist}_min"]  = float(np.min(vals))
            asm = graycoprops(glcm, "energy")
            features[f"glcm_asm_d{dist}_mean"] = float(np.mean(asm))
            features[f"glcm_asm_d{dist}_std"]  = float(np.std(asm))

        # overall across all distances
        glcm_all = graycomatrix(gray, distances=distances, angles=angles,
                                levels=levels, symmetric=True, normed=True)
        for prop in ["correlation", "homogeneity", "contrast", "energy"]:
            vals = graycoprops(glcm_all, prop)
            features[f"glcm_{prop}_overall_mean"] = float(np.mean(vals))
            features[f"glcm_{prop}_overall_std"]  = float(np.std(vals))
        asm_all = graycoprops(glcm_all, "energy")
        features["glcm_asm_overall_mean"] = float(np.mean(asm_all))
        features["glcm_asm_overall_std"]  = float(np.std(asm_all))

        features["glcm_variance"] = float(np.var(gray))
        features["glcm_mean"]     = float(np.mean(gray))
    except Exception as e:
        print(f"  GLCM extraction warning: {e}")
    return features


def extract_vegetation_indices(image: np.ndarray) -> Dict[str, float]:
    """RGB-based vegetation indices (forests-16-01777)."""
    r = image[:, :, 0].astype(np.float32) / 255.0
    g = image[:, :, 1].astype(np.float32) / 255.0
    b = image[:, :, 2].astype(np.float32) / 255.0
    eps = 1e-8
    return {
        "vvi":   float(((g - r) / (g + r + eps)).mean()),
        "grvi":  float(((g - r) / (g + r + eps)).mean()),
        "rgri":  float((r / (g + eps)).mean()),
        "grri":  float((g / (r + eps)).mean()),
        "exg":   float((2*g - r - b).mean()),
        "exr":   float((1.4*r - g).mean()),
        "cive":  float((0.441*r - 0.811*g + 0.385*b + 18.78745).mean()),
        "vari":  float(((g - r) / (g + r - b + eps)).mean()),
        "tgi":   float((-0.5*(190*(r - g) - 120*(r - b))).mean()),
        "ngrdi": float(((g - r) / (g + r + eps)).mean()),
    }


def extract_spectral_statistics(image: np.ndarray) -> Dict[str, float]:
    """Band-wise statistics for R, G, B (forests-16-01777)."""
    features: Dict[str, float] = {}
    for i, ch in enumerate(["r", "g", "b"]):
        band = image[:, :, i].astype(np.float32)
        features[f"{ch}_mean"]   = float(np.mean(band))
        features[f"{ch}_std"]    = float(np.std(band))
        features[f"{ch}_min"]    = float(np.min(band))
        features[f"{ch}_max"]    = float(np.max(band))
        features[f"{ch}_median"] = float(np.median(band))
        features[f"{ch}_q25"]    = float(np.percentile(band, 25))
        features[f"{ch}_q75"]    = float(np.percentile(band, 75))
    return features


def extract_all_paper_features(img_path: str) -> Dict[str, float]:
    """Extract the complete 81-feature vector for one image crop."""
    img = np.array(Image.open(img_path).convert("RGB"))
    feats: Dict[str, float] = {}
    feats.update(extract_glcm_features(img))
    feats.update(extract_vegetation_indices(img))
    feats.update(extract_spectral_statistics(img))
    return feats

# ──────────────────────────────────────────────────────────────────────
# 2.  Model helpers
# ──────────────────────────────────────────────────────────────────────

def calc_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    prmse = float(rmse / np.mean(y_true) * 100) if np.mean(y_true) > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2, "%rmse": prmse}


# Hyperparameter grids for inner-loop tuning (Nested LOO)
RF_GRID = [
    {"n_estimators": n, "max_depth": d}
    for n in [50, 100]
    for d in [3, 5, 10, None]
]

GBT_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in [50, 100]
    for d in [3, 5]
    for lr in [0.05, 0.1]
]

XGB_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr, "reg_lambda": lam}
    for n in [50, 100]
    for d in [3, 5]
    for lr in [0.05, 0.1]
    for lam in [1.0, 10.0]
]


def _inner_loo_tune(X_train, y_train, model_cls, grid, extra_kw=None):
    """Inner LOO hyperparameter search. Returns best config dict."""
    n = len(y_train)
    best_cfg, best_prmse = None, 1e9
    for cfg in grid:
        preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            kw = {**cfg, "random_state": 42}
            if extra_kw:
                kw.update(extra_kw)
            m = model_cls(**kw)
            m.fit(X_train[mask], y_train[mask])
            preds[i] = m.predict(X_train[~mask].reshape(1, -1))[0]
        prmse = float(np.sqrt(mean_squared_error(y_train, preds)) / np.mean(y_train) * 100)
        if prmse < best_prmse:
            best_prmse = prmse
            best_cfg = cfg
    return best_cfg


def _inner_loo_tune_stacking(X_train, y_train):
    """Inner LOO for the Stacking ensemble (fixed architecture, tune base n_estimators)."""
    n = len(y_train)
    best_n, best_prmse = 50, 1e9
    for n_est in [50, 100]:
        preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            base = [
                ("rf",  RandomForestRegressor(n_estimators=n_est, max_depth=5, random_state=42, n_jobs=-1)),
                ("gbt", GradientBoostingRegressor(n_estimators=n_est, max_depth=3, learning_rate=0.1, random_state=42)),
                ("xgb", xgb.XGBRegressor(n_estimators=n_est, max_depth=3, learning_rate=0.1,
                                          random_state=42, verbosity=0)),
            ]
            stk = StackingRegressor(estimators=base, final_estimator=Ridge(alpha=10),
                                     cv=min(5, int(mask.sum())))
            stk.fit(X_train[mask], y_train[mask])
            preds[i] = stk.predict(X_train[~mask].reshape(1, -1))[0]
        prmse = float(np.sqrt(mean_squared_error(y_train, preds)) / np.mean(y_train) * 100)
        if prmse < best_prmse:
            best_prmse = prmse
            best_n = n_est
    return best_n


def _inner_loo_tune_stepwise(X_train, y_train, feature_names):
    """Inner LOO for stepwise regression. Returns selected feature indices."""
    # Forward selection on the full inner training set
    selected = []
    remaining = list(range(X_train.shape[1]))
    while remaining:
        best_feat, best_p = None, 1.0
        for fi in remaining:
            test_feats = selected + [fi]
            Xt = X_train[:, test_feats]
            model = LinearRegression(); model.fit(Xt, y_train)
            resid = y_train - model.predict(Xt)
            mse = np.mean(resid**2)
            n, p = len(y_train), len(test_feats)
            dof = n - p - 1
            if dof > 0 and mse > 0:
                try:
                    Xwi = np.column_stack([np.ones(n), Xt])
                    cov = mse * np.linalg.inv(Xwi.T @ Xwi)
                    se = np.sqrt(np.diag(cov)[1:])
                    t_stats = model.coef_ / (se + 1e-10)
                    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
                    if p_vals[-1] < best_p:
                        best_p = p_vals[-1]
                        best_feat = fi
                except Exception:
                    pass
        if best_feat is not None and best_p < 0.05:
            selected.append(best_feat)
            remaining.remove(best_feat)
        else:
            break
    return selected if selected else list(range(X_train.shape[1]))


# ──────────────────────────────────────────────────────────────────────
# 3.  Nested LOO driver per method
# ──────────────────────────────────────────────────────────────────────

def nested_loo(X, y, method, feature_names):
    """Run a full nested LOO for one method. Returns metrics, y_true, y_pred."""
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        Xtr, ytr = X[mask], y[mask]
        Xte = X[~mask].reshape(1, -1)

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        if method == "RF":
            cfg = _inner_loo_tune(Xtr_s, ytr, RandomForestRegressor, RF_GRID, {"n_jobs": -1})
            m = RandomForestRegressor(**cfg, random_state=42, n_jobs=-1)
            m.fit(Xtr_s, ytr)
            preds[i] = m.predict(Xte_s)[0]

        elif method == "GBT":
            cfg = _inner_loo_tune(Xtr_s, ytr, GradientBoostingRegressor, GBT_GRID)
            m = GradientBoostingRegressor(**cfg, random_state=42)
            m.fit(Xtr_s, ytr)
            preds[i] = m.predict(Xte_s)[0]

        elif method == "XGBoost":
            cfg = _inner_loo_tune(Xtr_s, ytr, xgb.XGBRegressor, XGB_GRID, {"verbosity": 0})
            m = xgb.XGBRegressor(**cfg, random_state=42, verbosity=0)
            m.fit(Xtr_s, ytr)
            preds[i] = m.predict(Xte_s)[0]

        elif method == "Stacking":
            best_n = _inner_loo_tune_stacking(Xtr_s, ytr)
            base = [
                ("rf",  RandomForestRegressor(n_estimators=best_n, max_depth=5, random_state=42, n_jobs=-1)),
                ("gbt", GradientBoostingRegressor(n_estimators=best_n, max_depth=3, learning_rate=0.1, random_state=42)),
                ("xgb", xgb.XGBRegressor(n_estimators=best_n, max_depth=3, learning_rate=0.1,
                                          random_state=42, verbosity=0)),
            ]
            stk = StackingRegressor(estimators=base, final_estimator=Ridge(alpha=10),
                                     cv=min(5, len(ytr)))
            stk.fit(Xtr_s, ytr)
            preds[i] = stk.predict(Xte_s)[0]

        elif method == "StepwiseRegression":
            sel_idx = _inner_loo_tune_stepwise(Xtr_s, ytr, feature_names)
            m = LinearRegression()
            m.fit(Xtr_s[:, sel_idx], ytr)
            preds[i] = m.predict(Xte_s[:, sel_idx])[0]

        else:
            raise ValueError(method)

    metrics = calc_metrics(y, preds)
    return metrics, y, preds


# ──────────────────────────────────────────────────────────────────────
# 4.  Main
# ──────────────────────────────────────────────────────────────────────

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 28,
    "axes.labelsize": 32,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 28,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

METHODS = ["RF", "GBT", "XGBoost", "Stacking", "StepwiseRegression"]
METHOD_DISPLAY = {
    "RF": "Random Forest",
    "GBT": "Gradient Boosting",
    "XGBoost": "XGBoost",
    "Stacking": "Stacking Ensemble",
    "StepwiseRegression": "Stepwise Regression",
}
METHOD_COLORS = {
    "RF": "#1f77b4",
    "GBT": "#ff7f0e",
    "XGBoost": "#2ca02c",
    "Stacking": "#d62728",
    "StepwiseRegression": "#9467bd",
}


def main():
    # Dataset paths
    crops_dir = "/Users/assadabid/Desktop/Balakot data for Prof. Shafait/biomass/dataset_crops_152"
    output_dir = SCRIPT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("LITERATURE METHODS — NESTED LOO ON PAKISTANI DATA")
    print("=" * 70)

    # ── Load crops and extract labels from filenames ──
    crop_files = sorted(glob.glob(os.path.join(crops_dir, "crop_*_label_*.png")))
    print(f"\nFound {len(crop_files)} image crops in {crops_dir}")

    paths, labels = [], []
    for p in crop_files:
        m = re.search(r"label_(\d+)", os.path.basename(p))
        if m:
            paths.append(p)
            labels.append(float(m.group(1)))

    labels = np.array(labels, dtype=np.float32)
    # Convert Total Biomass (kg) to Biomass Density (t/ha)
    # Circular field plot: radius = 17.5 m, area = π × 17.5² ≈ 962.1 m²
    import math
    circle_area_m2 = math.pi * 17.5 ** 2
    labels = (labels / 1000.0) * (10000.0 / circle_area_m2)

    print(f"Total samples: {len(labels)} (including {int((labels == 0).sum())} with biomass=0)")
    print(f"Biomass range: {labels.min():.2f} – {labels.max():.2f} t/ha")
    print(f"Biomass mean: {labels.mean():.2f} ± {labels.std():.2f} t/ha")

    # ── Extract features ──
    print("\nExtracting paper-method features (GLCM + VI + RGB stats)...")
    feat_dicts = []
    for i, fp in enumerate(paths):
        feat_dicts.append(extract_all_paper_features(fp))
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(paths)} crops")
    print(f"  Done. Features per crop: {len(feat_dicts[0])}")

    df_feat = pd.DataFrame(feat_dicts)
    feature_names = list(df_feat.columns)
    X = df_feat.values.astype(np.float32)
    y = labels

    # ── Run Nested LOO per method ──
    print(f"\nRunning Nested LOO ({len(y)} outer folds) for {len(METHODS)} methods...\n")
    results_rows = []
    scatter_data: Dict[str, Tuple] = {}

    for method in METHODS:
        print(f"  {METHOD_DISPLAY[method]} ...")
        metrics, yt, yp = nested_loo(X, y, method, feature_names)
        scatter_data[method] = (metrics, yt, yp)
        results_rows.append({
            "Method": METHOD_DISPLAY[method],
            "%RMSE": metrics["%rmse"],
            "R²": metrics["r2"],
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
        })
        print(f"    %RMSE = {metrics['%rmse']:.2f}%,  R² = {metrics['r2']:.4f}")

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(os.path.join(output_dir, "paper_methods_nested_loo_results.csv"), index=False)

    # ── Comparison table with our DINO-based results ──
    # Load our best DINO results
    dino_csv = os.path.join(output_dir, "nested_loo_best_vs_all.csv")
    comparison_rows = []
    if os.path.exists(dino_csv):
        dino_df = pd.read_csv(dino_csv)
        for _, row in dino_df.iterrows():
            comparison_rows.append({
                "Method": f"DINO + {row['Method']} (Top {int(row['Best Top K'])})",
                "%RMSE": row["Best K %RMSE"],
                "R²": row["Best K R²"],
                "Source": "Proposed (DINO)",
            })
    for _, row in results_df.iterrows():
        comparison_rows.append({
            "Method": row["Method"],
            "%RMSE": row["%RMSE"],
            "R²": row["R²"],
            "Source": "Literature",
        })
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(output_dir, "paper_methods_nested_loo_comparison.csv"), index=False)

    print("\n" + "=" * 70)
    print("COMPARISON: DINO-BASED (PROPOSED) vs LITERATURE METHODS")
    print("=" * 70)
    print(comp_df.to_string(index=False))

    # ── Save predictions for future scatter regeneration ──────────────────────
    pred_save = {}
    for method in METHODS:
        m, yt, yp = scatter_data[method]
        pred_save[method] = {
            "y_true": yt.tolist(), "y_pred": yp.tolist(),
            "prmse": m["%rmse"], "r2": m["r2"],
        }
    pred_path = os.path.join(output_dir, "paper_methods_nested_loo_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(pred_save, f, indent=2)
    print(f"Predictions saved: {pred_path}")

    # ── Scatter plots (publication-ready) ─────────────────────────────────────
    global_mn, global_mx = np.inf, -np.inf
    for method in METHODS:
        _, yt, yp = scatter_data[method]
        global_mn = min(global_mn, yt.min(), yp.min())
        global_mx = max(global_mx, yt.max(), yp.max())
    pad = (global_mx - global_mn) * 0.07
    axis_mn = global_mn - pad
    axis_mx = global_mx + pad

    letters = "abcde"
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    for i, method in enumerate(METHODS):
        metrics, yt, yp = scatter_data[method]
        ax = axes[i]
        ax.scatter(yt, yp, color=METHOD_COLORS[method], alpha=0.85,
                   edgecolors="white", linewidths=0.5, s=90, zorder=3)
        ax.plot([axis_mn, axis_mx], [axis_mn, axis_mx], "k--", lw=1.5, alpha=0.6, zorder=2)
        ax.set_xlim(axis_mn, axis_mx)
        ax.set_ylim(axis_mn, axis_mx)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True AGB (t/ha)", fontsize=32, labelpad=6)
        ax.set_ylabel("Predicted AGB (t/ha)", fontsize=32, labelpad=6)
        ax.set_title(METHOD_DISPLAY[method], fontsize=30, fontweight="bold", pad=8)
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.text(0.97, 0.05,
                f"R² = {metrics['r2']:.3f}\n%RMSE = {metrics['%rmse']:.1f}%\nN = {len(yt)}",
                transform=ax.transAxes, fontsize=26,
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#AAAAAA", alpha=0.85))
        ax.text(-0.10, 1.04, f"({letters[i]})",
                transform=ax.transAxes,
                fontsize=40, fontweight="bold", va="bottom", ha="left")

    for j in range(len(METHODS), len(axes)):
        axes[j].axis("off")

    for ext in ("svg", "png"):
        out = os.path.join(output_dir, f"paper_methods_nested_loo_scatter.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none", pad_inches=0.15)
        print(f"Saved: {out}")
    plt.close(fig)

    print(f"\n✅ All paper-method artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
