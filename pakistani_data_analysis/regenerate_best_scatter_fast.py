#!/usr/bin/env python3
"""
Fast regeneration of nested_loo_best_scatter_plots using the same LOO code
as run_nested_loo.py, but only for the 3 authoritative best configs:
  NNLS   @ top-90   -> authoritative %RMSE=21.81%, R²=0.939
  Ridge  @ top-130  -> authoritative %RMSE=24.60%, R²=0.923
  XGB    @ top-40   -> authoritative %RMSE=49.64%, R²=0.685

Outputs: nested_loo_best_scatter_plots.svg + .png
"""
import os, sys, math, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_best_models_top_features import (
    get_feature_and_label_columns,
    load_ranked_feature_indices,
    tune_nnls_loo, tune_ridge_loo, tune_xgb_loo,
    predict_nnls, predict_linear, predict_xgb,
    compute_metrics,
    METHOD_DISPLAY, METHOD_COLORS,
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 24,
    "axes.labelsize": 27,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)

BEST = {"nnls": 90, "linear": 130, "xgb": 40}
METHOD_NAMES  = {"nnls": "NNLS", "linear": "Ridge Regression", "xgb": "XGBoost"}
LETTERS = "abc"


def nested_loo_single(X, y, method):
    n = len(y)
    preds, truths = [], []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        Xtr, ytr = X[mask], y[mask]
        Xte = X[~mask]

        if method == "nnls":
            cfg, _, _, _ = tune_nnls_loo(Xtr, ytr)
            p = predict_nnls(Xtr, ytr, Xte,
                             cfg.get("neighbor_k", 3),
                             cfg.get("cosine", True),
                             cfg.get("simplex", True))
        elif method == "linear":
            cfg, _, _, _ = tune_ridge_loo(Xtr, ytr)
            p = predict_linear(Xtr, ytr, Xte,
                               cfg.get("scale", True),
                               cfg.get("alpha", 10.0))
        elif method == "xgb":
            cfg, _, _, _ = tune_xgb_loo(Xtr, ytr)
            p = predict_xgb(Xtr, ytr, Xte, cfg)
        preds.append(float(p[0]))
        truths.append(float(y[~mask][0]))

    yt = np.array(truths)
    yp = np.array(preds)
    return yt, yp, compute_metrics(yt, yp)


def main():
    input_csv       = os.path.join(BASE_DIR, "dino_features_with_labels_and_split.csv")
    correlations_csv = os.path.join(SCRIPT_DIR, "feature_correlations.csv")

    print("Loading DINO features...")
    df = pd.read_csv(input_csv)
    feature_cols, label_col = get_feature_and_label_columns(df)
    X_full = df[feature_cols].values.astype(np.float32)
    y_full = df[label_col].values.astype(np.float32)

    circle_area_m2 = math.pi * 17.5 ** 2
    y_full = (y_full / 1000.0) * (10000.0 / circle_area_m2)
    print(f"  {len(y_full)} samples, biomass {y_full.min():.2f}–{y_full.max():.2f} t/ha")

    print("Loading feature ranking...")
    ranked_indices = load_ranked_feature_indices(correlations_csv, feature_cols)

    panels = {}
    for m_key, best_k in BEST.items():
        t0 = time.time()
        print(f"\nRunning nested LOO: {METHOD_NAMES[m_key]} top-{best_k} ...")
        top_idx = ranked_indices[:best_k]
        X = X_full[:, top_idx]
        yt, yp, metrics = nested_loo_single(X, y_full, m_key)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s — %RMSE={metrics['%rmse']:.2f}%  R²={metrics['r2']:.4f}")
        panels[m_key] = (yt, yp, metrics)

    print("\nGenerating figure...")
    all_vals = []
    for yt, yp, _ in panels.values():
        all_vals.extend(yt.tolist() + yp.tolist())
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.07
    ax_lo, ax_hi = lo - pad, hi + pad

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    fig.subplots_adjust(wspace=0.35, top=0.85, bottom=0.12, left=0.07, right=0.97)

    for idx, (m_key, best_k) in enumerate(BEST.items()):
        yt, yp, metrics = panels[m_key]
        ax = axes[idx]
        ax.scatter(yt, yp, color=METHOD_COLORS[m_key], alpha=0.85,
                   edgecolors="white", linewidths=0.6, s=100, zorder=3)
        ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], "k--", lw=1.5, alpha=0.6, zorder=2)
        ax.set_xlim(ax_lo, ax_hi); ax.set_ylim(ax_lo, ax_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True AGB (t/ha)", fontsize=27, labelpad=6)
        ax.set_ylabel("Predicted AGB (t/ha)", fontsize=27, labelpad=6)
        ax.set_title(f"{METHOD_NAMES[m_key]} (top {best_k})",
                     fontsize=22, pad=8)
        ax.grid(True, lw=0.5, alpha=0.25, ls=':')
        ax.text(0.97, 0.05,
                f"R² = {metrics['r2']:.3f}\n%RMSE = {metrics['%rmse']:.1f}%\nN = {len(yt)}",
                transform=ax.transAxes, fontsize=22, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#AAAAAA", alpha=0.85))
        ax.text(-0.10, 1.10, f"({LETTERS[idx]})",
                transform=ax.transAxes,
                fontsize=34, fontweight="bold", va="bottom", ha="left")

    for ext in ("svg", "png"):
        out = os.path.join(SCRIPT_DIR, f"nested_loo_best_scatter_plots.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none", pad_inches=0.15)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
