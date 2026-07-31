#!/usr/bin/env python3
"""
Regenerate key paper figures from the German experiment results:
  Fig 5 — Combined %RMSE vs N features (2-panel line plot)
  Fig 6 — Combined scatter plots for best configs (NNLS@20F, Ridge@40F, XGB@30F)

Output: SVG (publication-ready) + PNG (backup) saved to
  german_data_results/augmented_zero_biomass_analysis/plots/
  paper_writeup/
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── Publication style ────────────────────────────────────────────────────────
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

BASE      = os.path.dirname(os.path.abspath(__file__))
RES_DIR   = os.path.join(BASE, "augmented_zero_biomass_analysis",
                          "cluster_separated_distance_analysis")
OUT_DIR   = os.path.join(BASE, "augmented_zero_biomass_analysis", "plots")
PAPER_DIR = os.path.join(BASE, "..", "paper_writeup")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COUNTS = list(range(20, 301, 10))
BEST = {"nnls": 20, "linear": 40, "xgb": 30}
METHOD_NAMES  = {"nnls": "NNLS", "linear": "Ridge Regression", "xgb": "XGBoost"}
METHOD_COLORS = {"nnls": "#A23B72", "linear": "#2E86AB", "xgb": "#F18F01"}


def combined_metrics(r0, r1):
    y_true = np.concatenate([np.array(r0["true"]),        np.array(r1["true"])])
    y_pred = np.concatenate([np.array(r0["predictions"]), np.array(r1["predictions"])])
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    prmse = rmse / float(np.mean(y_true)) * 100
    r2    = float(r2_score(y_true, y_pred))
    mae   = float(mean_absolute_error(y_true, y_pred))
    return dict(rmse=rmse, prmse=prmse, r2=r2, mae=mae,
                y_true=y_true, y_pred=y_pred)


def subplot_label(ax, letter, fontsize=30):
    """Place (a), (b), … at top-left, outside the plot frame."""
    ax.text(-0.10, 1.04, f"({letter})",
            transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va="bottom", ha="left")


# ── Load full sweep ──────────────────────────────────────────────────────────
sweep = {m: [] for m in ["nnls", "linear", "xgb"]}
best_data = {}

for n in FEATURE_COUNTS:
    fpath = os.path.join(RES_DIR, f"results_top{n}_features.json")
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        d = json.load(f)
    c0, c1 = d.get("cluster_0", {}), d.get("cluster_1", {})
    for method in ["nnls", "linear", "xgb"]:
        r0, r1 = c0.get(method), c1.get(method)
        if r0 and r1:
            m = combined_metrics(r0, r1)
            sweep[method].append({"n": n, **m})
            if n == BEST[method]:
                best_data[method] = m


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — %RMSE and R² vs number of features
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.6))
fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.13, wspace=0.32)

for method in ["nnls", "linear", "xgb"]:
    df_m = pd.DataFrame(sweep[method]).sort_values("n")
    col  = METHOD_COLORS[method]
    name = METHOD_NAMES[method]
    axes[0].plot(df_m["n"], df_m["prmse"], "-o", markersize=5,
                 color=col, label=name, linewidth=1.8)
    axes[1].plot(df_m["n"], df_m["r2"],    "-o", markersize=5,
                 color=col, label=name, linewidth=1.8)
    # Star marker at best point
    best_n = BEST[method]
    brow = df_m[df_m["n"] == best_n].iloc[0]
    axes[0].scatter([best_n], [brow.prmse], s=120, color=col,
                    zorder=5, marker="*", edgecolors="k", linewidths=0.5)
    axes[1].scatter([best_n], [brow.r2],    s=120, color=col,
                    zorder=5, marker="*", edgecolors="k", linewidths=0.5)

for ax, ylabel, letter in [
    (axes[0], "%RMSE (%)", "a"),
    (axes[1], "R²",        "b"),
]:
    ax.set_xlabel("Number of Features", fontsize=27, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=27, labelpad=6)
    ax.legend(fontsize=24, frameon=True, edgecolor="#AAAAAA",
              loc="upper right" if ylabel == "%RMSE (%)" else "lower right")
    ax.grid(True, alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    subplot_label(ax, letter)

for ext in ("svg", "png"):
    for dest in [d for d in (OUT_DIR, PAPER_DIR) if os.path.isdir(d)]:
        stem = "german_performance_vs_features"
        out = os.path.join(dest, f"{stem}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Figure 6 — Best-config scatter plots (consistent axis scale)
# ════════════════════════════════════════════════════════════════════════════

# Compute global axis limits across all 3 methods for consistent scale
all_true, all_pred = [], []
for method in ["nnls", "linear", "xgb"]:
    bd = best_data[method]
    all_true.extend(bd["y_true"].tolist())
    all_pred.extend(bd["y_pred"].tolist())
axis_min = min(min(all_true), min(all_pred)) - 8
axis_max = max(max(all_true), max(all_pred)) + 8

fig, axes = plt.subplots(1, 3, figsize=(19.8, 7.7))
fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.12,
                    wspace=0.30)

for ax, method, letter in zip(axes, ["nnls", "linear", "xgb"], ["a", "b", "c"]):
    bd   = best_data[method]
    col  = METHOD_COLORS[method]
    name = METHOD_NAMES[method]
    n    = BEST[method]

    # Load cluster labels for colour-coding
    fpath = os.path.join(RES_DIR, f"results_top{n}_features.json")
    with open(fpath) as f:
        d = json.load(f)
    r0 = d["cluster_0"][method]
    r1 = d["cluster_1"][method]
    yt0, yp0 = np.array(r0["true"]), np.array(r0["predictions"])
    yt1, yp1 = np.array(r1["true"]), np.array(r1["predictions"])

    ax.scatter(yt0, yp0, color="#2196F3", edgecolors="k", s=70, alpha=0.85,
               linewidths=0.6, label=f"Cluster 0 (n={len(yt0)})", zorder=3)
    ax.scatter(yt1, yp1, color="#FF9800", edgecolors="k", s=70, alpha=0.85,
               linewidths=0.6, label=f"Cluster 1 (n={len(yt1)})", zorder=3)

    # 1:1 line using global limits
    ax.plot([axis_min, axis_max], [axis_min, axis_max],
            "k--", lw=1.8, alpha=0.7, label="1:1 line")
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Measured AGB (t/ha)", fontsize=27, labelpad=6)
    ax.set_ylabel("Predicted AGB (t/ha)", fontsize=27, labelpad=6)

    # Metrics annotation (inside lower-right)
    ax.text(0.97, 0.05,
            f"%RMSE = {bd['prmse']:.2f}%\n$R^2$ = {bd['r2']:.3f}",
            transform=ax.transAxes, fontsize=22,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#AAAAAA", alpha=0.9))

    ax.legend(fontsize=22, frameon=True, edgecolor="#AAAAAA",
              loc="upper left")
    ax.grid(True, alpha=0.25, linewidth=0.5, ls=':')
    ax.set_axisbelow(True)
    ax.set_title(f"{name} — Top {n} features", fontsize=20, pad=8)
    subplot_label(ax, letter)

for ext in ("svg", "png"):
    for dest in [d for d in (OUT_DIR, PAPER_DIR) if os.path.isdir(d)]:
        stem = "german_best_scatter_plots"
        out = os.path.join(dest, f"{stem}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
plt.close()

# ── Verification output ───────────────────────────────────────────────────────
print("\n=== VERIFIED BEST-PER-METHOD NUMBERS ===")
for method in ["nnls", "linear", "xgb"]:
    bd = best_data[method]
    print(f"  {METHOD_NAMES[method]:18s} top-{BEST[method]:3d}F: "
          f"%RMSE={bd['prmse']:.2f}%  R²={bd['r2']:.3f}  "
          f"RMSE={bd['rmse']:.2f}  MAE={bd['mae']:.2f}")
