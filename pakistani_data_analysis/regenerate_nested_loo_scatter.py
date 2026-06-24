#!/usr/bin/env python3
"""
Regenerate literature-methods scatter figure from saved predictions.

Reads paper_methods_nested_loo_predictions.json (written by run_paper_methods_nested_loo.py)
and produces publication-ready scatter plots without re-running the expensive LOO.

Output:
  paper_methods_nested_loo_scatter.svg  (publication-ready)
  paper_methods_nested_loo_scatter.png  (backup raster)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Publication style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

METHODS = ["RF", "GBT", "XGBoost", "Stacking", "StepwiseRegression"]
METHOD_DISPLAY = {
    "RF":                  "Random Forest",
    "GBT":                 "Gradient Boosting",
    "XGBoost":             "XGBoost",
    "Stacking":            "Stacking Ensemble",
    "StepwiseRegression":  "Stepwise Regression",
}
METHOD_COLORS = {
    "RF":                 "#1f77b4",
    "GBT":                "#ff7f0e",
    "XGBoost":            "#2ca02c",
    "Stacking":           "#d62728",
    "StepwiseRegression": "#9467bd",
}
LETTERS = "abcde"


def main():
    pred_path = os.path.join(SCRIPT_DIR, "paper_methods_nested_loo_predictions.json")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"{pred_path} not found.\n"
            "Run run_paper_methods_nested_loo.py first to generate predictions."
        )

    with open(pred_path) as f:
        data = json.load(f)

    # Global axis limits
    all_vals = []
    for method in METHODS:
        if method in data:
            all_vals.extend(data[method]["y_true"] + data[method]["y_pred"])
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.07
    ax_lo, ax_hi = lo - pad, hi + pad

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    for i, method in enumerate(METHODS):
        if method not in data:
            axes[i].axis("off"); continue
        d = data[method]
        yt = np.array(d["y_true"])
        yp = np.array(d["y_pred"])
        r2    = d["r2"]
        prmse = d["prmse"]

        ax = axes[i]
        ax.scatter(yt, yp, color=METHOD_COLORS[method], alpha=0.85,
                   edgecolors="white", linewidths=0.5, s=90, zorder=3)
        ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], "k--", lw=1.5, alpha=0.6, zorder=2)
        ax.set_xlim(ax_lo, ax_hi)
        ax.set_ylim(ax_lo, ax_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True AGB (t/ha)", fontsize=16, labelpad=6)
        ax.set_ylabel("Predicted AGB (t/ha)", fontsize=16, labelpad=6)
        ax.set_title(METHOD_DISPLAY[method], fontsize=15, fontweight="bold", pad=8)
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.text(0.97, 0.05,
                f"R² = {r2:.3f}\n%RMSE = {prmse:.1f}%\nN = {len(yt)}",
                transform=ax.transAxes, fontsize=13,
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#AAAAAA", alpha=0.85))
        ax.text(-0.10, 1.04, f"({LETTERS[i]})",
                transform=ax.transAxes,
                fontsize=20, fontweight="bold", va="bottom", ha="left")

    for j in range(len(METHODS), len(axes)):
        axes[j].axis("off")

    for ext in ("svg", "png"):
        out = os.path.join(SCRIPT_DIR, f"paper_methods_nested_loo_scatter.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none", pad_inches=0.15)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
