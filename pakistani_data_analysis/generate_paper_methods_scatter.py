#!/usr/bin/env python3
"""
Generate paper_methods_nested_loo_scatter.png from pre-computed predictions JSON.
Layout: 2×3 grid (5 methods, bottom-right panel hidden).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

METHOD_ORDER = ["RF", "GBT", "XGBoost", "Stacking", "StepwiseRegression"]
METHOD_DISPLAY = {
    "RF": "Random Forest",
    "GBT": "Gradient Boosted Trees",
    "XGBoost": "XGBoost",
    "Stacking": "Stacking",
    "StepwiseRegression": "Stepwise Regression",
}
METHOD_COLORS = {
    "RF":                 "#E74C3C",
    "GBT":                "#3498DB",
    "XGBoost":            "#F18F01",
    "Stacking":           "#9B59B6",
    "StepwiseRegression": "#2ECC71",
}
LETTERS = "abcde"

with open(os.path.join(SCRIPT_DIR, "paper_methods_nested_loo_predictions.json")) as f:
    data = json.load(f)

# Global axis limits
all_vals = []
for key in data:
    all_vals.extend(data[key]["y_true"] + data[key]["y_pred"])
lo, hi = min(all_vals), max(all_vals)
pad = (hi - lo) * 0.07
ax_lo, ax_hi = lo - pad, hi + pad

fig, axes_grid = plt.subplots(2, 3, figsize=(18, 12))
fig.subplots_adjust(wspace=0.35, hspace=0.50, top=0.95, bottom=0.07,
                    left=0.08, right=0.97)
axes_flat = axes_grid.flatten()

for idx, key in enumerate(METHOD_ORDER):
    entry = data[key]
    yt    = np.array(entry["y_true"])
    yp    = np.array(entry["y_pred"])
    prmse = entry["prmse"]
    r2    = entry["r2"]
    ax    = axes_flat[idx]

    ax.scatter(yt, yp, color=METHOD_COLORS[key], alpha=0.85,
               edgecolors="white", linewidths=0.6, s=100, zorder=3)
    ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], "k--", lw=1.5, alpha=0.6, zorder=2)
    ax.set_xlim(ax_lo, ax_hi)
    ax.set_ylim(ax_lo, ax_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True AGB (t/ha)", fontsize=27, labelpad=6)
    ax.set_ylabel("Predicted AGB (t/ha)", fontsize=27, labelpad=6)
    ax.set_title(METHOD_DISPLAY[key], fontsize=22, pad=8)
    ax.grid(True, lw=0.5, alpha=0.25, ls=':')
    ax.text(0.97, 0.05,
            f"R² = {r2:.3f}\n%RMSE = {prmse:.1f}%\nN = {len(yt)}",
            transform=ax.transAxes, fontsize=22, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#AAAAAA", alpha=0.85))
    ax.text(-0.10, 1.04, f"({LETTERS[idx]})",
            transform=ax.transAxes,
            fontsize=34, fontweight="bold", va="bottom", ha="left")

# Hide unused bottom-right panel
axes_flat[-1].axis("off")

for ext in ("svg", "png"):
    out = os.path.join(SCRIPT_DIR, f"paper_methods_nested_loo_scatter.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none", pad_inches=0.15)
    print(f"Saved: {out}")
plt.close(fig)
